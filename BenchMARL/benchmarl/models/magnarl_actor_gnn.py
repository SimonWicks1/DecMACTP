from __future__ import annotations

from dataclasses import dataclass, MISSING

import torch
from torch import nn
from tensordict import TensorDictBase

from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.magnarl_shared_trunk import get_or_create_shared_trunk


class MaNodeSimilarityMatchAggComm(nn.Module):
    """
    GNARL-style action head with coordination-aware query and graph-feedback bias.
    q_i = f(hG_i, hLoc_i, uComm_i)
    logits(v) = sim(hV_i(v), q_i) + coord_bias(v)
    """

    def __init__(self, embed_dim: int, temperature: float = 2.0, eps_dist: float = 1e-8):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.eps_dist = float(eps_dist)
        self.query_proj = nn.Linear(embed_dim * 3, embed_dim)
        self.node_bias = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.term_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, hV, hG, hComm, self_pos):
        h_loc = torch.einsum("ban,band->bad", self_pos.float(), hV)
        q = self.query_proj(torch.cat([hG, h_loc, hComm], dim=-1))
        diff = hV - q.unsqueeze(-2)
        dist = torch.sqrt((diff * diff).sum(dim=-1) + self.eps_dist)
        logits_nodes = -dist / self.temperature
        coord_bias = self.node_bias(torch.cat([hV, hComm.unsqueeze(-2).expand_as(hV)], dim=-1)).squeeze(-1)
        logits_nodes = logits_nodes + coord_bias
        logits_term = self.term_head(torch.cat([q, hComm], dim=-1)).squeeze(-1) / self.temperature
        return torch.cat([logits_nodes, logits_term.unsqueeze(-1)], dim=-1)


class MagnarlActorGNN(Model):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        num_gnn_layers: int = 2,
        status_emb_dim: int = 8,
        gnn_hidden_dim: int = 64,
        pooling_type: str = "mean",
        temperature: float = 2.0,
        eps_dist: float = 1e-8,
        comm_num_heads: int = 4,
        comm_top_k: int = 4,
        export_comm_stats: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not self.output_has_agent_dim:
            raise ValueError("MagnarlActorGNN must output per-agent logits.")

        self._agent_group = kwargs.get("agent_group", "agents")
        self._model_index = int(kwargs.get("model_index", 0))
        self.export_comm_stats = bool(export_comm_stats)

        self.trunk = get_or_create_shared_trunk(
            agent_group=self._agent_group,
            model_index=self._model_index,
            device=self.device,
            embed_dim=gnn_hidden_dim,
            num_gnn_layers=num_gnn_layers,
            status_emb_dim=status_emb_dim,
            pooling_type=pooling_type,
            comm_num_heads=comm_num_heads,
            comm_top_k=comm_top_k,
        )
        self.act_head = MaNodeSimilarityMatchAggComm(
            embed_dim=gnn_hidden_dim,
            temperature=temperature,
            eps_dist=eps_dist,
        ).to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        group = self._agent_group
        node_features = tensordict.get((group, "node_features")).to(self.device)
        edge_status = tensordict.get((group, "edge_status")).to(self.device)
        edge_weights = tensordict.get((group, "edge_weights")).to(self.device)
        phase = tensordict.get((group, "phase")).to(self.device)
        action_mask = tensordict.get((group, "action_mask")).to(self.device)

        batch_shape = tensordict.batch_size
        flat_batch = 1
        for s in batch_shape:
            flat_batch *= int(s)

        A = node_features.shape[-3]
        N = node_features.shape[-2]

        node_features = node_features.reshape(flat_batch, A, N, -1)
        edge_status = edge_status.reshape(flat_batch, A, N, N)
        edge_weights = edge_weights.reshape(flat_batch, A, N, N)
        phase = phase.reshape(flat_batch, A, -1)
        action_mask = action_mask.reshape(flat_batch, A, N + 1)

        self_pos = (node_features[..., 4] > 0.5).float()
        # hV, hG, hComm, aux = self.trunk(node_features, edge_status, edge_weights, phase)
        hV, hG, hComm, aux, diag = self.trunk(
            node_features=node_features,
            edge_status=edge_status,
            edge_weights=edge_weights,
            phase=phase,
        )

        # 写入当前 batch，供 callback 读取
        for k, v in diag.items():
            if not torch.is_tensor(v):
                v = torch.tensor(v, device=self.device, dtype=torch.float32)
            v = v.detach().float()
            # Broadcast scalar/reduced diag values to [*batch_shape, A, 1]
            target = torch.full(
                (*batch_shape, A, 1),
                fill_value=v.mean().item(),  # collapse to scalar regardless of shape
                dtype=torch.float32,
                device=self.device,
            )
            tensordict.set((group, "diag", k), target)
        logits = self.act_head(hV, hG, hComm, self_pos)
        logits = logits.masked_fill(~action_mask.bool(), -1e9)
        logits = logits.reshape(*batch_shape, A, N + 1)
        tensordict.set(self.out_key, logits)

        if self.export_comm_stats:
            comm_degree = aux["adj"].float().sum(dim=-1, keepdim=True)
            tensordict.set((group, "comm_degree"), comm_degree.reshape(*batch_shape, A, 1))
            tensordict.set((group, "comm_density"), (comm_degree / max(1, A - 1)).reshape(*batch_shape, A, 1))
            for name in ["coupling_prox", "coupling_goal", "coupling_frontier", "coupling_region", "coupling_risk"]:
                tensordict.set((group, name + "_mean"), aux[name].mean(dim=-1, keepdim=True).reshape(*batch_shape, A, 1))
        return tensordict


@dataclass
class MagnarlActorGNNConfig(ModelConfig):
    node_features: int = MISSING
    edge_features: int = MISSING
    num_gnn_layers: int = 2
    status_emb_dim: int = 8
    gnn_hidden_dim: int = 64
    pooling_type: str = "mean"
    temperature: float = 2.0
    eps_dist: float = 1e-8
    comm_num_heads: int = 4
    comm_top_k: int = 4
    export_comm_stats: bool = True

    @staticmethod
    def associated_class():
        return MagnarlActorGNN