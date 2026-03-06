from __future__ import annotations

from dataclasses import dataclass, MISSING

import torch
from torch import nn
from tensordict import TensorDictBase

from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.ignarl_shared_trunk import get_or_create_shared_trunk


class MaNodeSimilarityMatchAgg(nn.Module):
    """
    Act stage: proto-action/query and distance-based matching.
    logits(v) = -||k_v - q||_2 / tau
    terminate logits is produced from q and also scaled by /tau.
    """
    def __init__(self, embed_dim: int, temperature: float = 2.0, eps_dist: float = 1e-8):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.eps_dist = float(eps_dist)

        self.query_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.term_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, hV: torch.Tensor, hG: torch.Tensor, self_pos: torch.Tensor) -> torch.Tensor:
        # hV: [B,N,D], hG: [B,D], self_pos: [B,N]
        h_loc = torch.einsum("bn,bnd->bd", self_pos.float(), hV)  # [B,D]
        q = self.query_proj(torch.cat([hG, h_loc], dim=-1))       # [B,D]

        diff = hV - q.unsqueeze(1)  # [B,N,D]
        dist = torch.sqrt((diff * diff).sum(dim=-1) + self.eps_dist)  # [B,N]
        logits_nodes = -dist / self.temperature  # [B,N]

        logits_term = self.term_head(q).squeeze(-1) / self.temperature  # [B]
        logits = torch.cat([logits_nodes, logits_term.unsqueeze(-1)], dim=-1)  # [B,N+1]
        return logits


class IgnarlActorGNN(Model):
    """
    IGNARL actor:
      Shared trunk (Encode+Process) -> (hV, hG)
      Act head: distance-based matching over nodes + terminate head
    """
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not self.output_has_agent_dim:
            raise ValueError("IgnarlActorGNN must output per-agent logits.")

        # shared trunk key inputs come from Model base
        self._agent_group = kwargs.get("agent_group", "agents")
        self._model_index = int(kwargs.get("model_index", 0))

        self.trunk = get_or_create_shared_trunk(
            agent_group=self._agent_group,
            model_index=self._model_index,
            device=self.device,
            embed_dim=gnn_hidden_dim,
            num_gnn_layers=num_gnn_layers,
            status_emb_dim=status_emb_dim,
            pooling_type=pooling_type,
        )

        self.act_head = MaNodeSimilarityMatchAgg(
            embed_dim=gnn_hidden_dim,
            temperature=temperature,
            eps_dist=eps_dist,
        ).to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        node_features = tensordict.get(("agents", "node_features")).to(self.device)
        edge_status = tensordict.get(("agents", "edge_status")).to(self.device)
        edge_weights = tensordict.get(("agents", "edge_weights")).to(self.device)
        phase = tensordict.get(("agents", "phase")).to(self.device)
        action_mask = tensordict.get(("agents", "action_mask")).to(self.device)

        batch_shape = tensordict.batch_size
        M = 1
        for s in batch_shape:
            M *= int(s)

        A = node_features.shape[-3]
        N = node_features.shape[-2]
        B = M * A

        # flatten (batch_dims * A) -> B for strict independence
        node_features = node_features.reshape(B, N, -1)
        edge_status = edge_status.reshape(B, N, N)
        edge_weights = edge_weights.reshape(B, N, N)
        phase = phase.reshape(B, 1)
        action_mask = action_mask.reshape(B, N + 1)

        self_pos = (node_features[:, :, 4] > 0.5).float()  # [B,N]

        # shared Encode+Process
        hV, hG = self.trunk(node_features, edge_status, edge_weights, phase)

        # Act
        logits = self.act_head(hV, hG, self_pos)  # [B,N+1]
        logits = logits.masked_fill(~action_mask.bool(), -1e9)

        logits = logits.reshape(*batch_shape, A, N + 1)
        tensordict.set(self.out_key, logits)
        return tensordict


@dataclass
class IgnarlActorGNNConfig(ModelConfig):
    node_features: int = MISSING
    edge_features: int = MISSING
    num_gnn_layers: int = 2
    status_emb_dim: int = 8
    gnn_hidden_dim: int = 64
    pooling_type: str = "mean"
    temperature: float = 2.0
    eps_dist: float = 1e-8

    @staticmethod
    def associated_class():
        return IgnarlActorGNN