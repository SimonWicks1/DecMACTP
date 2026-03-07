# benchmarl/models/ignarl_critic_gnn.py
from __future__ import annotations

from dataclasses import dataclass, MISSING

import torch
from torch import nn
from tensordict import TensorDictBase

from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.ignarl_shared_trunk import get_or_create_shared_trunk


class IgnarlCriticGNN(Model):
    """
    IGNARL decentralized critic V_i(o_i):
      Shared trunk (Encode+Process) -> (hV, hG)
      Value head uses hG (and optionally h_loc) to estimate value per-agent.

    IMPORTANT:
      To avoid autograd version errors due to shared trunk being stepped by another optimizer
      in the same iteration, we detach trunk features by default (detach_trunk=True).
    """
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        num_gnn_layers: int = 2,
        status_emb_dim: int = 8,
        gnn_hidden_dim: int = 64,
        pooling_type: str = "mean",
        use_loc: bool = True,
        detach_trunk: bool = True,   # ✅ new
        **kwargs,
    ):
        super().__init__(**kwargs)

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

        self.use_loc = bool(use_loc)
        self.detach_trunk = bool(detach_trunk)

        head_in = gnn_hidden_dim * (2 if self.use_loc else 1)
        self.value_head = nn.Sequential(
            nn.Linear(head_in, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, 1),
        ).to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        node_features = tensordict.get(("agents", "node_features")).to(self.device)
        edge_status = tensordict.get(("agents", "edge_status")).to(self.device)
        edge_weights = tensordict.get(("agents", "edge_weights")).to(self.device)
        phase = tensordict.get(("agents", "phase")).to(self.device)

        batch_shape = tensordict.batch_size
        M = 1
        for s in batch_shape:
            M *= int(s)

        A = node_features.shape[-3]
        N = node_features.shape[-2]
        B = M * A

        node_features = node_features.reshape(B, N, -1)
        edge_status = edge_status.reshape(B, N, N)
        edge_weights = edge_weights.reshape(B, N, N)
        phase = phase.reshape(B, 1)

        self_pos = (node_features[:, :, 4] > 0.5).float()  # [B,N]

        # ✅ shared Encode+Process, but optionally detach to avoid version conflicts
        if self.detach_trunk:
            with torch.no_grad():
                hV, hG = self.trunk(node_features, edge_status, edge_weights, phase)
            # no_grad already detaches; keep explicit for safety
            hV = hV.detach()
            hG = hG.detach()
        else:
            hV, hG = self.trunk(node_features, edge_status, edge_weights, phase)

        if self.use_loc:
            h_loc = torch.einsum("bn,bnd->bd", self_pos, hV)  # [B,D]
            h_in = torch.cat([hG, h_loc], dim=-1)             # [B,2D]
        else:
            h_in = hG

        v = self.value_head(h_in)                              # [B,1]
        v = v.reshape(*batch_shape, A, 1)
        tensordict.set(self.out_key, v)                        # ('agents','state_value')
        return tensordict


@dataclass
class IgnarlCriticGNNConfig(ModelConfig):
    node_features: int = MISSING
    edge_features: int = MISSING
    num_gnn_layers: int = 2
    status_emb_dim: int = 8
    gnn_hidden_dim: int = 64
    pooling_type: str = "mean"
    use_loc: bool = True
    detach_trunk: bool = True   # ✅ new

    @staticmethod
    def associated_class():
        return IgnarlCriticGNN