from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# -------------------------
# Base blocks
# -------------------------
class LayerNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=-1, keepdim=True)
        v = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - m) * (self.g * torch.rsqrt(v + self.eps)) + self.b


class DenseEdgeAwareMPNNConv(nn.Module):
    """Dense message passing with max aggregation."""
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.M = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, out_dim),
            LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.U1 = nn.Linear(node_dim, out_dim)
        self.U2 = nn.Linear(out_dim, out_dim)
        self.ln = LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        # x: [B,N,Dn], edge_attr: [B,N,N,De], edge_mask: [B,N,N] bool
        B, N, _ = x.shape
        xi = x.unsqueeze(-2).expand(B, N, N, -1)
        xj = x.unsqueeze(-3).expand(B, N, N, -1)
        m_in = torch.cat([xi, xj, edge_attr], dim=-1)  # [B,N,N,2Dn+De]
        msg = self.M(m_in)                              # [B,N,N,H]
        msg = msg.masked_fill(~edge_mask.unsqueeze(-1), -1e9)
        aggr = msg.max(dim=-2).values                   # max over neighbors j
        aggr = torch.clamp(aggr, min=-1e5)
        h = self.U1(x) + self.U2(aggr)
        return F.relu(self.ln(h))


# ============================================================
#  MaGraphFeatureTransformer (Encode)
# ============================================================
class MaGraphFeatureTransformer(nn.Module):
    """
    Encode node/edge/graph(time) features into a common embedding dimension.
    Inputs (dense):
      node_features: [B,N,7]
      edge_status:   [B,N,N] int in {0..3}
      edge_weights:  [B,N,N] float
      phase:         [B,1]   scalar (step counter)
    Outputs:
      node0:   [B,N,D]
      edge0:   [B,N,N,D]
      graph0:  [B,D]
      edge_mask: [B,N,N] bool
    """
    def __init__(self, embed_dim: int, node_in_dim: int = 7, status_emb_dim: int = 8):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.status_emb = nn.Embedding(4, status_emb_dim)
        self.edge_enc = nn.Sequential(
            nn.Linear(1 + status_emb_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.graph_enc = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_status: torch.Tensor,
        edge_weights: torch.Tensor,
        phase: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node0 = self.node_enc(node_features.float())  # [B,N,D]
        st = self.status_emb(edge_status.long().clamp(0, 3))  # [B,N,N,Se]
        edge_raw = torch.cat([edge_weights.unsqueeze(-1).float(), st], dim=-1)  # [B,N,N,1+Se]
        edge0 = self.edge_enc(edge_raw)  # [B,N,N,D]
        edge_mask = edge_weights > 0
        graph0 = self.graph_enc(phase.float())  # [B,D]
        return node0, edge0, graph0, edge_mask


# ============================================================
#  MaGraphFeatureEncoderProcessor (Process)
# ============================================================
class MaGraphFeatureEncoderProcessor(nn.Module):
    """
    Process encoded features using a GNN and pooling.
    Inputs:
      node0: [B,N,D], edge0: [B,N,N,D], graph0: [B,D], edge_mask: [B,N,N]
    Outputs:
      hV: [B,N,D] node embeddings
      hG: [B,D]   graph embedding
    """
    def __init__(self, embed_dim: int, num_gnn_layers: int = 2, pooling_type: str = "mean"):
        super().__init__()
        self.pooling_type = pooling_type
        self.gnns = nn.ModuleList()
        in_dim = embed_dim
        for _ in range(num_gnn_layers):
            self.gnns.append(DenseEdgeAwareMPNNConv(in_dim, embed_dim, embed_dim))
            in_dim = embed_dim

    def forward(
        self,
        node0: torch.Tensor,
        edge0: torch.Tensor,
        graph0: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = node0
        for gnn in self.gnns:
            x = gnn(x, edge0, edge_mask)  # [B,N,D]

        if self.pooling_type == "mean":
            hG = x.mean(dim=-2)
        elif self.pooling_type == "max":
            hG = x.max(dim=-2).values
        elif self.pooling_type == "sum":
            hG = x.sum(dim=-2)
        else:
            raise ValueError(f"Unknown pooling_type={self.pooling_type}")

        # fuse graph feature (time embedding)
        hG = hG + graph0
        return x, hG


# ============================================================
# Shared trunk registry (actor/critic share the same instance)
# ============================================================
_SHARED_TRUNKS: Dict[str, "SharedEncodeProcessTrunk"] = {}


def _make_trunk_key(agent_group: str, model_index: int, device: torch.device) -> str:
    # include device to avoid accidental cross-device module sharing
    return f"{agent_group}::model{model_index}::{str(device)}"


class SharedEncodeProcessTrunk(nn.Module):
    """
    Shared Encode+Process trunk:
      node/edge/phase -> (hV, hG)
    """
    def __init__(
        self,
        embed_dim: int = 64,
        num_gnn_layers: int = 2,
        status_emb_dim: int = 8,
        pooling_type: str = "mean",
    ):
        super().__init__()
        self.transformer = MaGraphFeatureTransformer(embed_dim=embed_dim, status_emb_dim=status_emb_dim)
        self.processor = MaGraphFeatureEncoderProcessor(embed_dim=embed_dim, num_gnn_layers=num_gnn_layers, pooling_type=pooling_type)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_status: torch.Tensor,
        edge_weights: torch.Tensor,
        phase: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node0, edge0, graph0, edge_mask = self.transformer(node_features, edge_status, edge_weights, phase)
        hV, hG = self.processor(node0, edge0, graph0, edge_mask)
        return hV, hG


def get_or_create_shared_trunk(
    agent_group: str,
    model_index: int,
    device: torch.device,
    embed_dim: int,
    num_gnn_layers: int,
    status_emb_dim: int,
    pooling_type: str,
) -> SharedEncodeProcessTrunk:
    key = _make_trunk_key(agent_group, model_index, device)
    if key not in _SHARED_TRUNKS:
        trunk = SharedEncodeProcessTrunk(
            embed_dim=embed_dim,
            num_gnn_layers=num_gnn_layers,
            status_emb_dim=status_emb_dim,
            pooling_type=pooling_type,
        ).to(device)
        _SHARED_TRUNKS[key] = trunk
    return _SHARED_TRUNKS[key]