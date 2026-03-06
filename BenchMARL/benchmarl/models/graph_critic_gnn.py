from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tensordict import TensorDictBase

from benchmarl.models.common import Model, ModelConfig


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
        B, N, _ = x.shape
        xi = x.unsqueeze(-2).expand(B, N, N, -1)
        xj = x.unsqueeze(-3).expand(B, N, N, -1)
        m_in = torch.cat([xi, xj, edge_attr], dim=-1)
        msg = self.M(m_in)
        msg = msg.masked_fill(~edge_mask.unsqueeze(-1), -1e9)
        aggr = msg.max(dim=-2).values
        aggr = torch.clamp(aggr, min=-1e5)
        h = self.U1(x) + self.U2(aggr)
        return F.relu(self.ln(h))


def _flatten_bt(t: torch.Tensor, keep_last: int) -> Tuple[torch.Tensor, int]:
    lead = t.shape[:-keep_last]
    E = 1
    for s in lead:
        E *= int(s)
    return t.reshape((E,) + t.shape[-keep_last:]), E


class GraphCriticGNN(Model):
    """
    输入 keys（Level-1）：
      ("agents","node_features"), ("agents","edge_status"), ("agents","edge_weights"), ("agents","phase")
    输出：
      "state_value" : [E,1]（share_param_critic=True 时由 MAPPO expand 到 (group,"state_value")）
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        num_gnn_layers: int = 2,
        status_emb_dim: int = 8,
        gnn_hidden_dim: int = 64,
        use_phase: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_phase = use_phase
        self.status_emb = nn.Embedding(4, status_emb_dim).to(self.device)
        self.phase_proj = nn.Linear(1, gnn_hidden_dim).to(self.device) if use_phase else None

        self.global_node_in_dim = 7
        self.edge_in_dim = 1 + status_emb_dim

        self.gnns = nn.ModuleList()
        in_dim = self.global_node_in_dim
        for _ in range(num_gnn_layers):
            self.gnns.append(DenseEdgeAwareMPNNConv(in_dim, self.edge_in_dim, gnn_hidden_dim).to(self.device))
            in_dim = gnn_hidden_dim

        head_in = gnn_hidden_dim * 2 if use_phase else gnn_hidden_dim
        self.value_head = nn.Sequential(
            nn.Linear(head_in, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, 1),
        ).to(self.device)

    @staticmethod
    def _build_global_node_features(node_features: torch.Tensor) -> torch.Tensor:
        dest = node_features[:, 0, :, 0]
        visited = node_features[:, 0, :, 1]
        opt = node_features[:, 0, :, 2]
        pess = node_features[:, 0, :, 3]
        self_pos = (node_features[:, :, :, 4] > 0.5).float()
        density = node_features[:, :, :, 5]
        terminated = node_features[:, :, :, 6]

        occupancy = self_pos.sum(dim=1)
        density_mean = density.mean(dim=1)
        term_count = terminated.sum(dim=1)

        return torch.stack([dest, visited, opt, pess, occupancy, density_mean, term_count], dim=-1)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        node_features = tensordict.get(("agents", "node_features")).to(self.device)
        edge_status = tensordict.get(("agents", "edge_status")).to(self.device)
        edge_weights = tensordict.get(("agents", "edge_weights")).to(self.device)
        phase = tensordict.get(("agents", "phase")).to(self.device) if self.use_phase else None

        # 原始 batch 形状（例如 [B,T]）
        batch_shape = tensordict.batch_size
        M = 1
        for s in batch_shape:
            M *= int(s)

        # 内部 flatten 计算：[..., A, N, 7] -> [M, A, N, 7]
        node_features = node_features.reshape(M, *node_features.shape[-3:])  # [M,A,N,7]
        edge_status = edge_status.reshape(M, *edge_status.shape[-3:])        # [M,A,N,N]
        edge_weights = edge_weights.reshape(M, *edge_weights.shape[-3:])     # [M,A,N,N]
        if phase is not None:
            phase = phase.reshape(M, *phase.shape[-2:])                      # [M,A,1]

        M, A, N, _ = node_features.shape

        edge_status_g = edge_status[:, 0]    # [M,N,N]
        edge_weights_g = edge_weights[:, 0]  # [M,N,N]

        x = self._build_global_node_features(node_features)  # [M,N,7]
        status_emb = self.status_emb(edge_status_g.long().clamp(0, 3))
        edge_attr = torch.cat([edge_weights_g.unsqueeze(-1), status_emb], dim=-1)
        edge_mask = edge_weights_g > 0

        for gnn in self.gnns:
            x = gnn(x, edge_attr, edge_mask)

        h_graph = x.mean(dim=-2)  # [M,H]
        if self.use_phase:
            phase_g = phase[:, 0].float()         # [M,1]
            phase_h = self.phase_proj(phase_g)    # [M,H]
            h = torch.cat([h_graph, phase_h], dim=-1)
        else:
            h = h_graph

        v = self.value_head(h)  # [M,1]

        # 写回前 reshape 成原 batch：[*batch_shape, 1]
        v = v.reshape(*batch_shape, 1)
        tensordict.set(self.out_key, v)
        return tensordict

@dataclass
class GraphCriticGNNConfig(ModelConfig):
    node_features: int = MISSING
    edge_features: int = MISSING
    num_gnn_layers: int = 2
    status_emb_dim: int = 8
    gnn_hidden_dim: int = 64
    use_phase: bool = True

    @staticmethod
    def associated_class():
        return GraphCriticGNN