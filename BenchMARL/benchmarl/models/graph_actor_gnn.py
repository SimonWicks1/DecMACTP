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
    """Dense MPNN，用 edge_mask 表示是否存在边（edge_weights>0）"""
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
        # x: [B,N,D], edge_attr: [B,N,N,De], edge_mask: [B,N,N] bool
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
    """
    把前面的 batch/time 维合并成一维 E。
    keep_last: 保留末尾 keep_last 个维度（例如 A,N,7 -> keep_last=3）
    """
    if t.dim() < keep_last + 1:
        raise ValueError(f"Tensor rank {t.dim()} too small for keep_last={keep_last}")
    lead = t.shape[:-keep_last]
    E = 1
    for s in lead:
        E *= int(s)
    new_shape = (E,) + t.shape[-keep_last:]
    return t.reshape(new_shape), E

class GraphActorGNN(Model):
    """
    严格去中心化的图 Actor 网络。
    通过合并 M 和 A 维度，确保智能体间绝对的数据隔离。
    """
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        num_gnn_layers: int = 2,
        status_emb_dim: int = 8,
        gnn_hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not self.output_has_agent_dim:
            raise ValueError("GraphActorGNN must output per-agent logits.")

        self.status_emb = nn.Embedding(4, status_emb_dim).to(self.device)
        self.phase_proj = nn.Linear(1, gnn_hidden_dim).to(self.device)

        # 节点特征的原始输入就是 7 维，不再进行跨智能体聚合
        self.local_node_in_dim = 7
        self.edge_in_dim = 1 + status_emb_dim

        self.gnns = nn.ModuleList()
        in_dim = self.local_node_in_dim
        for _ in range(num_gnn_layers):
            self.gnns.append(DenseEdgeAwareMPNNConv(in_dim, self.edge_in_dim, gnn_hidden_dim).to(self.device))
            in_dim = gnn_hidden_dim

        self.key_proj = nn.Linear(gnn_hidden_dim, gnn_hidden_dim).to(self.device)

        self.query_proj = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 3, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
        ).to(self.device)

        self.terminate_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, 1),
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

        # 【核心修改 1】：合并批次与智能体维度 B = M * A
        B = M * A
        
        # 将所有张量 reshape 为 [B, N, ...] 的形式
        node_features = node_features.reshape(B, N, -1)     # [B, N, 7]
        edge_status = edge_status.reshape(B, N, N)          # [B, N, N]
        edge_weights = edge_weights.reshape(B, N, N)        # [B, N, N]
        phase = phase.reshape(B, 1)                         # [B, 1]
        action_mask = action_mask.reshape(B, N + 1)         # [B, N+1]

        # 【核心修改 2】：直接使用局部节点特征，废除跨智能体聚合逻辑
        x = node_features # 原始输入已经是 [dest, visited, opt, pess, self_pos, density, terminated]

        # 【核心修改 3】：每个智能体使用各自的 edge_status 和 edge_weights (解决泄露点 A)
        status_emb = self.status_emb(edge_status.long().clamp(0, 3))  # [B, N, N, Se]
        edge_attr = torch.cat([edge_weights.unsqueeze(-1), status_emb], dim=-1)  # [B, N, N, 1+Se]
        edge_mask = edge_weights > 0

        # GNN 独立消息传递
        for gnn in self.gnns:
            x = gnn(x, edge_attr, edge_mask)  # [B, N, H]

        h_nodes = x
        h_graph = h_nodes.mean(dim=-2)  # [B, H]

        # 获取自身位置表示 (self_pos 位于特征的第 4 维)
        self_pos = (node_features[:, :, 4] > 0.5).float()  # [B, N]
        h_loc = torch.einsum("bn,bnh->bh", self_pos, h_nodes)  # [B, H]

        phase_h = self.phase_proj(phase.float())  # [B, H]

        # Query 拼接
        q_in = torch.cat([h_graph, h_loc, phase_h], dim=-1)  # [B, 3H]
        q = self.query_proj(q_in)  # [B, H]
        k = self.key_proj(h_nodes)  # [B, N, H]

        # 计算 Logits
        logits_nodes = torch.einsum("bh,bnh->bn", q, k) / (k.shape[-1] ** 0.5)  # [B, N]
        logits_term = self.terminate_head(q).squeeze(-1)  # [B]
        logits = torch.cat([logits_nodes, logits_term.unsqueeze(-1)], dim=-1)  # [B, N+1]

        # 动作掩码
        logits = logits.masked_fill(~action_mask.bool(), -1e9)

        # 【核心修改 4】：将合并后的维度拆解回 MAPPO 需要的 [*batch_shape, A, N+1]
        logits = logits.reshape(*batch_shape, A, N + 1)
        tensordict.set(self.out_key, logits)
        return tensordict

@dataclass
class GraphActorGNNConfig(ModelConfig):
    node_features: int = MISSING
    edge_features: int = MISSING
    num_gnn_layers: int = 2
    status_emb_dim: int = 8
    gnn_hidden_dim: int = 64

    @staticmethod
    def associated_class():
        return GraphActorGNN