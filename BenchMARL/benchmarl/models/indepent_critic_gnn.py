from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tensordict import TensorDictBase

from benchmarl.models.common import Model, ModelConfig

# 复用基础 GNN 组件
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


class IndependentGraphCriticGNN(Model):
    """
    专为 IPPO 设计的去中心化图 Critic 网络。
    每个智能体基于自身局部观测，独立评估预期价值 [..., A, 1]。
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

        # 局部节点特征直接使用环境提供的 7 维输入
        self.local_node_in_dim = 7
        self.edge_in_dim = 1 + status_emb_dim

        self.gnns = nn.ModuleList()
        in_dim = self.local_node_in_dim
        for _ in range(num_gnn_layers):
            self.gnns.append(DenseEdgeAwareMPNNConv(in_dim, self.edge_in_dim, gnn_hidden_dim).to(self.device))
            in_dim = gnn_hidden_dim

        # 价值头的输入：全局图表征 (h_graph) + 自身节点表征 (h_loc) + [可选的阶段信息]
        head_in = (gnn_hidden_dim * 2) + (gnn_hidden_dim if use_phase else 0)
        self.value_head = nn.Sequential(
            nn.Linear(head_in, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, 1),
        ).to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # 获取输入张量
        node_features = tensordict.get(("agents", "node_features")).to(self.device)
        edge_status = tensordict.get(("agents", "edge_status")).to(self.device)
        edge_weights = tensordict.get(("agents", "edge_weights")).to(self.device)
        phase = tensordict.get(("agents", "phase")).to(self.device) if self.use_phase else None

        batch_shape = tensordict.batch_size
        M = 1
        for s in batch_shape:
            M *= int(s)

        A = node_features.shape[-3]
        N = node_features.shape[-2]

        # 【核心隔离策略】：合并批次与智能体维度 B = M * A
        B = M * A
        
        node_features = node_features.reshape(B, N, -1)     # [B, N, 7]
        edge_status = edge_status.reshape(B, N, N)          # [B, N, N]
        edge_weights = edge_weights.reshape(B, N, N)        # [B, N, N]
        if phase is not None:
            phase = phase.reshape(B, 1)                     # [B, 1]

        # 1. 节点与边特征处理 (完全局部化)
        x = node_features
        status_emb = self.status_emb(edge_status.long().clamp(0, 3))  # [B, N, N, Se]
        edge_attr = torch.cat([edge_weights.unsqueeze(-1), status_emb], dim=-1)  # [B, N, N, 1+Se]
        edge_mask = edge_weights > 0

        # 2. 独立图神经网络传播
        for gnn in self.gnns:
            x = gnn(x, edge_attr, edge_mask)  # [B, N, H]
        h_nodes = x

        # 3. 提取价值评估所需特征
        # 3.1 局部图结构的池化表征
        h_graph = h_nodes.mean(dim=-2)  # [B, H]

        # 3.2 智能体所处当前节点的表征 (精确定位局部价值)
        self_pos = (node_features[:, :, 4] > 0.5).float()  # [B, N]
        h_loc = torch.einsum("bn,bnh->bh", self_pos, h_nodes)  # [B, H]

        # 4. 特征拼接与价值评估
        if self.use_phase:
            phase_h = self.phase_proj(phase.float())  # [B, H]
            h_combine = torch.cat([h_graph, h_loc, phase_h], dim=-1)  # [B, 3H]
        else:
            h_combine = torch.cat([h_graph, h_loc], dim=-1)  # [B, 2H]

        v = self.value_head(h_combine)  # [B, 1]

        # 5. 重组回标准 BenchMARL 维度输出 [*batch_shape, A, 1]
        v = v.reshape(*batch_shape, A, 1)
        tensordict.set(self.out_key, v)
        
        print(f"Critic output shape: {v.shape} (should be [*batch_shape, A, 1])")
        print(f"out_key: {self.out_key}")
        return tensordict

@dataclass
class IndependentGraphCriticGNNConfig(ModelConfig):
    node_features: int = MISSING
    edge_features: int = MISSING
    num_gnn_layers: int = 2
    status_emb_dim: int = 8
    gnn_hidden_dim: int = 64
    use_phase: bool = True

    @staticmethod
    def associated_class():
        return IndependentGraphCriticGNN