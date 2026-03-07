import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

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
    
class BaseGraphTrunk(nn.Module):
    """
    统一的图特征编码主干。
    """
    def __init__(
        self,
        node_in_dim: int = 7,
        status_emb_dim: int = 8,
        gnn_hidden_dim: int = 64,
        num_gnn_layers: int = 2,
    ):
        super().__init__()
        self.status_emb = nn.Embedding(4, status_emb_dim)
        self.edge_in_dim = 1 + status_emb_dim
        
        self.gnns = nn.ModuleList()
        in_dim = node_in_dim
        for _ in range(num_gnn_layers):
            self.gnns.append(DenseEdgeAwareMPNNConv(in_dim, self.edge_in_dim, gnn_hidden_dim))
            in_dim = gnn_hidden_dim

    def forward(
        self,
        node_features: torch.Tensor,
        edge_status: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        status_emb = self.status_emb(edge_status.long().clamp(0, 3))  
        edge_attr = torch.cat([edge_weights.unsqueeze(-1), status_emb], dim=-1)  
        edge_mask = edge_weights > 0

        x = node_features
        for gnn in self.gnns:
            x = gnn(x, edge_attr, edge_mask)  
            
        h_nodes = x
        h_graph = h_nodes.mean(dim=-2)  
        
        return h_nodes, h_graph

# ============================================================
# 共享主干注册表 (用于实现 Actor 和 Critic 的物理参数共享)
# ============================================================
_SHARED_GRAPH_TRUNKS: dict[str, BaseGraphTrunk] = {}

def _make_graph_trunk_key(agent_group: str, model_index: int, device: torch.device) -> str:
    return f"{agent_group}::model{model_index}::{str(device)}"

def get_or_create_graph_trunk(
    agent_group: str,
    model_index: int,
    device: torch.device,
    node_in_dim: int,
    status_emb_dim: int,
    gnn_hidden_dim: int,
    num_gnn_layers: int,
) -> BaseGraphTrunk:
    key = _make_graph_trunk_key(agent_group, model_index, device)
    if key not in _SHARED_GRAPH_TRUNKS:
        trunk = BaseGraphTrunk(
            node_in_dim=node_in_dim,
            status_emb_dim=status_emb_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gnn_layers=num_gnn_layers,
        ).to(device)
        _SHARED_GRAPH_TRUNKS[key] = trunk
    return _SHARED_GRAPH_TRUNKS[key]