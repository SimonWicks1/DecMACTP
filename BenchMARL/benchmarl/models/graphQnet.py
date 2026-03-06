import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass, MISSING
from tensordict import TensorDictBase
from benchmarl.models.common import Model, ModelConfig

class LayerNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps
    def forward(self, x):
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

    def forward(self, x, edge_attr, edge_mask):
        # x: [M,N,D], edge_attr: [M,N,N,De], edge_mask: [M,N,N]
        M, N, _ = x.shape
        xi = x.unsqueeze(-2).expand(M, N, N, -1)
        xj = x.unsqueeze(-3).expand(M, N, N, -1)
        m_in = torch.cat([xi, xj, edge_attr], dim=-1)
        msg = self.M(m_in)
        msg = msg.masked_fill(~edge_mask.unsqueeze(-1), -1e9)
        aggr = msg.max(dim=-2).values
        aggr = torch.clamp(aggr, min=-1e5)
        h = self.U1(x) + self.U2(aggr)
        return F.relu(self.ln(h))

class GraphQNet(Model):
    """
    Outputs (group,'action_value') for IQL/VDN.
    Assumes env provides:
      ("agents","node_features"): [...,A,N,7]  (dest, visited, opt, pess, self_pos, density, terminated)
      ("agents","edge_status"):  [...,A,N,N]
      ("agents","edge_weights"): [...,A,N,N]
      ("agents","phase"):        [...,A,1]
      ("agents","action_mask"):  [...,A,N+1]  (optional but recommended)
    """
    def __init__(self, node_features: int, edge_features: int,
                 num_gnn_layers: int = 2, status_emb_dim: int = 8, hidden_dim: int = 64,
                 use_action_mask: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_action_mask = use_action_mask
        self.status_emb = nn.Embedding(4, status_emb_dim).to(self.device)
        self.phase_proj = nn.Linear(1, hidden_dim).to(self.device)

        self.global_node_in_dim = 7
        self.edge_in_dim = 1 + status_emb_dim

        self.gnns = nn.ModuleList()
        in_dim = self.global_node_in_dim
        for _ in range(num_gnn_layers):
            self.gnns.append(DenseEdgeAwareMPNNConv(in_dim, self.edge_in_dim, hidden_dim).to(self.device))
            in_dim = hidden_dim

        self.key_proj = nn.Linear(hidden_dim, hidden_dim).to(self.device)
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(self.device)

        self.terminate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

    @staticmethod
    def _build_global_node_features(node_features: torch.Tensor) -> torch.Tensor:
        # node_features: [M,A,N,7]
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
        node_features = tensordict.get(("agents","node_features")).to(self.device)
        edge_status = tensordict.get(("agents","edge_status")).to(self.device)
        edge_weights = tensordict.get(("agents","edge_weights")).to(self.device)
        phase = tensordict.get(("agents","phase")).to(self.device)
        action_mask = tensordict.get(("agents","action_mask")) if self.use_action_mask else None
        if action_mask is not None:
            action_mask = action_mask.to(self.device)

        batch_shape = tensordict.batch_size  # e.g. [B,T]
        M = 1
        for s in batch_shape:
            M *= int(s)

        # reshape to [M,A,...]
        node_features = node_features.reshape(M, *node_features.shape[-3:])  # [M,A,N,7]
        edge_status = edge_status.reshape(M, *edge_status.shape[-3:])        # [M,A,N,N]
        edge_weights = edge_weights.reshape(M, *edge_weights.shape[-3:])     # [M,A,N,N]
        phase = phase.reshape(M, *phase.shape[-2:])                          # [M,A,1]
        if action_mask is not None:
            action_mask = action_mask.reshape(M, *action_mask.shape[-2:])    # [M,A,N+1]

        M, A, N, _ = node_features.shape

        # shared graph (agent0 replica)
        edge_status_g = edge_status[:,0]
        edge_weights_g = edge_weights[:,0]

        x = self._build_global_node_features(node_features)  # [M,N,7]
        status_emb = self.status_emb(edge_status_g.long().clamp(0,3))
        edge_attr = torch.cat([edge_weights_g.unsqueeze(-1), status_emb], dim=-1)
        edge_mask = edge_weights_g > 0

        for gnn in self.gnns:
            x = gnn(x, edge_attr, edge_mask)

        h_nodes = x                  # [M,N,H]
        h_graph = h_nodes.mean(-2)   # [M,H]

        self_pos = (node_features[:,:,:,4] > 0.5).float()      # [M,A,N]
        h_loc = torch.einsum("man,mnh->mah", self_pos, h_nodes)  # [M,A,H]
        phase_h = self.phase_proj(phase.float())               # [M,A,H]

        q_in = torch.cat([h_graph.unsqueeze(1).expand(M,A,-1), h_loc, phase_h], dim=-1)  # [M,A,3H]
        q = self.query_proj(q_in)  # [M,A,H]
        k = self.key_proj(h_nodes) # [M,N,H]

        q_nodes = torch.einsum("mah,mnh->man", q, k) / (k.shape[-1] ** 0.5)  # [M,A,N]
        q_term = self.terminate_head(q).squeeze(-1)                          # [M,A]
        q_all = torch.cat([q_nodes, q_term.unsqueeze(-1)], dim=-1)           # [M,A,N+1]

        if action_mask is not None:
            q_all = q_all.masked_fill(~action_mask.bool(), -1e9)

        q_all = q_all.reshape(*batch_shape, A, N+1)  # restore batch
        tensordict.set(self.out_key, q_all)
        return tensordict

@dataclass
class GraphQNetConfig(ModelConfig):
    node_features: int = MISSING
    edge_features: int = MISSING
    num_gnn_layers: int = 2
    status_emb_dim: int = 8
    hidden_dim: int = 64
    use_action_mask: bool = True

    @staticmethod
    def associated_class():
        return GraphQNet