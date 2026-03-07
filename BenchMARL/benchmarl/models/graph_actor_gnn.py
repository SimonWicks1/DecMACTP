from __future__ import annotations
from dataclasses import dataclass, MISSING
import torch
from torch import nn
from tensordict import TensorDictBase
from benchmarl.models.common import Model, ModelConfig

from .graph_shared_trunk import get_or_create_graph_trunk

class GraphActorGNN(Model):
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

        # 获取环境信息以构建唯一的共享 Key
        self._agent_group = kwargs.get("agent_group", "agents")
        self._model_index = int(kwargs.get("model_index", 0))

        # 1. 从注册表获取共享的图信息编码主干
        self.trunk = get_or_create_graph_trunk(
            agent_group=self._agent_group,
            model_index=self._model_index,
            device=self.device,
            node_in_dim=7, 
            status_emb_dim=status_emb_dim, 
            gnn_hidden_dim=gnn_hidden_dim, 
            num_gnn_layers=num_gnn_layers
        )

        self.phase_proj = nn.Linear(1, gnn_hidden_dim).to(self.device)
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
        B = M * A
        
        # 张量重构为 [B, N, ...]
        node_features = node_features.reshape(B, N, -1)     
        edge_status = edge_status.reshape(B, N, N)          
        edge_weights = edge_weights.reshape(B, N, N)        
        phase = phase.reshape(B, 1)                         
        action_mask = action_mask.reshape(B, N + 1)         

        # 【核心逻辑】：通过统一主干提取特征
        h_nodes, h_graph = self.trunk(node_features, edge_status, edge_weights)

        # 提取自身位置并计算 h_loc
        self_pos = (node_features[:, :, 4] > 0.5).float()  # [B, N]
        h_loc = torch.einsum("bn,bnh->bh", self_pos, h_nodes)  # [B, H]
        phase_h = self.phase_proj(phase.float())  # [B, H]

        # 计算动作 Query 与 Keys
        q_in = torch.cat([h_graph, h_loc, phase_h], dim=-1)  # [B, 3H]
        q = self.query_proj(q_in)  # [B, H]
        k = self.key_proj(h_nodes)  # [B, N, H]

        logits_nodes = torch.einsum("bh,bnh->bn", q, k) / (k.shape[-1] ** 0.5)  # [B, N]
        logits_term = self.terminate_head(q).squeeze(-1)  # [B]
        logits = torch.cat([logits_nodes, logits_term.unsqueeze(-1)], dim=-1)  # [B, N+1]

        logits = logits.masked_fill(~action_mask.bool(), -1e9)
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