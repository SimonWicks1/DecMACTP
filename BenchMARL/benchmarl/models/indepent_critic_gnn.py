from __future__ import annotations
from dataclasses import dataclass, MISSING
import torch
from torch import nn
from tensordict import TensorDictBase
from benchmarl.models.common import Model, ModelConfig

from .graph_shared_trunk import get_or_create_graph_trunk

class IndependentGraphCriticGNN(Model):
    """
    专为 IPPO 设计的去中心化图 Critic 网络。
    采用 Stop-Gradient (detach_trunk) 机制防止计算图冲突与 Actor-Critic 梯度干扰。
    """
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        num_gnn_layers: int = 2,
        status_emb_dim: int = 8,
        gnn_hidden_dim: int = 64,
        use_phase: bool = True,
        detach_trunk: bool = True,  # 默认开启截断机制
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_phase = use_phase
        self.detach_trunk = detach_trunk

        self._agent_group = kwargs.get("agent_group", "agents")
        self._model_index = int(kwargs.get("model_index", 0))

        # 从同一注册表获取共享主干
        self.trunk = get_or_create_graph_trunk(
            agent_group=self._agent_group,
            model_index=self._model_index,
            device=self.device,
            node_in_dim=7,
            status_emb_dim=status_emb_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gnn_layers=num_gnn_layers
        )

        self.phase_proj = nn.Linear(1, gnn_hidden_dim).to(self.device) if use_phase else None

        head_in = (gnn_hidden_dim * 2) + (gnn_hidden_dim if use_phase else 0)
        self.value_head = nn.Sequential(
            nn.Linear(head_in, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, 1),
        ).to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
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
        B = M * A
        
        node_features = node_features.reshape(B, N, -1)     
        edge_status = edge_status.reshape(B, N, N)          
        edge_weights = edge_weights.reshape(B, N, N)        
        if phase is not None:
            phase = phase.reshape(B, 1)                     

        # 通过统一主干提取节点和图的特征
        h_nodes, h_graph = self.trunk(node_features, edge_status, edge_weights)

        # 【关键修复】：截断梯度，防止 PyTorch 计算图冲突
        if self.detach_trunk:
            h_nodes = h_nodes.detach()
            h_graph = h_graph.detach()

        self_pos = (node_features[:, :, 4] > 0.5).float()
        h_loc = torch.einsum("bn,bnh->bh", self_pos, h_nodes)

        if self.use_phase:
            phase_h = self.phase_proj(phase.float())
            h_combine = torch.cat([h_graph, h_loc, phase_h], dim=-1)
        else:
            h_combine = torch.cat([h_graph, h_loc], dim=-1)

        v = self.value_head(h_combine)
        v = v.reshape(*batch_shape, A, 1)
        tensordict.set(self.out_key, v)
        
        return tensordict

@dataclass
class IndependentGraphCriticGNNConfig(ModelConfig):
    node_features: int = MISSING
    edge_features: int = MISSING
    num_gnn_layers: int = 2
    status_emb_dim: int = 8
    gnn_hidden_dim: int = 64
    use_phase: bool = True
    detach_trunk: bool = True  # 添加配置项

    @staticmethod
    def associated_class():
        return IndependentGraphCriticGNN