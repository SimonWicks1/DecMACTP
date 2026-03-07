from __future__ import annotations
from dataclasses import dataclass, MISSING
from typing import Tuple

import torch
from torch import nn
from tensordict import TensorDictBase

from benchmarl.models.common import Model, ModelConfig
from .graph_shared_trunk import BaseGraphTrunk  # 导入基础主干

class GraphCriticGNN(Model):
    """
    MAPPO 中心化 Critic 网络。
    具有独立实例化的 BaseGraphTrunk 以处理全局语义的节点特征。
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
        
        # 1. 独立实例化图信息编码主干 (不与 Actor 共享参数)
        self.trunk = BaseGraphTrunk(
            node_in_dim=7, 
            status_emb_dim=status_emb_dim, 
            gnn_hidden_dim=gnn_hidden_dim, 
            num_gnn_layers=num_gnn_layers
        ).to(self.device)

        # 2. 价值评估相关的特有组件
        self.phase_proj = nn.Linear(1, gnn_hidden_dim).to(self.device) if use_phase else None

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

        batch_shape = tensordict.batch_size
        M = 1
        for s in batch_shape:
            M *= int(s)

        # 内部 flatten 计算：[..., A, N, 7] -> [M, A, N, 7]
        node_features = node_features.reshape(M, *node_features.shape[-3:])  
        edge_status = edge_status.reshape(M, *edge_status.shape[-3:])        
        edge_weights = edge_weights.reshape(M, *edge_weights.shape[-3:])     
        if phase is not None:
            phase = phase.reshape(M, *phase.shape[-2:])                      

        M, A, N, _ = node_features.shape

        # 获取环境级图结构
        edge_status_g = edge_status[:, 0]    # [M, N, N]
        edge_weights_g = edge_weights[:, 0]  # [M, N, N]

        # 构建具有全局语义的节点特征
        x_global = self._build_global_node_features(node_features)  # [M, N, 7]

        # 【核心逻辑】：通过独立主干提取特征
        _, h_graph = self.trunk(x_global, edge_status_g, edge_weights_g) # [M, H]

        if self.use_phase:
            phase_g = phase[:, 0].float()         
            phase_h = self.phase_proj(phase_g)    
            h = torch.cat([h_graph, phase_h], dim=-1)
        else:
            h = h_graph

        v = self.value_head(h)  # [M, 1]
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