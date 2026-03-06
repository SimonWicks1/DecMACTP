from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tensordict import TensorDictBase

from benchmarl.models.common import Model, ModelConfig


class MaGraphFeatureTransformer(nn.Module):
    """
    负责将环境原始观测转换为初始的节点和边特征表示。
    观测结构基于之前的 MACTP 基线推断：
    node_features: [..., A, N, 7]
    edge_weights: [..., A, N, N]
    edge_status: [..., A, N, N]
    """
    def __init__(self, node_in_dim: int = 7, status_emb_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.status_emb = nn.Embedding(4, status_emb_dim)
        self.edge_proj = nn.Linear(1 + status_emb_dim, hidden_dim)

    def forward(self, node_features: torch.Tensor, edge_weights: torch.Tensor, edge_status: torch.Tensor):
        # 提取全局节点特征 (基于基线逻辑)
        dest = node_features[..., 0, :, 0]
        visited = node_features[..., 0, :, 1]
        opt = node_features[..., 0, :, 2]
        pess = node_features[..., 0, :, 3]
        self_pos = (node_features[..., :, 4] > 0.5).float()
        density = node_features[..., :, 5]
        terminated = node_features[..., :, 6]

        occupancy = self_pos.sum(dim=-2)
        density_mean = density.mean(dim=-2)
        term_count = terminated.sum(dim=-2)
        
        # global_node_features: [..., N, 7]
        global_node_features = torch.stack(
            [dest, visited, opt, pess, occupancy, density_mean, term_count], dim=-1
        )
        
        # 提取全局边特征
        edge_status_g = edge_status[..., 0, :, :]
        edge_weights_g = edge_weights[..., 0, :, :]
        
        status_embeds = self.status_emb(edge_status_g.long().clamp(0, 3))
        raw_edge_features = torch.cat([edge_weights_g.unsqueeze(-1), status_embeds], dim=-1)
        
        # 映射到 hidden_dim
        node_embeds = self.node_proj(global_node_features)  # [..., N, H]
        edge_embeds = self.edge_proj(raw_edge_features)     # [..., N, N, H]
        edge_mask = edge_weights_g > 0                      # [..., N, N]
        
        return node_embeds, edge_embeds, edge_mask


class MaGraphFeatureEncoderProcessor(nn.Module):
    """
    NAR 核心处理器：结合当前节点特征与历史算法隐状态，执行一步算法推理。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # 模拟消息传递
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # GRU 单元，用于融合历史隐状态
        self.gru_cell = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_mask: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, H] 节点环境特征
        edge_attr: [B, N, N, H] 边特征
        h_prev: [B, N, H] 上一步的算法隐状态
        """
        B, N, H = x.shape
        
        # 将 x 与隐状态结合作为当前节点的表征
        h_in = x + h_prev 
        
        hi = h_in.unsqueeze(-2).expand(B, N, N, H)
        hj = h_in.unsqueeze(-3).expand(B, N, N, H)
        m_in = torch.cat([hi, hj, edge_attr], dim=-1)
        
        msg = self.msg_mlp(m_in)
        msg = msg.masked_fill(~edge_mask.unsqueeze(-1), -1e9)
        aggr = msg.max(dim=-2).values
        aggr = torch.clamp(aggr, min=-1e5) # [B, N, H]
        
        # 使用 GRU 更新隐状态 (需要将 B 和 N 展平送入 GRUCell)
        aggr_flat = aggr.view(B * N, H)
        h_prev_flat = h_prev.view(B * N, H)
        
        h_next_flat = self.gru_cell(aggr_flat, h_prev_flat)
        h_next = h_next_flat.view(B, N, H)
        
        return h_next


class MaNodeSimilarityMatchAgg(nn.Module):
    """
    通过全局图表示生成 Proto-action，并与各节点隐状态计算相似度，生成对各节点的偏好 logits。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proto_action_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # 包含 h_graph 和 phase_h
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.terminate_head = nn.Linear(hidden_dim, 1)

    def forward(self, h_nodes: torch.Tensor, phase_h: torch.Tensor, self_pos: torch.Tensor) -> torch.Tensor:
        """
        h_nodes: [B, N, H] - processor 输出的新隐状态
        phase_h: [B, A, H] - 阶段特征编码
        self_pos: [B, A, N] - 智能体位置 (one-hot)
        返回: [B, A, N+1] 的 logits
        """
        B, A, N = self_pos.shape
        # 全局图特征使用 mean pooling 提取 (这里可依据原文替换为 DeepSets)
        h_graph = h_nodes.mean(dim=1) # [B, H]
        
        # 智能体当前所在节点特征
        h_loc = torch.einsum("ban,bnh->bah", self_pos, h_nodes) # [B, A, H]
        
        # 生成 proto_action (Query)
        q_in = torch.cat([h_graph.unsqueeze(1).expand(B, A, -1) + h_loc, phase_h], dim=-1)
        proto_action = self.proto_action_mlp(q_in) # [B, A, H]
        
        # 计算节点 keys
        keys = self.key_proj(h_nodes) # [B, N, H]
        
        # 点积相似度 -> logits
        logits_nodes = torch.einsum("bah,bnh->ban", proto_action, keys) / (keys.shape[-1] ** 0.5)
        
        # 终止动作的 logit
        logits_term = self.terminate_head(proto_action).squeeze(-1) # [B, A]
        
        logits = torch.cat([logits_nodes, logits_term.unsqueeze(-1)], dim=-1)
        return logits


class GraphActorGNARL(Model):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        status_emb_dim: int = 8,
        gnn_hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gnn_hidden_dim = gnn_hidden_dim
        
        self.transformer = MaGraphFeatureTransformer(node_in_dim=7, status_emb_dim=status_emb_dim, hidden_dim=gnn_hidden_dim).to(self.device)
        self.processor = MaGraphFeatureEncoderProcessor(hidden_dim=gnn_hidden_dim).to(self.device)
        self.aggregator = MaNodeSimilarityMatchAgg(hidden_dim=gnn_hidden_dim).to(self.device)
        self.phase_proj = nn.Linear(1, gnn_hidden_dim).to(self.device)
        
        # 注册隐状态 key
        self.hidden_state_key = ("agents", "hidden_state")
        self.next_hidden_state_key = ("next", "agents", "hidden_state")
        
        # 必须通知 BenchMARL 预期接收和输出这些键
        self.in_keys.extend([self.hidden_state_key])
        self.out_keys.extend([self.next_hidden_state_key])

    def _step(self, x_embed: torch.Tensor, edge_embed: torch.Tensor, edge_mask: torch.Tensor, h_prev: torch.Tensor, phase_h: torch.Tensor, self_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步推进逻辑"""
        h_next = self.processor(x_embed, edge_embed, edge_mask, h_prev)
        logits = self.aggregator(h_next, phase_h, self_pos)
        return logits, h_next

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        node_features = tensordict.get(("agents", "node_features")).to(self.device)
        edge_weights = tensordict.get(("agents", "edge_weights")).to(self.device)
        edge_status = tensordict.get(("agents", "edge_status")).to(self.device)
        phase = tensordict.get(("agents", "phase")).to(self.device)
        action_mask = tensordict.get(("agents", "action_mask")).to(self.device)
        
        batch_shape = tensordict.batch_size
        
        # 核心判断：数据是否带有时间维度 (T)
        # 在 Rollout 时，batch_shape 通常为 [E] (环境数)
        # 在 PPO 训练时，batch_shape 通常为 [E, T] (环境数, 时间步)
        has_time_dim = len(batch_shape) > 1 

        if not has_time_dim:
            # === 单步推理模式 (环境收集阶段) ===
            B = batch_shape[0]
            N = node_features.shape[-2]
            
            # 尝试获取上一隐状态；如果是回合第一步，可能为全 0
            h_prev = tensordict.get(self.hidden_state_key, None)
            if h_prev is None:
                h_prev = torch.zeros(B, N, self.gnn_hidden_dim, device=self.device)
            else:
                h_prev = h_prev.to(self.device)

            x_embed, edge_embed, edge_mask = self.transformer(node_features, edge_weights, edge_status)
            phase_h = self.phase_proj(phase.float())
            self_pos = (node_features[..., 4] > 0.5).float()
            
            logits, h_next = self._step(x_embed, edge_embed, edge_mask, h_prev, phase_h, self_pos)
            
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
            
            tensordict.set(self.out_key, logits)
            tensordict.set(self.next_hidden_state_key, h_next)
            
        else:
            # === 序列训练模式 (BPTT 阶段) ===
            B, T = batch_shape[0], batch_shape[1]
            N = node_features.shape[-2]
            
            # 取轨迹开始的初始隐状态 [B, N, H]
            h_prev = tensordict.get(self.hidden_state_key)[:, 0, ...].to(self.device)
            
            # 预计算所有时间步的静态特征，减少循环内计算开销
            x_embed, edge_embed, edge_mask = self.transformer(node_features, edge_weights, edge_status)
            phase_h = self.phase_proj(phase.float())
            self_pos = (node_features[..., 4] > 0.5).float()
            
            logits_seq = []
            h_next_seq = []
            
            # 沿时间步自回归展开
            for t in range(T):
                logits_t, h_prev = self._step(
                    x_embed[:, t], edge_embed[:, t], edge_mask[:, t], 
                    h_prev, phase_h[:, t], self_pos[:, t]
                )
                logits_seq.append(logits_t)
                h_next_seq.append(h_prev)
            
            logits = torch.stack(logits_seq, dim=1)      # [B, T, A, N+1]
            h_next_out = torch.stack(h_next_seq, dim=1)  # [B, T, N, H]
            
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
            
            tensordict.set(self.out_key, logits)
            tensordict.set(self.next_hidden_state_key, h_next_out)

        return tensordict

@dataclass
class GraphActorGNARLConfig(ModelConfig):
    node_features: int = MISSING
    edge_features: int = MISSING
    status_emb_dim: int = 8
    gnn_hidden_dim: int = 64

    @staticmethod
    def associated_class():
        return GraphActorGNARL