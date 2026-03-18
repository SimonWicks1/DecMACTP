from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
#  GNARL-style local encode + local graph process
# ============================================================
class MaGraphFeatureTransformer(nn.Module):
    """
    Encode explicit node / edge / graph features at each step.

    Inputs:
      node_features: [B,N,7]
      edge_status:   [B,N,N]
      edge_weights:  [B,N,N]
      phase:         [B,1] or [B,Dp]

    Outputs:
      node0:    [B,N,D]
      edge0:    [B,N,N,D]
      graph0:   [B,D]
      edge_mask:[B,N,N]
    """

    def __init__(self, embed_dim: int = 64, node_in_dim: int = 7, status_emb_dim: int = 8):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.node_proj = nn.Linear(node_in_dim, embed_dim)
        self.status_emb = nn.Embedding(4, status_emb_dim)
        self.edge_proj = nn.Linear(status_emb_dim + 1, embed_dim)
        self.phase_proj = nn.Linear(1, embed_dim)
        self.node_norm = nn.LayerNorm(embed_dim)
        self.edge_norm = nn.LayerNorm(embed_dim)
        self.graph_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_status: torch.Tensor,
        edge_weights: torch.Tensor,
        phase: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = node_features.shape
        phase = phase.float().view(B, -1)
        if phase.shape[-1] != 1:
            phase = phase[..., :1]

        node0 = self.node_norm(self.node_proj(node_features.float()))

        status = edge_status.long().clamp(min=0, max=3)
        status_e = self.status_emb(status)
        ew = edge_weights.float().unsqueeze(-1)
        edge_in = torch.cat([status_e, ew], dim=-1)
        edge0 = self.edge_norm(self.edge_proj(edge_in))

        graph0 = self.graph_norm(self.phase_proj(phase))

        edge_mask = edge_weights > 0
        return node0, edge0, graph0, edge_mask


class EdgeAwareNodeBlock(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, edge0: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        xi = x.unsqueeze(2).expand(B, N, N, D)
        xj = x.unsqueeze(1).expand(B, N, N, D)
        msg_in = torch.cat([xi, xj, edge0], dim=-1)
        msg = self.msg_mlp(msg_in)
        msg = msg * edge_mask.unsqueeze(-1).float()
        agg = msg.sum(dim=2) / (edge_mask.sum(dim=2, keepdim=True).float() + 1e-6)
        out = self.upd_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(x + out)


class MaGraphFeatureEncoderProcessor(nn.Module):
    def __init__(self, embed_dim: int = 64, num_gnn_layers: int = 2, pooling_type: str = "mean"):
        super().__init__()
        self.pooling_type = pooling_type
        self.gnns = nn.ModuleList([EdgeAwareNodeBlock(embed_dim) for _ in range(num_gnn_layers)])

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "sum":
            return x.sum(dim=1)
        if self.pooling_type == "max":
            return x.max(dim=1).values
        return x.mean(dim=1)

    def forward(self, node0, edge0, graph0, edge_mask):
        h = node0
        for gnn in self.gnns:
            h = gnn(h, edge0, edge_mask)
        hG = self.pool(h) + graph0
        return h, hG


# ============================================================
#  Coordination summaries from local graph state
# ============================================================
class AgentGraphSummaryExtractor(nn.Module):
    """
    Extract explicit coordination summaries from local graph embeddings.

    Returns:
      coord_token:      [B,A,D]
      region_summary:   [B,A,D]
      goal_summary:     [B,A,D]
      frontier_summary: [B,A,D]
      risk_summary:     [B,A,D]
      node_gate:        [B,A,N,1]
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.goal_gate = nn.Linear(embed_dim + 2, 1)
        self.frontier_gate = nn.Linear(embed_dim + 1, 1)
        self.risk_gate = nn.Linear(embed_dim + 3, 1)
        self.region_gate = nn.Linear(embed_dim + 4, 1)
        self.coord_proj = nn.Sequential(
            nn.Linear(embed_dim * 5 + 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _weighted_summary(self, hV: torch.Tensor, gate_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(gate_logits.squeeze(-1), dim=-1)
        summary = torch.einsum("ban,band->bad", weights, hV)
        return summary, weights.unsqueeze(-1)

    def forward(
        self,
        hV: torch.Tensor,
        hG: torch.Tensor,
        node_features: torch.Tensor,
        edge_status: torch.Tensor,
        edge_weights: torch.Tensor,
        phase: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # channels in node_features
        # 0 dest_mask / goal mask
        # 1 visit mask
        # 2 opt_cost
        # 3 pess_cost
        # 4 self_pos
        # 5 density
        # 6 terminated feat
        dest = node_features[..., 0:1].float()
        visited = node_features[..., 1:2].float()
        opt_cost = node_features[..., 2:3].float()
        pess_cost = node_features[..., 3:4].float()
        self_pos = node_features[..., 4:5].float()
        density = node_features[..., 5:6].float()
        terminated = node_features[..., 6:7].float()

        uncertainty = (edge_status == 0).float() + (edge_status == 1).float()
        frontier_score = uncertainty.max(dim=-1, keepdim=True).values
        risk_score = (pess_cost - opt_cost).abs()

        goal_in = torch.cat([hV, dest, 1.0 - visited], dim=-1)
        frontier_in = torch.cat([hV, frontier_score], dim=-1)
        risk_in = torch.cat([hV, risk_score, opt_cost, pess_cost], dim=-1)
        region_in = torch.cat([hV, dest, frontier_score, density, self_pos], dim=-1)

        goal_summary, goal_w = self._weighted_summary(hV, self.goal_gate(goal_in))
        frontier_summary, frontier_w = self._weighted_summary(hV, self.frontier_gate(frontier_in))
        risk_summary, risk_w = self._weighted_summary(hV, self.risk_gate(risk_in))
        region_summary, region_w = self._weighted_summary(hV, self.region_gate(region_in))
        loc_summary = torch.einsum("ban,band->bad", self_pos.squeeze(-1), hV)

        phase = phase.float().view(*phase.shape[:2], -1)
        if phase.shape[-1] != 1:
            phase = phase[..., :1]
        term_scalar = terminated.amax(dim=-2)

        coord_token = self.coord_proj(torch.cat([
            hG,
            loc_summary,
            goal_summary,
            frontier_summary,
            risk_summary,
            phase,
            term_scalar,
        ], dim=-1))

        return {
            "coord_token": coord_token,
            "region_summary": region_summary,
            "goal_summary": goal_summary,
            "frontier_summary": frontier_summary,
            "risk_summary": risk_summary,
            "goal_weights": goal_w,
            "frontier_weights": frontier_w,
            "risk_weights": risk_w,
            "region_weights": region_w,
        }


# ============================================================
#  Problem-induced dynamic communication graph
# ============================================================
class ProblemInducedCommGraphBuilder(nn.Module):
    def __init__(self, max_distance_norm: float = 32.0, top_k: int = 4):
        super().__init__()
        self.max_distance_norm = float(max_distance_norm)
        self.top_k = int(top_k)

    @staticmethod
    def _current_pos(node_features: torch.Tensor) -> torch.Tensor:
        return (node_features[..., 4] > 0.5).float()

    @staticmethod
    def _known_open_adj(edge_status: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
        return ((edge_weights > 0) & (edge_status != 3)).float()

    def _pairwise_graph_distance(
        self,
        self_pos: torch.Tensor,
        edge_status: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        B, A, N = self_pos.shape
        open_adj = self._known_open_adj(edge_status[:, 0], edge_weights[:, 0])
        base_w = edge_weights[:, 0].float()
        inf = torch.full_like(base_w, 1e6)
        base_w = torch.where(open_adj.bool(), base_w, inf)
        eye = torch.eye(N, device=base_w.device, dtype=torch.bool).unsqueeze(0).expand_as(base_w)
        dist = torch.where(eye, torch.zeros_like(base_w), base_w)
        for k in range(N):
            dist = torch.minimum(dist, dist[:, :, k].unsqueeze(-1) + dist[:, k, :].unsqueeze(1))
        # dist: [B,N,N]. We want agent_pair_dist[b,a,c] = dist[b, loc[b,a], loc[b,c]]
        # Step 1: gather row dist[b, loc[b,a], :] for each agent a -> [B,A,N]
        loc_idx = self_pos.argmax(dim=-1)                          # [B,A]
        row_idx = loc_idx.unsqueeze(-1).expand(B, A, N)            # [B,A,N]
        dist_rows = dist.gather(1, row_idx)                        # [B,A,N]
        # Step 2: gather column loc[b,c] for each agent c -> [B,A,A]
        col_idx = loc_idx.unsqueeze(1).expand(B, A, A)             # [B,A,A]
        agent_pair_dist = dist_rows.gather(2, col_idx)             # [B,A,A]
        agent_pair_dist = torch.where(
            agent_pair_dist >= 1e5,
            torch.full_like(agent_pair_dist, self.max_distance_norm),
            agent_pair_dist,
        )
        return agent_pair_dist

    @staticmethod
    def _cosine_overlap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return torch.einsum("bid,bjd->bij", x, y).clamp(min=-1.0, max=1.0)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_status: torch.Tensor,
        edge_weights: torch.Tensor,
        summaries: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        B, A, N, _ = node_features.shape
        self_pos = self._current_pos(node_features)
        pair_dist = self._pairwise_graph_distance(self_pos, edge_status, edge_weights)
        prox = torch.exp(-pair_dist / max(1.0, self.max_distance_norm))

        goal = 0.5 * (self._cosine_overlap(summaries["goal_summary"], summaries["goal_summary"]) + 1.0)
        frontier = 0.5 * (self._cosine_overlap(summaries["frontier_summary"], summaries["frontier_summary"]) + 1.0)
        region = 0.5 * (self._cosine_overlap(summaries["region_summary"], summaries["region_summary"]) + 1.0)
        risk = 0.5 * (self._cosine_overlap(summaries["risk_summary"], summaries["risk_summary"]) + 1.0)

        rel_feat = torch.stack([prox, goal, frontier, region, risk], dim=-1)  # [B,A,A,5]
        base_score = (prox + goal + frontier + region + risk) / 5.0

        eye = torch.eye(A, device=base_score.device, dtype=torch.bool).unsqueeze(0)
        base_score = base_score.masked_fill(eye, -1e9)

        k = min(max(1, self.top_k), max(1, A - 1))
        topk_vals, topk_idx = torch.topk(base_score, k=k, dim=-1)
        src_bool = torch.ones(B, A, k, dtype=torch.bool, device=base_score.device)
        adj = torch.zeros_like(base_score, dtype=torch.bool).scatter(-1, topk_idx, src_bool)
        softmax_vals = F.softmax(topk_vals, dim=-1)
        weights = torch.zeros_like(base_score).scatter(-1, topk_idx, softmax_vals)

        return {
            "rel_feat": rel_feat,
            "base_score": base_score,
            "rule_adj": adj,
            "rule_weights": weights,
            "coupling_prox": prox,
            "coupling_goal": goal,
            "coupling_frontier": frontier,
            "coupling_region": region,
            "coupling_risk": risk,
        }


class SemanticEdgeGating(nn.Module):
    """
    Learned semantic gating on top of the rule-induced candidate graph.
    Only candidate edges from rule_adj are scored.
    """

    def __init__(self, embed_dim: int, rel_dim: int = 5, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = int(hidden_dim or embed_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 6 + rel_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        summaries: Dict[str, torch.Tensor],
        rel_feat: torch.Tensor,
        rule_adj: torch.Tensor,
        base_score: torch.Tensor,
        graph_context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        coord = summaries["coord_token"]
        goal = summaries["goal_summary"]
        frontier = summaries["frontier_summary"]
        region = summaries["region_summary"]
        risk = summaries["risk_summary"]
        B, A, D = coord.shape

        def pair_expand(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            xi = x.unsqueeze(2).expand(B, A, A, D)
            xj = x.unsqueeze(1).expand(B, A, A, D)
            return xi, xj

        ci, cj = pair_expand(coord)
        gi, gj = pair_expand(goal)
        fi, fj = pair_expand(frontier)
        gc = graph_context.unsqueeze(1).unsqueeze(1).expand(B, A, A, D)

        edge_in = torch.cat([ci, cj, gi, gj, fi, fj, rel_feat, gc], dim=-1)
        learned = self.edge_mlp(edge_in).squeeze(-1)
        score = base_score + learned
        score = score.masked_fill(~rule_adj, -1e9)
        weights = F.softmax(score, dim=-1)
        weights = weights * rule_adj.float()
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return {
            "adj": rule_adj,
            "weights": weights,
            "score": score,
            "learned_edge_score": learned,
        }


# ============================================================
#  Agent-to-graph-to-agent coordination
# ============================================================
class RelationCommLayer(nn.Module):
    def __init__(self, embed_dim: int, rel_dim: int = 5, num_heads: int = 4):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.rel_proj = nn.Linear(rel_dim, num_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, rel_feat, adj, edge_weights):
        B, A, D = x.shape
        q = self.q_proj(x).view(B, A, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, A, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, A, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.einsum("bhid,bhjd->bhij", q, k) / (self.head_dim ** 0.5)
        rel_bias = self.rel_proj(rel_feat).permute(0, 3, 1, 2)
        attn = attn + rel_bias
        attn = attn.masked_fill(~adj.unsqueeze(1), -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = attn * edge_weights.unsqueeze(1)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).contiguous().view(B, A, D)
        h1 = self.norm1(x + self.out_proj(out))
        h2 = self.norm2(h1 + self.ff(h1))
        return h2


class CoordinationToGraphFeedback(nn.Module):
    """
    Feed coordination embeddings back into node-level reasoning.
    This is the key 'agent -> graph' return path.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.node_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.node_bias = nn.Linear(embed_dim * 2, embed_dim)
        self.graph_fuse = nn.Linear(embed_dim * 2, embed_dim)
        self.node_norm = nn.LayerNorm(embed_dim)
        self.graph_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hV: torch.Tensor,
        hG: torch.Tensor,
        summaries: Dict[str, torch.Tensor],
        u_comm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hV:[B,A,N,D], hG:[B,A,D], u_comm:[B,A,D]
        B, A, N, D = hV.shape
        region_w = summaries["region_weights"]  # [B,A,N,1]
        graph_token = summaries["region_summary"] + u_comm  # [B,A,D]

        gate = torch.sigmoid(self.node_gate(torch.cat([
            hV,
            graph_token.unsqueeze(-2).expand(B, A, N, D),
        ], dim=-1)))
        bias = self.node_bias(torch.cat([
            hV,
            graph_token.unsqueeze(-2).expand(B, A, N, D),
        ], dim=-1))
        hV_out = self.node_norm(hV + gate * bias)

        region_ctx = torch.einsum("ban,band->bad", region_w.squeeze(-1), hV_out)
        hG_out = self.graph_norm(hG + self.graph_fuse(torch.cat([region_ctx, u_comm], dim=-1)))
        return hV_out, hG_out, graph_token


# ============================================================
#  Shared MAGNARL trunk: Encode -> Process(local) -> Coordinate(agent-graph-agent)
# ============================================================
_SHARED_TRUNKS: Dict[Tuple[str, int, str], "SharedEncodeCoordinateProcessTrunk"] = {}


def _make_trunk_key(agent_group: str, model_index: int, device: torch.device) -> Tuple[str, int, str]:
    return (str(agent_group), int(model_index), str(device))


class SharedEncodeCoordinateProcessTrunk(nn.Module):
    """
    Idealized MAGNARL trunk (v2):
      1) GNARL local encode
      2) GNARL local graph process
      3) coordination-summary extraction from local graph state
      4) problem-induced dynamic communication graph
      5) agent-to-agent coordination on the communication graph
      6) coordination feedback to node-level graph reasoning

    Forward returns:
      hV_out : [B,A,N,D]   coordination-aware node embeddings
      hG_out : [B,A,D]     coordination-aware graph embeddings
      u_comm : [B,A,D]     coordinated agent embeddings
      aux    : dict        diagnostics and intermediate summaries
    """

    def __init__(
        self,
        embed_dim: int,
        num_gnn_layers: int = 2,
        status_emb_dim: int = 8,
        pooling_type: str = "mean",
        comm_num_heads: int = 4,
        comm_top_k: int = 4,
    ):
        super().__init__()
        self.transformer = MaGraphFeatureTransformer(
            embed_dim=embed_dim,
            node_in_dim=7,
            status_emb_dim=status_emb_dim,
        )
        self.processor = MaGraphFeatureEncoderProcessor(
            embed_dim=embed_dim,
            num_gnn_layers=num_gnn_layers,
            pooling_type=pooling_type,
        )
        self.summary_extractor = AgentGraphSummaryExtractor(embed_dim=embed_dim)
        self.comm_builder = ProblemInducedCommGraphBuilder(top_k=comm_top_k)
        self.edge_gating = SemanticEdgeGating(embed_dim=embed_dim, rel_dim=5)
        self.comm_layer = RelationCommLayer(embed_dim=embed_dim, rel_dim=5, num_heads=comm_num_heads)
        self.feedback = CoordinationToGraphFeedback(embed_dim=embed_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_status: torch.Tensor,
        edge_weights: torch.Tensor,
        phase: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, A, N, _ = node_features.shape
        phase = phase.float().view(B, A, -1)
        flat_B = B * A

        node0, edge0, graph0, edge_mask = self.transformer(
            node_features.reshape(flat_B, N, -1),
            edge_status.reshape(flat_B, N, N),
            edge_weights.reshape(flat_B, N, N),
            phase.reshape(flat_B, -1),
        )
        hV_local, hG_local = self.processor(node0, edge0, graph0, edge_mask)
        hV_local = hV_local.view(B, A, N, -1)
        hG_local = hG_local.view(B, A, -1)

        summaries = self.summary_extractor(hV_local, hG_local, node_features, edge_status, edge_weights, phase)
        graph_context = hG_local.mean(dim=1)

        comm_graph = self.comm_builder(node_features, edge_status, edge_weights, summaries)
        gated = self.edge_gating(
            summaries=summaries,
            rel_feat=comm_graph["rel_feat"],
            rule_adj=comm_graph["rule_adj"],
            base_score=comm_graph["base_score"],
            graph_context=graph_context,
        )
        u_comm = self.comm_layer(
            summaries["coord_token"],
            comm_graph["rel_feat"],
            gated["adj"],
            gated["weights"],
        )

        hV_out, hG_out, graph_token = self.feedback(hV_local, hG_local, summaries, u_comm)

        aux = {
            **summaries,
            **comm_graph,
            **gated,
            "agent_coord_embed": u_comm,
            "graph_feedback_token": graph_token,
            "hV_local": hV_local,
            "hG_local": hG_local,
        }
        diag = {
            "comm_degree_mean":       gated["adj"].sum(dim=-1).float().mean(),
            "comm_density":           gated["adj"].float().mean(),
            "rule_edge_score_mean":   comm_graph["base_score"].mean(),
            "learned_edge_score_mean": gated["learned_edge_score"].mean(),   # was "learned_score"
            "edge_gate_mean":         gated["weights"].mean(),
            # "edge_gate_entropy" removed — not returned by SemanticEdgeGating

            "coupling_prox_mean":     comm_graph["coupling_prox"].mean(),     # was "prox_score"
            "coupling_goal_mean":     comm_graph["coupling_goal"].mean(),     # was "goal_score"
            "coupling_frontier_mean": comm_graph["coupling_frontier"].mean(), # was "frontier_score"
            "coupling_region_mean":   comm_graph["coupling_region"].mean(),   # was "region_score"
            "coupling_risk_mean":     comm_graph["coupling_risk"].mean(),     # was "risk_score"

            "goal_summary_norm":      summaries["goal_summary"].norm(dim=-1).mean(),
            "frontier_summary_norm":  summaries["frontier_summary"].norm(dim=-1).mean(),
            "risk_summary_norm":      summaries["risk_summary"].norm(dim=-1).mean(),
            "region_summary_norm":    summaries["region_summary"].norm(dim=-1).mean(),

            "node_feedback_delta":    (hV_out - hV_local).norm(dim=-1).mean(),
            "graph_feedback_delta":   (hG_out - hG_local).norm(dim=-1).mean(),
        }
        # return hV_out, hG_out, u_comm, aux
        return hV_out, hG_out, u_comm, aux, diag


def get_or_create_shared_trunk(
    agent_group: str,
    model_index: int,
    device: torch.device,
    embed_dim: int,
    num_gnn_layers: int,
    status_emb_dim: int,
    pooling_type: str,
    comm_num_heads: int,
    comm_top_k: int,
    **kwargs,
) -> SharedEncodeCoordinateProcessTrunk:
    key = _make_trunk_key(agent_group, model_index, device)
    if key not in _SHARED_TRUNKS:
        _SHARED_TRUNKS[key] = SharedEncodeCoordinateProcessTrunk(
            embed_dim=embed_dim,
            num_gnn_layers=num_gnn_layers,
            status_emb_dim=status_emb_dim,
            pooling_type=pooling_type,
            comm_num_heads=comm_num_heads,
            comm_top_k=comm_top_k,
        ).to(device)
    return _SHARED_TRUNKS[key]