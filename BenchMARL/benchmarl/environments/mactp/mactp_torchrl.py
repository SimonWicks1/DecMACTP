import copy
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import torch
from tensordict import TensorDict
from torch_geometric.utils import to_dense_adj
from torchrl.data import Composite, Categorical, Unbounded, Bounded
from torchrl.envs import EnvBase

STATUS_UNCERTAIN = 0
STATUS_NORMAL = 1
STATUS_OBSERVED_OPEN = 2
STATUS_BLOCKED = 3

def _import_from_path(class_path: str):
    module_path, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)

def _maybe_complete_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from gnarl.util.bc import complete_config
        return complete_config(cfg)
    except Exception:
        return cfg

class MultiTravelerCTPTorchRLEnv(EnvBase):
    def __init__(
        self,
        max_nodes: Optional[int],
        num_agents: Optional[int],
        graph_generator: Union[object, Dict[str, Any]],
        device: str = "cpu",
        num_envs: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be >= 1")

        super().__init__(device=device, batch_size=torch.Size([self.num_envs]))
        self._graph_generator_cfg = graph_generator if isinstance(graph_generator, dict) else None
        self._base_seed = seed
        self._generators = self._build_generators(graph_generator, seed=seed, num_envs=self.num_envs)

        inferred_max_nodes, inferred_num_agents = self._infer_sizes_from_generator_cfg(self._graph_generator_cfg)
        self.max_nodes = int(max_nodes if max_nodes is not None else inferred_max_nodes)
        self.num_agents = int(num_agents if num_agents is not None else inferred_num_agents)

        self.num_phases = self.max_nodes * 3
        self.agent_colors = ["red", "blue", "green", "orange", "purple", "brown"]

        self._build_specs()

        E, A, N = self.num_envs, self.num_agents, self.max_nodes
        self.graph_data: List[Optional[object]] = [None] * E
        self.current_locations = torch.zeros((E, A), dtype=torch.int64, device=self.device)
        # [防震荡] 记忆上一时间步的位置
        self.previous_locations = torch.zeros((E, A), dtype=torch.int64, device=self.device)
        self.destination_mask = torch.zeros((E, N), dtype=torch.int32, device=self.device)
        self.goals_visited_mask = torch.zeros((E, N), dtype=torch.int32, device=self.device)
        self.goals_occupied_mask = torch.zeros((E, N), dtype=torch.bool, device=self.device)

        self.static_weights = torch.zeros((E, N, N), dtype=torch.float32, device=self.device)
        self.adj_matrix = torch.zeros((E, N, N), dtype=torch.float32, device=self.device)
        self.edge_status = torch.zeros((E, N, N), dtype=torch.int32, device=self.device)

        self.agent_terminated = torch.zeros((E, A), dtype=torch.bool, device=self.device)
        self.current_phase = torch.zeros((E,), dtype=torch.int64, device=self.device)
        
        # [评估用] Oracle 全局最短路基准
        self.gt_distances = torch.full((E, N, N), float("inf"), dtype=torch.float32, device=self.device)

    def _build_specs(self) -> None:
        A, N = self.num_agents, self.max_nodes

        # ---------- action ----------
        action_unbatched = Composite({
            "agents": Composite(
                {"action": Categorical(n=N + 1, shape=(A,), dtype=torch.int64, device=self.device)},
                shape=(A,),
            )
        })

        # ---------- observation ----------
        # 全局状态展平后的总维度: 2N(nodes) + N^2(status) + N(opt) + N(pess) + N^2(weights) + 2N(masks) = 2N^2 + 6N
        state_dim = 2 * N * N + 6 * N

        obs_unbatched = Composite({
            "agents": Composite(
                {
                    "node_features": Unbounded(shape=(A, N, 7), dtype=torch.float32, device=self.device),
                    "edge_status": Bounded(low=0, high=3, shape=(A, N, N), dtype=torch.int32, device=self.device),
                    "edge_weights": Unbounded(shape=(A, N, N), dtype=torch.float32, device=self.device),
                    "agent_id": Unbounded(shape=(A, 1), dtype=torch.int64, device=self.device),
                    "phase": Unbounded(shape=(A, 1), dtype=torch.int64, device=self.device),
                    # 对于布尔类型的掩码，使用 Bounded(low=0, high=1) 比 Categorical 更符合 TorchRL 规范
                    "action_mask": Bounded(low=0, high=1, shape=(A, N + 1), dtype=torch.bool, device=self.device),
                    "goals_visited_mask": Bounded(low=0, high=1, shape=(A, N), dtype=torch.int32, device=self.device),
                    "destination_mask": Bounded(low=0, high=1, shape=(A, N), dtype=torch.int32, device=self.device),
                },
                shape=(A,),
            ),
            # "state": Unbounded(shape=(state_dim,), dtype=torch.float32, device=self.device),
            "penalty": Unbounded(shape=(1,), dtype=torch.float32, device=self.device)
        })

        # ---------- reward ----------
        reward_unbatched = Composite({
            "agents": Composite(
                {"reward": Unbounded(shape=(A, 1), dtype=torch.float32, device=self.device)},
                shape=(A,),
            )
        })

        # ---------- done ----------
        done_unbatched = Composite({
            "done": Bounded(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
            "terminated": Bounded(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
        })

        # ==========================================
        # [核心修复] 显式扩展至环境维度
        # TorchRL EnvBase 会自动识别这四大核心规格，并反向推导对应的 _unbatched 属性
        # ==========================================
        self.action_spec = action_unbatched.expand(self.num_envs)
        self.observation_spec = obs_unbatched.expand(self.num_envs)
        self.reward_spec = reward_unbatched.expand(self.num_envs)
        self.done_spec = done_unbatched.expand(self.num_envs)

    def _build_generators(self, graph_generator, seed: Optional[int], num_envs: int):
        if isinstance(graph_generator, dict):
            GenCls, datasets, base_kwargs = self._build_graph_datasets_from_cfg(graph_generator, seed=seed)
            gens = []
            for i in range(num_envs):
                env_seed = seed + i if seed is not None else None
                kwargs = base_kwargs.copy()
                gen_instance = GenCls(datasets=datasets, seed=env_seed, **kwargs)
                gens.append(gen_instance.generate())
            return gens
        
        gens = []
        for i in range(num_envs):
            g = graph_generator if i == 0 else copy.deepcopy(graph_generator)
            if seed is not None and hasattr(g, "seed") and callable(getattr(g, "seed")):
                try: g.seed(seed + i)
                except Exception: pass
            gens.append(g.generate())
        return gens
    
    def _build_graph_datasets_from_cfg(self, gen_cfg: Dict[str, Any], seed: Optional[int]):
        class_path = gen_cfg.get("class_path", None)
        GenCls = _import_from_path(class_path)
        data_cfg = _maybe_complete_config(gen_cfg.get("data", {}))
        from gnarl.envs.generate.data import GraphProblemDataset
        
        root = (Path(data_cfg.get("data_root", ".")) / data_cfg["graph_dir"]).resolve()
        ds_seed = data_cfg.get("seed", None) if seed is None else seed

        datasets = []
        for n_key, num_samples in data_cfg["node_samples"].items():
            datasets.append(GraphProblemDataset(
                root=str(root), split=data_cfg.get("split", "train"), algorithm=data_cfg["algorithm"],
                num_nodes=int(n_key), num_samples=int(num_samples), seed=ds_seed,
                graph_generator=data_cfg["graph_generator"], graph_generator_kwargs=data_cfg.get("graph_generator_kwargs", None),
                num_starts=data_cfg.get("num_starts", 1), num_goals=data_cfg.get("num_goals", 1),
            ))
        return GenCls, datasets, dict(gen_cfg.get("kwargs", {}))
    
    def _infer_sizes_from_generator_cfg(self, cfg: Optional[Dict[str, Any]]) -> Tuple[int, int]:
        if not cfg: return 30, 1
        data_cfg = cfg.get("data", {})
        return max(int(k) for k in data_cfg.get("node_samples", {"30": 1}).keys()), int(data_cfg.get("num_starts", 1))

    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        if tensordict is None:
            reset_mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        else:
            reset_mask = tensordict.get("_reset", None)
            reset_mask = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device) if reset_mask is None else reset_mask.to(self.device).view(-1)

        for e in torch.nonzero(reset_mask).squeeze(-1).tolist():
            self._reset_one_env(e)
        return self._make_tensordict()

    def _reset_one_env(self, e: int) -> None:
        gd = next(self._generators[e])
        for key in ["edge_index", "edge_realisation", "stochastic_edges", "A", "s", "g"]:
            if hasattr(gd, key) and isinstance(getattr(gd, key), torch.Tensor):
                setattr(gd, key, getattr(gd, key).to(self.device))
        self.graph_data[e] = gd

        starts = gd.s.clone().detach().long()
        if len(starts) < self.num_agents:
            starts = starts.repeat(self.num_agents // len(starts) + 1)
        self.current_locations[e] = starts[: self.num_agents].to(self.device)
        self.previous_locations[e] = self.current_locations[e].clone()

        self.destination_mask[e].zero_()
        self.destination_mask[e, gd.g.long()] = 1
        self.goals_visited_mask[e].zero_()
        self.goals_occupied_mask[e].zero_()

        W = to_dense_adj(gd.edge_index, edge_attr=gd.A, max_num_nodes=self.max_nodes)[0].float().to(self.device)
        self.static_weights[e] = W
        self.adj_matrix[e] = (W > 0).float()

        self.edge_status[e] = self.adj_matrix[e].int() * STATUS_NORMAL
        stochastic_mask = to_dense_adj(gd.edge_index, edge_attr=gd.stochastic_edges, max_num_nodes=self.max_nodes)[0].int().to(self.device)
        self.edge_status[e][(stochastic_mask == 1) & (self.adj_matrix[e] == 1)] = STATUS_UNCERTAIN

        self.agent_terminated[e].zero_()
        self.current_phase[e] = 0

        self._precompute_ground_truth_distances_one(e)
        self._observe_edge_status_update_one(e)

    def _compute_heuristic_costs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        E, N = self.num_envs, self.max_nodes
        inf = float('inf')
        w_opt = torch.full((E, N, N), inf, device=self.device, dtype=torch.float32)
        w_pess = torch.full((E, N, N), inf, device=self.device, dtype=torch.float32)

        idx = torch.arange(N, device=self.device)
        w_opt[:, idx, idx] = 0.0
        w_pess[:, idx, idx] = 0.0

        mask_opt = (self.edge_status != 3) & (self.static_weights > 0)
        mask_pess = (self.edge_status != 3) & (self.edge_status != 0) & (self.static_weights > 0)

        w_opt = torch.where(mask_opt, self.static_weights, w_opt)
        w_pess = torch.where(mask_pess, self.static_weights, w_pess)

        for k in range(N):
            w_opt = torch.min(w_opt, w_opt[:, :, k:k+1] + w_opt[:, k:k+1, :])
            w_pess = torch.min(w_pess, w_pess[:, :, k:k+1] + w_pess[:, k:k+1, :])

        unvisited = (self.destination_mask == 1) & (self.goals_visited_mask == 0)
        mask_unvisited = unvisited.unsqueeze(1).expand(E, N, N)
        
        cost_opt, _ = torch.min(torch.where(mask_unvisited, w_opt, torch.tensor(inf, device=self.device)), dim=2)
        cost_pess, _ = torch.min(torch.where(mask_unvisited, w_pess, torch.tensor(inf, device=self.device)), dim=2)

        cost_opt = torch.where(cost_opt == inf, torch.zeros_like(cost_opt), cost_opt)
        cost_pess = torch.where(cost_pess == inf, torch.zeros_like(cost_pess), cost_pess)
        return cost_opt, cost_pess

    def _make_tensordict(self) -> TensorDict:
        E, A, N = self.num_envs, self.num_agents, self.max_nodes

        opt_cost, pess_cost = self._compute_heuristic_costs()
        opt_cost_exp = opt_cost.unsqueeze(1).expand(E, A, N)
        pess_cost_exp = pess_cost.unsqueeze(1).expand(E, A, N)

        dest_mask = self.destination_mask.unsqueeze(1).expand(E, A, N)
        visit_mask = self.goals_visited_mask.unsqueeze(1).expand(E, A, N)

        self_pos = torch.zeros((E, A, N), device=self.device)
        self_pos.scatter_(2, self.current_locations.unsqueeze(-1), 1.0)

        total_counts = torch.zeros((E, N), device=self.device)
        for e in range(E):
            for loc in self.current_locations[e]:
                total_counts[e, loc] += 1.0
        density = total_counts.unsqueeze(1).expand(E, A, N) - self_pos

        terminated_feat = self.agent_terminated.float().unsqueeze(-1).expand(E, A, N)

        # Actor Local Observation (7 features)
        node_features = torch.stack(
            [dest_mask, visit_mask, opt_cost_exp, pess_cost_exp, self_pos, density, terminated_feat], dim=-1
        )

        action_mask = torch.zeros((E, A, N + 1), dtype=torch.bool, device=self.device)
        e_idx = torch.arange(E, device=self.device).unsqueeze(1)
        loc = self.current_locations

        # Basic validity: exists in adjacency matrix and is not blocked
        valid_move = (self.adj_matrix[e_idx, loc] == 1) & (self.edge_status[e_idx, loc] != STATUS_BLOCKED)
        valid_move = valid_move & (~self.agent_terminated.unsqueeze(-1))

        # Removed the "no-backtracking" logic (is_prev and has_other_options)
        # The mask now allows moving back to the previous location if it is adjacent and unblocked
        action_mask[:, :, :N] = valid_move

        # Termination and goal logic
        at_goal = torch.gather(self.destination_mask, dim=1, index=loc).bool()
        goal_free = ~torch.gather(self.goals_occupied_mask, dim=1, index=loc)
        terminated_agents = self.agent_terminated
        
        action_mask[:, :, N] = terminated_agents
        action_mask[:, :, N] |= (~terminated_agents) & at_goal & goal_free
    
        agents_td = TensorDict(
            {
                "node_features": node_features,
                "edge_status": self.edge_status.unsqueeze(1).expand(E, A, N, N),
                "edge_weights": self.static_weights.unsqueeze(1).expand(E, A, N, N),
                "agent_id": torch.arange(A, device=self.device, dtype=torch.int64).view(1, A, 1).expand(E, A, 1),
                "phase": self.current_phase.view(E, 1, 1).expand(E, A, 1),
                "action_mask": action_mask,
                "goals_visited_mask": visit_mask.to(torch.int32),
                "destination_mask": dest_mask.to(torch.int32),
            },
            batch_size=[E, A],
        )

        # # Critic Global State (展平拼接, 2N^2+6N)
        # all_agents_feat = torch.zeros((E, A, N, 2), device=self.device)
        # all_agents_feat.scatter_(2, self.current_locations.unsqueeze(-1).unsqueeze(-1).expand(E, A, 1, 2), 1.0)
        # all_agents_feat[:, :, :, 1] = all_agents_feat[:, :, :, 0] * self.agent_terminated.unsqueeze(-1).float()
        
        # global_state_flat = torch.cat([
        #     all_agents_feat.sum(dim=1).view(E, -1),                 # global_nodes [E, 2N]
        #     self.edge_status.float().view(E, -1),                   # status [E, N^2]
        #     opt_cost.view(E, -1),                                   # opt [E, N]
        #     pess_cost.view(E, -1),                                  # pess [E, N]
        #     self.static_weights.view(E, -1),                        # weights [E, N^2]
        #     self.destination_mask.float().view(E, -1),              # dest [E, N]
        #     self.goals_visited_mask.float().view(E, -1)             # visited [E, N]
        # ], dim=-1)

        return TensorDict({
            "agents": agents_td,
            # "state": global_state_flat,
            "penalty": self.current_phase.float().view(E, 1)
        }, batch_size=[E])

    def _step(self, tensordict: TensorDict) -> TensorDict:
        actions = tensordict.get(("agents", "action"))
        if actions.dim() == 1: actions = actions.view(self.num_envs, self.num_agents)

        step_costs = self._step_logic(actions)
        self.current_phase += 1

        done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        is_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        for e in range(self.num_envs):
            is_success[e] = torch.equal(self.destination_mask[e], self.goals_visited_mask[e])
            all_stopped = bool(self.agent_terminated[e].all().item())
            timeout = bool((self.current_phase[e] >= self.num_phases).item())
            done[e] = is_success[e] or all_stopped or timeout

        rewards = (-step_costs.mean(dim=1, keepdim=True)).unsqueeze(-1).expand(-1, step_costs.size(1), -1).clone()

        penalty = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        for e in range(self.num_envs):
            if done[e] and (not is_success[e]):
                unvisited_goals = torch.nonzero((self.destination_mask[e] == 1) & (self.goals_visited_mask[e] == 0)).squeeze(1)
                if unvisited_goals.numel() > 0:
                    agent_penalty = []
                    for i in range(self.num_agents):
                        dists = self.gt_distances[e, self.current_locations[e, i], unvisited_goals]
                        agent_penalty.append(dists[dists != float("inf")].sum().item() if dists[dists != float("inf")].numel() > 0 else 0.0)
                    if agent_penalty: penalty[e] = sum(agent_penalty) / len(agent_penalty)

        rewards -= penalty.view(-1, 1, 1)

        out_td = self._make_tensordict()
        out_td.set(("agents", "reward"), rewards)
        done_tensor = done.view(self.num_envs, 1)
        out_td.set("done", done_tensor)
        out_td.set("terminated", done_tensor)
        out_td.set("penalty", penalty.view(self.num_envs, 1))

        return out_td

    def _step_logic(self, actions: torch.Tensor) -> torch.Tensor:
        E, A, N = self.num_envs, self.num_agents, self.max_nodes
        step_costs = torch.zeros((E, A), device=self.device, dtype=torch.float32)

        for e in range(E):
            active_indices = torch.nonzero(~self.agent_terminated[e]).squeeze(1).tolist()
            np.random.shuffle(active_indices)
            for idx in active_indices:
                curr = int(self.current_locations[e, idx].item())
                act = int(actions[e, idx].item())

                self.previous_locations[e, idx] = curr

                if act == N: # Terminate
                    if self.destination_mask[e, curr] == 1 and not self.goals_occupied_mask[e, curr]:
                        self.agent_terminated[e, idx] = True
                        self.goals_occupied_mask[e, curr] = True
                        self.goals_visited_mask[e, curr] = 1
                    step_costs[e, idx] = 0.0
                    continue

                if self.adj_matrix[e, curr, act] == 0 and curr != act: continue

                status = int(self.edge_status[e, curr, act].item())
                is_blocked = (status == STATUS_BLOCKED)
                
                if status == STATUS_UNCERTAIN:
                    is_blocked = self._check_blockage_ground_truth(e, curr, act)
                    new_st = STATUS_BLOCKED if is_blocked else STATUS_OBSERVED_OPEN
                    self.edge_status[e, curr, act] = new_st
                    self.edge_status[e, act, curr] = new_st

                if is_blocked:
                    step_costs[e, idx] = 1.0
                else:
                    step_costs[e, idx] = float(self.static_weights[e, curr, act].item())
                    self.current_locations[e, idx] = act
                    if self.destination_mask[e, act] == 1: self.goals_visited_mask[e, act] = 1

            self._observe_edge_status_update_one(e)
        return step_costs

    def _precompute_ground_truth_distances_one(self, e: int) -> None:
        gd, N = self.graph_data[e], self.max_nodes
        dist = torch.full((N, N), float("inf"), device=self.device, dtype=torch.float32)
        dist.fill_diagonal_(0)
        mask = (gd.edge_realisation != 3)
        valid_u, valid_v, valid_w = gd.edge_index[0][mask], gd.edge_index[1][mask], gd.A[mask].float()
        dist[valid_u, valid_v] = dist[valid_v, valid_u] = valid_w
        for k in range(N):
            dist = torch.min(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))
        self.gt_distances[e] = dist

    def _check_blockage_ground_truth(self, e: int, u: int, v: int) -> bool:
        gd = self.graph_data[e]
        mask = (gd.edge_index[0] == u) & (gd.edge_index[1] == v)
        return bool((gd.edge_realisation[mask][0] == 3).item()) if mask.any() else False

    def _observe_edge_status_update_one(self, e: int) -> None:
        for loc in self.current_locations[e]:
            neighbors = torch.nonzero(self.adj_matrix[e, int(loc.item())]).squeeze(1).tolist()
            for n in neighbors:
                if int(self.edge_status[e, int(loc.item()), n].item()) == STATUS_UNCERTAIN:
                    new_val = STATUS_BLOCKED if self._check_blockage_ground_truth(e, int(loc.item()), n) else STATUS_OBSERVED_OPEN
                    self.edge_status[e, int(loc.item()), n] = self.edge_status[e, n, int(loc.item())] = new_val

    def _set_seed(self, seed: int) -> None: torch.manual_seed(seed)

    # ---------------------------
    # Rendering
    # ---------------------------
    def render(self, env_idx: int = 0):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import networkx as nx
        import numpy as np

        e = int(env_idx)
        if self.graph_data[e] is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        if plt.get_backend() != 'Agg':
            plt.switch_backend('Agg')

        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

        current_locs = self.current_locations[e].detach().cpu().numpy()
        terminated = self.agent_terminated[e].detach().cpu().numpy()
        dest_mask = self.destination_mask[e].detach().cpu().numpy()
        visited_mask = self.goals_visited_mask[e].detach().cpu().numpy()
        edge_status_dense = self.edge_status[e].detach().cpu().numpy()
        edge_index = self.graph_data[e].edge_index.detach().cpu().numpy()
        edge_weights = self.graph_data[e].A.detach().cpu().numpy()

        G = nx.Graph()
        G.add_nodes_from(range(self.max_nodes))
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            G.add_edge(u, v, weight=edge_weights[i], status=edge_status_dense[u, v])

        pos = nx.spring_layout(G, seed=42 + e)

        def get_node_color(i):
            if dest_mask[i] == 1:
                return "lightgrey" if visited_mask[i] == 1 else "gold"
            return "lightblue"

        def get_node_border_params(i):
            agents_at_node = [idx for idx, loc in enumerate(current_locs) if loc == i]
            active_agents = [idx for idx in agents_at_node if not terminated[idx]]
            terminated_agents = [idx for idx in agents_at_node if terminated[idx]]
            if active_agents:
                color = self.agent_colors[active_agents[0] % len(self.agent_colors)]
                return color, 4
            if terminated_agents:
                return "grey", 6
            return "black", 1

        border_styles = [get_node_border_params(i) for i in G.nodes]
        edgecolors = [style[0] for style in border_styles]
        linewidths = [style[1] for style in border_styles]

        def get_edge_color(status):
            if status == STATUS_NORMAL: return "grey"
            if status == STATUS_UNCERTAIN: return "blue"
            if status == STATUS_OBSERVED_OPEN: return "green"
            if status == STATUS_BLOCKED: return "red"
            return "grey"

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=[get_node_color(i) for i in G.nodes],
            edgecolors=edgecolors, linewidths=linewidths, node_size=500,
        )

        edges = list(G.edges(data=True))
        if len(edges) > 0:
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=[(u, v) for u, v, d in edges],
                edge_color=[get_edge_color(d["status"]) for u, v, d in edges],
                width=2,
            )
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
            nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=8)

        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

        ax.set_title(
            f"TorchRL CTP (env={e})\nPhase: {int(self.current_phase[e].item())} | Coverage: {int(np.sum(visited_mask))}/{int(np.sum(dest_mask))}"
        )

        legend_elements = []
        for i, color in enumerate(self.agent_colors[: self.num_agents]):
            legend_elements.append(
                patches.Patch(facecolor="white", edgecolor=color, linewidth=2, label=f"Agent {i}")
            )
        legend_elements.extend([
            patches.Patch(facecolor="gold", label="Goal (Unvisited)"),
            patches.Patch(facecolor="lightgrey", label="Goal (Visited)"),
            patches.Patch(color="red", label="Blocked Edge"),
            patches.Patch(color="green", label="Open Edge"),
            patches.Patch(color="blue", label="Uncertain Edge"),
        ])
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        fig.canvas.draw()
        
        try:
            buf = fig.canvas.tostring_rgb()
        except AttributeError:
            buf = fig.canvas.buffer_rgba()

        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype=np.uint8)
        
        if len(image) == width * height * 4:
            image = image.reshape(height, width, 4)[:, :, :3]
        else:
            image = image.reshape(height, width, 3)

        plt.close(fig)
        return image