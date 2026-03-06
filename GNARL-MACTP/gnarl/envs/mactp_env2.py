import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gnarl.envs.alg_env import PhasedNodeSelectEnv
from gnarl.envs.generate.graph_generator import GraphGenerator
import torch as th
from torch_geometric.utils import to_dense_adj
from gnarl.util.algorithms import pad_to_max_nodes
from gnarl.util.graph_data import GraphProblemData, GraphProblemData_to_dense
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

def pad_to_shape(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Helper: Pad array to target shape."""
    arr = np.asarray(arr)
    # Handle dimension expansion
    if arr.ndim < len(target_shape):
        arr = arr.reshape(arr.shape + (1,) * (len(target_shape) - arr.ndim))
    
    slicer = [slice(None)] * arr.ndim
    pad_width = []
    
    for ax, tgt in enumerate(target_shape):
        cur = arr.shape[ax]
        if cur >= tgt:
            slicer[ax] = slice(0, tgt)
            pad_width.append((0, 0))
        else:
            pad_width.append((0, tgt - cur))
            
    arr = arr[tuple(slicer)]
    
    # Only pad if necessary
    if any(pw != (0, 0) for pw in pad_width):
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
    return arr

class MultiTravelerCTPEnv2(PhasedNodeSelectEnv):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        max_nodes: int,
        graph_generator: GraphGenerator,
        num_agents: int = None,
        num_goals: int = None,
        num_phases: int = None,
        **kwargs,
    ):
        # Parameter Init
        if num_agents is None:
            num_agents = kwargs.get("num_agents", 2)
        if num_goals is None:
            num_goals = kwargs.get("num_goals", 2)

        default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        self.traveler_colors = default_colors[:num_agents]
        self.num_agents = num_agents
        self.num_goals = num_goals
        
        # Define Agent IDs
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = self.agents[:]

        if num_phases is None:
            num_phases = max_nodes * 3 
        
        # Call Super Init
        super(MultiTravelerCTPEnv2, self).__init__(
            max_nodes=max_nodes,
            num_phases=num_phases,
            graph_generator=graph_generator,
            observe_final_selection=False,
            **kwargs,
        )
        
        # Action Space: Dict {agent_id: Discrete(max_nodes + 1)}
        self.TERMINATE_ACTION = self.max_nodes
        self._single_action_space = spaces.Discrete(self.max_nodes + 1)
        self.action_space = spaces.Dict({agent: self._single_action_space for agent in self.agents})
        
        # Observation Space: Wrap the single-agent space into a Dict
        single_agent_space = self.observation_space
        self.observation_space = spaces.Dict({agent: single_agent_space for agent in self.agents})

        # Internal State
        self.current_locations = []
        self.agent_terminated = np.zeros(self.num_agents, dtype=bool)
        self.total_cost = 0.0

    @staticmethod
    def get_max_episode_steps(n: int, e: int) -> int:
        return n * 4

    def _init_observation_space(self) -> tuple[gym.spaces.Dict, dict[str, tuple[str, str, str]]]:
        """
        Defines an IMPROVED, Ego-Centric Observation Space.
        Format designed for GNNs: Node Features (Matrix) + Edge Features (Adj).
        """
        
        # Feature Dimensions
        # Node Features Channels:
        # 0: Is Goal? (1/0)
        # 1: Is Visited Goal? (1/0)
        # 2: Optimistic Cost (Float)
        # 3: Pessimistic Cost (Float)
        # 4: Self Position (1/0) - Ego-centric
        # 5: Teammate Density (Count) - Cooperative Context
        # 6: Terminated Status (Self) (1/0)
        num_node_features = 7 

        single_obs_space = spaces.Dict(
            {
                # Combined Node Features: (Max Nodes, Num Features)
                "node_features": spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(self.max_nodes, num_node_features), 
                    dtype=np.float32
                ),
                
                # Dynamic Edge Status (Adjacency Matrix channel 1)
                # 0: Unknown, 1: Open, 3: Blocked
                "edge_status": spaces.Box(
                    low=0, high=3, shape=(self.max_nodes, self.max_nodes), dtype=np.int32
                ),
                
                # Static Edge Weights (Adjacency Matrix channel 2)
                "edge_weights": spaces.Box(
                    low=0, high=np.inf, shape=(self.max_nodes, self.max_nodes), dtype=np.float32
                ),
                
                # Agent ID
                "agent_id": spaces.Box(
                    low=0, high=self.num_agents, shape=(1,), dtype=np.int32
                )
            }
        )
        
        # Spec for GNARL encoder compatibility
        spec = {
            "node_features": ("state", "node", "magent", 1, num_node_features),
            "edge_status": ("state", "edge", "categorical", 4),
            "edge_weights": ("state", "edge", "scalar"),
            "agent_id": ("state", "graph", "scalar") 
        }
        return single_obs_space, spec

    def _add_state_to_spec(self):
        state_spec = self.graph_spec.copy()
        max_observable_phases = (
            self.num_phases if self.observe_final_selection else self.num_phases - 1
        )
        state_spec.update(
            {
                f"last_selected_{i}": ("state", "node", "mask")
                for i in range(max_observable_phases)
            }
        )
        state_spec.update({"phase": ("state", "graph", "categorical", self.num_phases)})
        return state_spec

    def _realisation_from_threshold(self, weights, edge_probs, edge_statuses, threshold):
        edge_mask = ((edge_probs <= threshold) * (edge_statuses == 1) + 
                    (edge_statuses == 2) + (edge_statuses == 0))
        return weights * edge_mask.float()

    def _shortest_path_to_goals(self, weights, goals):
        """Calculates shortest path from ANY goal to all nodes."""
        G = nx.from_numpy_array(weights.numpy(), edge_attr="weight")
        
        # [FIX] Initialize as 1D array of size (num_nodes,), not 2D matrix
        num_nodes = weights.shape[0]
        shortest_paths = th.zeros(num_nodes)
        
        if goals:
            # Multi-source Dijkstra: Distance from set of goals to every node
            lengths = nx.multi_source_dijkstra_path_length(G, goals, weight="weight")
            for node, l in lengths.items():
                shortest_paths[node] = l
        return shortest_paths

    def _get_cost_at_threshold(self, threshold: float) -> np.ndarray:
        weights = to_dense_adj(self.graph_data.edge_index, edge_attr=self.graph_data.A, max_num_nodes=self.num_nodes)[0]
        edge_probs = to_dense_adj(self.graph_data.edge_index, edge_attr=self.graph_data.edge_probs, max_num_nodes=self.num_nodes)[0]
        realisation = self._realisation_from_threshold(weights, edge_probs, self.edge_status, threshold)
        unvisited_goals = [i for i, val in enumerate(self.destination_mask) 
                          if val == 1 and self.goals_visited_mask[i] == 0]
        
        if not unvisited_goals:
            return np.zeros((self.num_nodes,), dtype=np.float32)
        
        shortest_paths = self._shortest_path_to_goals(realisation, unvisited_goals)
        return shortest_paths.numpy().astype(np.float32)

    def _generate_global_state_components(self):
        """Pre-calculate shared components to avoid re-computing for every agent."""
        max_n = int(self.max_nodes)
        
        # 1. Global Masks & Costs
        dest_mask = self.destination_mask.astype(np.float32)
        visited_mask = self.goals_visited_mask.astype(np.float32)
        
        # These now return 1D arrays correctly
        opt_cost = self._get_cost_at_threshold(1.0)
        pess_cost = self._get_cost_at_threshold(0.0)
        
        # 2. Teammate Density Map (Global)
        teammate_counts = np.zeros(max_n, dtype=np.float32)
        for loc in self.current_locations:
            if loc < max_n:
                teammate_counts[loc] += 1.0
                
        # 3. Edge Data
        edge_stat = self.edge_status.astype(np.int32)
        
        # Graph weights (dense adj from graph data)
        edge_w = to_dense_adj(
            self.graph_data.edge_index, 
            edge_attr=self.graph_data.A, 
            max_num_nodes=self.num_nodes
        )[0].numpy().astype(np.float32)

        return dest_mask, visited_mask, opt_cost, pess_cost, teammate_counts, edge_stat, edge_w

    def _get_observation(self) -> dict:
        """Returns Dec-POMDP dictionary {agent_id: obs} with ego-centric features."""
        
        dest_mask, visited_mask, opt_cost, pess_cost, team_counts, edge_stat, edge_w = \
            self._generate_global_state_components()
            
        max_n = self.max_nodes
        # current number of nodes in this specific graph instance (<= max_n)
        curr_n = len(opt_cost)

        observations = {}
        
        for i, agent_id in enumerate(self.agents):
            current_loc = self.current_locations[i]
            is_terminated = 1.0 if self.agent_terminated[i] else 0.0
            
            # --- Construct Node Features Matrix (Ego-Centric) ---
            # Shape: (Max_Nodes, 7)
            node_feat = np.zeros((max_n, 7), dtype=np.float32)
            
            # Use slicing to fill only valid nodes (0 to curr_n)
            # Channel 0: Is Goal?
            node_feat[:curr_n, 0] = dest_mask[:curr_n]
            
            # Channel 1: Is Visited Goal?
            node_feat[:curr_n, 1] = visited_mask[:curr_n]
            
            # Channel 2: Optimistic Cost
            node_feat[:curr_n, 2] = opt_cost
            
            # Channel 3: Pessimistic Cost
            node_feat[:curr_n, 3] = pess_cost
            
            # Channel 4: Self Position (Ego-centric One-Hot)
            if not self.agent_terminated[i] and current_loc < curr_n:
                node_feat[current_loc, 4] = 1.0
                
            # Channel 5: Teammate Density (excludes self)
            # Global count - 1 if self is there
            if not self.agent_terminated[i] and current_loc < curr_n:
                 # Subtract self from count
                 team_counts_copy = team_counts.copy()
                 team_counts_copy[current_loc] = max(0.0, team_counts_copy[current_loc] - 1.0)
                 node_feat[:curr_n, 5] = team_counts_copy[:curr_n]
            else:
                 node_feat[:curr_n, 5] = team_counts[:curr_n]
            
            # Channel 6: Self Terminated Status
            node_feat[:, 6] = is_terminated

            # Construct obs dict
            obs = {
                "node_features": pad_to_shape(node_feat, (max_n, 7)).astype(np.float32),
                "edge_status": pad_to_shape(edge_stat, (max_n, max_n)).astype(np.int32),
                "edge_weights": pad_to_shape(edge_w, (max_n, max_n)).astype(np.float32),
                "agent_id": np.array([i], dtype=np.int32)
            }
            
            observations[agent_id] = obs
            
        return observations
    
    def _get_observation_with_info(self):
        """Get obs and inject GNARL features."""
        observations = self._get_observation()
        
        info_data = {}
        max_observable_phases = (
            self.num_phases if self.observe_final_selection else self.num_phases - 1
        )
        for i in range(max_observable_phases):
            key = f"last_selected_{i}"
            m = np.zeros((self.max_nodes), dtype=np.int32)
            actions_at_phase = self.last_selected[i]
            if actions_at_phase is not None:
                for node in actions_at_phase:
                    if 0 <= node < self.max_nodes:
                        m[node] = 1
            info_data[key] = m.astype(np.int32)

        info_data["phase"] = np.array([self.current_phase], dtype=np.int32)

        # Use the first agent's space as reference
        ref_space = self.observation_space[self.agents[0]]
        
        input_features = GraphProblemData_to_dense(
            self.graph_data,
            self.graph_spec, 
            stage="input",
            obs_space=ref_space, 
            max_nodes=self.max_nodes,
        )

        # Safe cast features
        for key, val in input_features.items():
            if key in ref_space:
                space = ref_space[key]
                if isinstance(space, spaces.Box):
                    input_features[key] = val.astype(space.dtype)

        for agent_id in self.agents:
            observations[agent_id].update(input_features)
            observations[agent_id].update(info_data)

        return observations

    def _observe_edge_status(self):
        for agent_idx in range(self.num_agents):
            current_location = self.current_locations[agent_idx]
            for neighbour in range(self.graph_data.num_nodes):
                if self.adj[current_location, neighbour] == 1:
                    edge_idx = ((self.graph_data.edge_index[0] == current_location) & 
                                (self.graph_data.edge_index[1] == neighbour)).nonzero(as_tuple=True)[0]
                    val = self.graph_data.edge_realisation[edge_idx].item()
                    self.edge_status[current_location, neighbour] = val
                    self.edge_status[neighbour, current_location] = val

    def _reset_state(self, seed=None, options=None):
        self.current_locations = []
        self.agent_terminated = np.zeros(self.num_agents, dtype=bool)
        
        starts = [int(s) for s in self.graph_data.s]
        if len(starts) < self.num_agents:
            starts = starts * (self.num_agents // len(starts) + 1)
            
        for i in range(self.num_agents):
            self.current_locations.append(starts[i])
        
        self.destination_mask = np.zeros(self.graph_data.num_nodes, dtype=np.int32)
        goals = self.graph_data.g
        for goal in goals:
            self.destination_mask[goal] = 1
        
        self.goals_visited_mask = np.zeros_like(self.destination_mask, dtype=np.int32)
        
        self.edge_status = to_dense_adj(
            self.graph_data.edge_index,
            edge_attr=self.graph_data.stochastic_edges,
            max_num_nodes=self.num_nodes,
        ).numpy()[0].astype(np.int32)
        
        self._observe_edge_status()
        self.total_cost = 0.0
        
        self.current_phase = 0
        self.last_selected = [None] * self.num_phases
        self.previous_obj = 0.0

        if self.render_mode == "human":
            if not hasattr(self, "ep_counter"):
                self.ep_counter = 0
            else:
                self._save_render()
                self._gif_frames = []
                self.ep_counter += 1
                
        return {}

    def is_terminal(self) -> bool:
        success, agents_stopped, timeout = self._is_episode_done()
        return success or agents_stopped or timeout

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_observation_with_info(), {}

    def is_success(self) -> bool:
        return np.array_equal(self.destination_mask, self.goals_visited_mask)

    def _update_termination_status(self, actions_list: np.ndarray):
        new_terminations = (actions_list == self.TERMINATE_ACTION)
        self.agent_terminated = self.agent_terminated | new_terminations

    def _get_active_agents(self, actions_list: np.ndarray):
        for i, action in enumerate(actions_list):
            if not self.agent_terminated[i] and action != self.TERMINATE_ACTION:
                yield i, action

    def _is_episode_done(self):
        all_goals_reached = self.is_success()
        all_agents_stopped = np.all(self.agent_terminated)
        timeout = self.current_phase >= self.num_phases
        return all_goals_reached, all_agents_stopped, timeout

    def _step_env(self, actions_list) -> dict:
        step_cost = 0.0
        actions = np.asarray(actions_list)
        
        self._update_termination_status(actions)
        
        for i, action in self._get_active_agents(actions):
            current_loc = self.current_locations[i]
            
            if self.adj[current_loc, action] == 0 and current_loc != action:
                 continue 
            
            edge_idx = ((self.graph_data.edge_index[0] == current_loc) & 
                        (self.graph_data.edge_index[1] == action)).nonzero(as_tuple=True)[0]
            
            is_blocked = False
            if len(edge_idx) > 0:
                 realisation = self.graph_data.edge_realisation[edge_idx].item()
                 if realisation in [3, 1]: is_blocked = True
            
            if is_blocked:
                step_cost += 1.0 
            else:
                if len(edge_idx) > 0:
                    step_cost += self.graph_data.A[edge_idx].item()
                self.current_locations[i] = action
                if self.destination_mask[action] == 1:
                    self.goals_visited_mask[action] = 1

        self.total_cost += step_cost
        self._observe_edge_status()
        return {"step_cost": step_cost}

    def step(self, actions):
        action_list = np.zeros(self.num_agents, dtype=np.int32)
        
        if isinstance(actions, dict):
            for i, agent_id in enumerate(self.agents):
                if agent_id in actions:
                    action_list[i] = actions[agent_id]
                else:
                    action_list[i] = self.TERMINATE_ACTION
        else:
            action_list = np.array(actions)
            if hasattr(action_list, "cpu"): action_list = action_list.cpu().numpy()

        action_list[self.agent_terminated] = self.TERMINATE_ACTION
        
        info_dict = self._step_env(action_list)
        self.current_phase += 1
        
        obs = self._get_observation_with_info()
        
        reward_val = -info_dict["step_cost"]
        
        is_success, is_terminated, is_timeout = self._is_episode_done()
        
        if is_timeout and not is_success:
            reward_val -= 2.0 * (sum(self.destination_mask) - sum(self.goals_visited_mask))
        
        if np.all(self.agent_terminated) and not is_success:
            reward_val -= 2.0 * self.num_agents

        rewards = {agent: float(reward_val) for agent in self.agents}
        
        done = is_success or is_terminated or is_timeout
        terminated = {agent: done for agent in self.agents}
        truncated = {agent: is_timeout for agent in self.agents}
        infos = {agent: info_dict for agent in self.agents}

        return obs, rewards, terminated, truncated, infos
    
    def action_masks(self) -> dict:
        masks = {}
        adj_dense = self.adj.to_dense().numpy()
        
        for i, agent_id in enumerate(self.agents):
            if self.agent_terminated[i]:
                mask = np.zeros(self.max_nodes + 1, dtype=bool)
                mask[self.TERMINATE_ACTION] = True
            else:
                current_loc = self.current_locations[i]
                valid_neighbor_mask = (self.edge_status[current_loc] != 3) & (adj_dense[current_loc] == 1)
                node_mask = pad_to_max_nodes(valid_neighbor_mask, self.max_nodes).astype(bool)
                
                on_goal = False
                if 0 <= current_loc < len(self.destination_mask):
                    on_goal = bool(self.destination_mask[current_loc] == 1)
                mask = np.append(node_mask, [on_goal])
            
            masks[agent_id] = mask
            
        return masks
    
    def _draw_graph(self):
        G = nx.from_numpy_array(
            to_dense_adj(
                self.graph_data.edge_index,
            )[0].numpy()
        )

        for u, v, d in G.edges(data=True):
            edge_idx = (
                (self.graph_data.edge_index[0] == u)
                & (self.graph_data.edge_index[1] == v)
            ).nonzero(as_tuple=True)[0]
            if len(edge_idx) > 0:
                d["weight"] = self.graph_data.A[edge_idx].item()
                d["status"] = int(self.edge_status[u, v])

        pos = nx.spring_layout(G, seed=1)

        def node_colour(i,destoinations,visited):
            if i in destoinations:
                if i in visited:
                    return "lightgray"
                else:
                    return "gold"
            return "lightblue"

        def node_border_color(i):
            for idx, loc in enumerate(self.current_locations):
                if loc == i:
                    if self.agent_terminated[idx]:
                        return "gray" 
                    return self.traveler_colors[idx % len(self.traveler_colors)]
            return "black"

        def node_border_width(i):
            return 4 if any(loc == i for loc in self.current_locations) else 1

        def status_to_colour(status):
            if status == 0: return "gray"
            elif status == 1: return "blue"
            elif status == 2: return "green"
            else: return "red"

        destoinations = np.where(self.destination_mask == 1)[0]
        visited = np.where(self.goals_visited_mask == 1)[0]
        nx.draw_networkx_nodes(
            G, pos,
            node_color=[node_colour(i,destoinations,visited) for i in range(self.graph_data.num_nodes)],
            edgecolors=[node_border_color(i) for i in range(self.graph_data.num_nodes)],
            linewidths=[node_border_width(i) for i in range(self.graph_data.num_nodes)],
            node_size=500
        )

        nx.draw_networkx_edges(
            G, pos,
            edge_color=[status_to_colour(d["status"]) for _, _, d in G.edges(data=True)]
        )

        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

        edge_labels = {}
        for u, v, d in G.edges(data=True):
            if "weight" in d:
                edge_labels[(u, v)] = f"{d['weight']:.2f}"

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            rotate=False,
            font_size=8,
            bbox=dict(boxstyle="square,pad=0", linewidth=0, facecolor="none"),
        )

        plt.title(f"Multi-Traveler CTP (Dec-POMDP)\n"
                 f"Coverage: {sum(self.goals_visited_mask)}/{sum(self.destination_mask)}",
                 fontsize=14)

    def render(self):
        if self.render_mode != "human":
            return

        if not hasattr(self, "_fig") or getattr(self, "_fig", None) is None:
            self._fig, self._ax = plt.subplots(figsize=(12, 10))

        self._ax.clear()
        plt.sca(self._ax)
        self._draw_graph()

        legend_elements = []
        for i, color in enumerate(self.traveler_colors):
            if i < len(self.current_locations):
                legend_elements.append(
                    patches.Patch(facecolor='white', edgecolor=color,
                                label=f'Agent {i} (Cost: {self.total_cost:.2f})')
                )

        legend_elements.extend([
            patches.Patch(facecolor='gold', label='Unvisited Goal'),
            patches.Patch(facecolor='lightgray', label='Visited Goal'),
        ])

        self._ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)

        self._save_frame_to_gif()

    def _save_frame_to_gif(self):
        if not hasattr(self, "_fig") or self._fig is None:
            return

        self._fig.canvas.draw()
        width, height = self._fig.canvas.get_width_height()
        frame = None
        canvas = self._fig.canvas

        try:
            if hasattr(canvas, "tostring_rgb"):
                buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                frame = buf.reshape((height, width, 3))
            elif hasattr(canvas, "buffer_rgba"):
                buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                rgba = buf.reshape((height, width, 4))
                frame = rgba[..., :3]
            elif hasattr(canvas, "print_to_buffer"):
                buf, (w, h) = canvas.print_to_buffer()
                arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
                frame = arr[..., :3]
                width, height = w, h
            else:
                return

            if frame is not None:
                if not hasattr(self, "_gif_frames"):
                    self._gif_frames = []
                self._gif_frames.append(frame.copy())

        except Exception as e:
            pass

    def _save_render(self):
        fps = max(1, int(self.metadata.get("render_fps", 2)))
        if not hasattr(self, "_gif_frames") or not self._gif_frames:
            return
        try:
            from PIL import Image
            frames = [Image.fromarray(f) for f in self._gif_frames]
            if frames:
                base_dir = "." if not hasattr(self, 'folder') or self.folder is None else self.folder
                os.makedirs(base_dir, exist_ok=True)
                if not hasattr(self, 'env_id'): self.env_id = 0
                if not hasattr(self, 'ep_counter'): self.ep_counter = 0

                gif_path = os.path.join(
                    base_dir, f"decpomdp_ctp_render_{self.env_id}_{self.ep_counter}.gif"
                )
                print(f"\nSaving render to {gif_path}")
                frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
        except Exception as e:
            print(f"Error saving GIF: {e}")

    def close(self):
        if hasattr(self, "_fig") and getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    @staticmethod
    def _objective_function(**kwargs) -> float:
        return -kwargs["total_cost"].item()

    def objective_function(self, **kwargs) -> float:
        return self._objective_function(**kwargs)

    @staticmethod
    def expert_policy(obs, *args, **kwargs):
        pass

    @staticmethod
    def pre_transform(data: GraphProblemData) -> GraphProblemData:
        edge_mask = to_dense_adj(
            data.edge_index,
            edge_attr=((data.edge_realisation == 2) | (data.edge_realisation == 0))
            * data.A,
            max_num_nodes=data.num_nodes,
        )[0].numpy()
        G = nx.from_numpy_array(edge_mask, edge_attr="weight")
        starts = np.where(data.s > 0)[0]
        goals = np.where(data.g > 0)[0]
        if len(goals) > 0:
            all_points = list(starts) + list(goals)
            subgraph = G.subgraph(all_points)
            try:
                mst = nx.minimum_spanning_tree(subgraph, weight='weight')
                total_cost = sum(d['weight'] for u, v, d in mst.edges(data=True))
            except:
                total_cost = 0.0
        else:
            total_cost = 0.0
        obj = -total_cost
        data.expert_objective = th.tensor(obj, dtype=th.float32)
        return data