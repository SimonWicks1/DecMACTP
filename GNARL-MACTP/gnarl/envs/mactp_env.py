import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gnarl.envs.alg_env import PhasedNodeSelectEnv
from gnarl.envs.generate.graph_generator import GraphGenerator
import torch as th
from torch_geometric.utils import to_dense_adj
from gnarl.util.algorithms import pad_to_max_nodes
from gnarl.util.graph_data import GraphProblemData
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def pad_to_shape(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(arr)
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
    if any(pw != (0, 0) for pw in pad_width):
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
    return arr

class MultiTravelerCTPEnv(PhasedNodeSelectEnv):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        max_nodes: int,
        graph_generator: GraphGenerator,
        num_agents: int = None,
        num_goals: int = None,
        num_phases: int = None,  # [Fix] 接收 num_phases
        **kwargs,
    ):
        if num_agents is None:
            num_agents = kwargs.get("num_agents", 2)
        if num_goals is None:
            num_goals = kwargs.get("num_goals", 2)

        default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        self.traveler_colors = default_colors[:num_agents]
        self.num_agents = num_agents
        self.num_goals = num_goals
        
        # [Fix] 设定默认 num_phases
        if num_phases is None:
            num_phases = max_nodes * 3 

        super(MultiTravelerCTPEnv, self).__init__(
            max_nodes=max_nodes,
            num_phases=num_phases, # [Fix] 传递给父类
            graph_generator=graph_generator,
            observe_final_selection=False,
            **kwargs,
        )

        # Override Action Space to MultiDiscrete
        self.action_space = spaces.MultiDiscrete([self.max_nodes] * self.num_agents)
        
        self.current_locations = []
        self.total_cost = 0.0

    @staticmethod
    def get_max_episode_steps(n: int, e: int) -> int:
        return n * 4

    def _init_observation_space(self) -> tuple[gym.spaces.Dict, dict[str, tuple[str, str, str]]]:
        obs_space = spaces.Dict(
            {
                # [User Request] 形状改回 (num_agents, max_nodes)
                "current_nodes": spaces.Box(
                    low=0, high=1,
                    shape=(self.num_agents, self.max_nodes),
                    dtype=np.int32,
                ),
                "destination_mask": spaces.Box(
                    low=0, high=1, shape=(self.max_nodes,), dtype=np.int32,
                ),
                "goals_visited_mask": spaces.Box(
                    low=0, high=1, shape=(self.max_nodes,), dtype=np.int32,
                ),
                "optimistic_cost": spaces.Box(
                    low=0, high=np.inf, shape=(self.max_nodes, self.max_nodes), dtype=np.float32,
                ),
                "pessimistic_cost": spaces.Box(
                    low=0, high=np.inf, shape=(self.max_nodes, self.max_nodes), dtype=np.float32,
                ),
                "edge_status": spaces.Box(
                    low=0, high=3, shape=(self.max_nodes, self.max_nodes), dtype=np.int32,
                ),
                "total_cost": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32,
                ),
            }
        )
        spec = {
            # 这里保持 user 的定义，确保 Encoder 能处理 (num_agents, max_nodes)
            "current_nodes": ("state", "node", "magent", self.num_agents), 
            "destination_mask": ("state", "node", "mask"),
            "goals_visited_mask": ("state", "node", "mask"),
            "optimistic_cost": ("state", "edge", "scalar"),
            "pessimistic_cost": ("state", "edge", "scalar"),
            "edge_status": ("state", "edge", "categorical", 4),
            "total_cost": ("state", "graph", "scalar"),
        }
        return obs_space, spec

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
        G = nx.from_numpy_array(weights.numpy(), edge_attr="weight")
        shortest_paths = th.zeros_like(weights)
        if goals:
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
        return shortest_paths.numpy()

    def _get_observation(self) -> dict:
        # 形状: (num_agents, max_nodes)
        current_nodes = np.zeros((self.num_agents, int(self.graph_data.num_nodes)), dtype=np.int32)
        for agent_i, node_idx in enumerate(self.current_locations):
            current_nodes[agent_i, int(node_idx)] = 1
            
        optimistic_cost = np.asarray(self._get_cost_at_threshold(1.0))
        pessimistic_cost = np.asarray(self._get_cost_at_threshold(0.0))
        max_n = int(self.max_nodes)
        
        obs = {
            # [User Request] Pad to (num_agents, max_nodes)
            "current_nodes": pad_to_shape(current_nodes, (self.num_agents, max_n)),
            "destination_mask": pad_to_shape(self.destination_mask, (max_n,)),
            "goals_visited_mask": pad_to_shape(self.goals_visited_mask, (max_n,)),
            "optimistic_cost": pad_to_shape(optimistic_cost, (max_n,)),
            "pessimistic_cost": pad_to_shape(pessimistic_cost, (max_n,)),
            "edge_status": pad_to_shape(self.edge_status, (max_n, max_n)),
            "total_cost": np.array([self.total_cost], dtype=np.float32),
        }
        return obs
    
    def _get_observation_with_info(self):
        _cls_obs = self._get_observation()
        last_selected_dict = {}
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
            last_selected_dict[key] = m

        from gnarl.util.graph_data import GraphProblemData_to_dense
        input_features = GraphProblemData_to_dense(
            self.graph_data,
            self.graph_spec,
            stage="input",
            obs_space=self.observation_space,
            max_nodes=self.max_nodes,
        )
        return {
            "phase": np.array([self.current_phase]),
            **last_selected_dict,
            **_cls_obs,
            **input_features,
        }

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
        starts = [int(s) for s in self.graph_data.s]
        if len(starts) < self.num_agents:
            raise ValueError(f"Not enough start nodes: {len(starts)} < {self.num_agents}")
            
        for i in range(self.num_agents):
            self.current_locations.append(starts[i])
        
        self.destination_mask = np.zeros(self.graph_data.num_nodes, dtype=np.int32)
        goals = self.graph_data.g
        for goal in goals:
            self.destination_mask[goal] = 1
        if len(goals) < self.num_goals:
            raise ValueError(f"Not enough goals nodes: {len(goals)} < {self.num_goals}. You should regenerate the graph data with sufficient goals.")
        
        self.goals_visited_mask = np.zeros_like(self.destination_mask)
        
        self.edge_status = to_dense_adj(
            self.graph_data.edge_index,
            edge_attr=self.graph_data.stochastic_edges,
            max_num_nodes=self.num_nodes,
        ).numpy()[0]
        
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

    def is_success(self) -> bool:
        return np.array_equal(self.destination_mask, self.goals_visited_mask)

    def is_terminal(self) -> bool:
        return self.is_success() == True

    # [Fix] 实现抽象方法 _step_env
    def _step_env(self, actions) -> dict:
        step_cost = 0.0
        if isinstance(actions, th.Tensor):
            actions = actions.cpu().numpy()
            
        for agent_idx, action in enumerate(actions):
            current_location = self.current_locations[agent_idx]
            
            if self.adj[current_location, action] == 0 and current_location != action:
                 raise ValueError(f"Invalid action {action} from {current_location}")
            
            edge_idx = ((self.graph_data.edge_index[0] == current_location) & 
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
                self.current_locations[agent_idx] = action
                if self.destination_mask[action] == 1:
                    self.goals_visited_mask[action] = 1

        self.total_cost += step_cost
        self._observe_edge_status()
        return {}

    def step(self, actions):
        if self.current_phase is None:
             raise ValueError("Environment is not in a valid phase.")

        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")
        
        info = self._step_env(actions)
        info["total_cost"] = self.total_cost

        # Update History
        self.last_selected[self.current_phase] = actions
        
        self.current_phase += 1
        obs = self._get_observation_with_info()
        
        reward = self._get_reward(**obs)
        
        # done = self.is_terminal() or (self.current_phase >= self.num_phases)
        # Edit by Xiao: Add failure penalty
        if self.is_terminal():
            done = True
        elif self.current_phase >= self.num_phases:
            done = True
            reward -= 15.0*(sum(self.destination_mask)-sum(self.goals_visited_mask))  # Penalty for not completing the task
        else:
            done = False
        # End Edit

        return obs, reward, done, False, info

    def action_masks(self):
        masks = []
        adj_dense = self.adj.to_dense().numpy()
        for i in range(self.num_agents):
            current_loc = self.current_locations[i]
            valid_neighbor_mask = (
                (self.edge_status[current_loc] != 3) 
            ) & (adj_dense[current_loc] == 1)
            mask = pad_to_max_nodes(valid_neighbor_mask, self.max_nodes)
            masks.append(mask.astype(bool))
        return np.array(masks)
    
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
            if i in self.current_locations:
                traveler_idx = self.current_locations.index(i)
                return self.traveler_colors[traveler_idx % len(self.traveler_colors)]
            return "black"

        def node_border_width(i):
            return 3 if i in self.current_locations else 1

        def status_to_colour(status):
            if status == 0:
                return "gray"
            elif status == 1:
                return "blue"
            elif status == 2:
                return "green"
            else:
                return "red"

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

        plt.title(f"Multi-Traveler CTP\n"
                 f"Coverage: {sum(self.goals_visited_mask)}/{sum(self.destination_mask)} "
                 f"({(sum(self.goals_visited_mask)/sum(self.destination_mask)*100) if len(self.goals_visited_mask)>0 else 0:.1f}%)",
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
                                label=f'Traveler {i+1} (Cost: {self.total_cost:.2f})')
                )

        legend_elements.extend([
            patches.Patch(facecolor='gold', label='Unvisited Destination'),
            patches.Patch(facecolor='lightgray', label='Visited Destination'),
            patches.Patch(facecolor='lightblue', label='Normal Node')
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
            # print(f"Error saving frame to GIF: {e}")
            pass

    def _save_render(self):
        fps = max(1, int(self.metadata.get("render_fps", 2)))

        if not hasattr(self, "_gif_frames") or not self._gif_frames:
            # print("No frames to save for GIF")
            return

        try:
            from PIL import Image
            frames = [Image.fromarray(f) for f in self._gif_frames]

            if frames:
                base_dir = "." if not hasattr(self, 'folder') or self.folder is None else self.folder
                os.makedirs(base_dir, exist_ok=True)

                if not hasattr(self, 'env_id'):
                    self.env_id = 0
                if not hasattr(self, 'ep_counter'):
                    self.ep_counter = 0

                gif_path = os.path.join(
                    base_dir, f"multi_ctp_render_{self.env_id}_{self.ep_counter}.gif"
                )
                print(f"\nSaving render to {gif_path} with {len(frames)} frames")

                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(1000 / fps),
                    loop=0,
                )
                print(f"Successfully saved GIF with {len(frames)} frames")

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
    def expert_policy(obs: dict[str, np.ndarray], *args, **kwargs) -> np.ndarray:
        return

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