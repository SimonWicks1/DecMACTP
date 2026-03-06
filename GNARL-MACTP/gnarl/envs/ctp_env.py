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


class CTPEnv(PhasedNodeSelectEnv):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        max_nodes: int,
        graph_generator: GraphGenerator,
        **kwargs,
    ):
        super(CTPEnv, self).__init__(
            max_nodes=max_nodes,
            num_phases=1,
            graph_generator=graph_generator,
            observe_final_selection=False,
            **kwargs,
        )

    def _init_observation_space(
        self,
    ) -> tuple[gym.spaces.Dict, dict[str, tuple[str, str, str]]]:
        obs_space = spaces.Dict(
            {
                "current_node": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
                "optimistic_cost": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.max_nodes, self.max_nodes),
                    dtype=np.float32,
                ),
                "pessimistic_cost": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.max_nodes, self.max_nodes),
                    dtype=np.float32,
                ),
                "edge_status": spaces.Box(
                    low=0,
                    high=3,
                    shape=(self.max_nodes, self.max_nodes),
                    dtype=np.int32,
                ),  # deterministic, unknown, traversable, blocked
                "total_cost": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )
        spec = {
            "current_node": ("state", "node", "categorical", 2),
            "optimistic_cost": ("state", "edge", "scalar"),
            "pessimistic_cost": ("state", "edge", "scalar"),
            "edge_status": ("state", "edge", "categorical", 4),
            "total_cost": ("state", "graph", "scalar"),  # not encoded
        }  # TODO: do not observe edge realisation at all
        return obs_space, spec

    def _realisation_from_threshold(
        self,
        weights: th.Tensor,
        edge_probs: th.Tensor,
        edge_statuses: th.Tensor,
        threshold: float,
    ) -> th.Tensor:
        edge_mask = (
            (edge_probs <= threshold) * (edge_statuses == 1)
            + (edge_statuses == 2)
            + (edge_statuses == 0)
        )
        return weights * edge_mask.float()

    def _shortest_path_to_goal(
        self,
        weights: th.Tensor,
        goal: int,
    ) -> th.Tensor:
        G = nx.from_numpy_array(weights.numpy(), edge_attr="weight")
        source = [goal]
        shortest_paths = th.zeros_like(weights)
        lengths = nx.multi_source_dijkstra_path_length(G, source, weight="weight")
        for node, l in lengths.items():
            shortest_paths[node] = l
        return shortest_paths

    def _get_cost_at_threshold(self, threshold: float) -> np.ndarray:
        weights = to_dense_adj(
            self.graph_data.edge_index,
            edge_attr=self.graph_data.A,
            max_num_nodes=self.num_nodes,
        )[0]
        edge_probs = to_dense_adj(
            self.graph_data.edge_index,
            edge_attr=self.graph_data.edge_probs,
            max_num_nodes=self.num_nodes,
        )[0]
        edge_statuses = self.edge_status
        realisation = self._realisation_from_threshold(
            weights, edge_probs, edge_statuses, threshold
        )
        shortest_paths = self._shortest_path_to_goal(
            realisation, np.argmax(self.graph_data.g).item()
        )
        return shortest_paths.numpy()

    def _get_observation(self) -> dict:
        current_node = np.zeros((self.graph_data.num_nodes,), dtype=np.int32)
        current_node[self.current_location] = 1
        optimistic_cost = self._get_cost_at_threshold(1.0)
        pessimistic_cost = self._get_cost_at_threshold(0.0)
        return {
            "current_node": pad_to_max_nodes(current_node, self.max_nodes),
            "optimistic_cost": pad_to_max_nodes(optimistic_cost, self.max_nodes),
            "pessimistic_cost": pad_to_max_nodes(pessimistic_cost, self.max_nodes),
            "edge_status": pad_to_max_nodes(self.edge_status, self.max_nodes),
            "total_cost": np.array([self.total_cost], dtype=np.float32),
        }

    def _observe_edge_status(self):
        # Update edge_status based on edge_realisation from current_location
        for neighbour in range(self.graph_data.num_nodes):
            if self.adj[self.current_location, neighbour] == 1:
                edge_idx = (
                    (self.graph_data.edge_index[0] == self.current_location)
                    & (self.graph_data.edge_index[1] == neighbour)
                ).nonzero(as_tuple=True)[0]
                self.edge_status[self.current_location, neighbour] = (
                    self.graph_data.edge_realisation[edge_idx].item()
                )
                self.edge_status[neighbour, self.current_location] = (
                    self.graph_data.edge_realisation[edge_idx].item()
                )

    def _reset_state(self, seed=None, options=None):
        self.current_location = np.argmax(self.graph_data.s).item()
        self.edge_status = to_dense_adj(
            self.graph_data.edge_index,
            edge_attr=self.graph_data.stochastic_edges,
            max_num_nodes=self.num_nodes,
        ).numpy()[0]
        self._observe_edge_status()

        self.total_cost = 0.0

        if self.render_mode == "human":
            if not hasattr(self, "ep_counter"):
                self.ep_counter = 0
            else:
                self._save_render()
                self._gif_frames = []
                self.ep_counter += 1
        return {}

    def get_max_episode_steps(n: int, e: int) -> int:
        return n * 2

    def is_success(self) -> bool | None:
        return self.current_location == np.argmax(self.graph_data.g).item()

    def is_terminal(self) -> bool:
        return self.is_success() == True

    def _step_env(self, action: int) -> dict:
        if self.adj[self.current_location, action] == 0:
            raise ValueError(
                f"Invalid action: {action} from current location {self.current_location}"
            )
        # get the edge index of the action
        edge_idx = (
            (self.graph_data.edge_index[0] == self.current_location)
            & (self.graph_data.edge_index[1] == action)
        ).nonzero(as_tuple=True)[0]
        if self.graph_data.edge_realisation[edge_idx] in [3, 1]:  # blocked or unknown
            # edge is blocked, stay put
            self.total_cost += 1.0
            return {}
        self.current_location = action
        self.total_cost += self.graph_data.A[edge_idx].item()
        # observe nearby edges
        self._observe_edge_status()

        return {}

    def _draw_graph(self):
        G = nx.from_numpy_array(
            to_dense_adj(
                self.graph_data.edge_index,
            )[0].numpy()
        )
        # add weight, status to edges
        for u, v, d in G.edges(data=True):
            edge_idx = (
                (self.graph_data.edge_index[0] == u)
                & (self.graph_data.edge_index[1] == v)
            ).nonzero(as_tuple=True)[0]
            d["weight"] = self.graph_data.A[edge_idx].item()
            d["status"] = self.edge_status[u, v]

        pos = nx.spring_layout(G, seed=1)

        def node_colour(i):
            if i == self.current_location:
                return "purple"
            if i == np.argmax(self.graph_data.s).item():
                return "orange"
            if i == np.argmax(self.graph_data.g).item():
                return "green"
            return "lightblue"

        def status_to_colour(status):
            if status == 0:
                return "gray"  # traversable
            elif status == 1:
                return "blue"  # unknown
            elif status == 2:
                return "green"  # traversable
            else:
                return "red"  # blocked

        nx.draw(
            G,
            with_labels=True,
            node_color=[node_colour(i) for i in range(self.graph_data.num_nodes)],
            edge_color=[
                status_to_colour(d["status"]) for _, _, d in G.edges(data=True)
            ],
            pos=pos,
            node_size=500,
            font_size=12,
            font_weight="bold",
        )
        nx.draw_networkx_edge_labels(
            G,
            pos=pos,
            edge_labels={
                k: f"{w:.2f}" for k, w in nx.get_edge_attributes(G, "weight").items()
            },
            rotate=False,
            font_size=8,
            bbox=dict(boxstyle="square,pad=0", linewidth=0, facecolor="none"),
        )

    def render(self):
        # print("+++++++++++Here in render+++++++++++")
        import matplotlib.pyplot as plt

        if self.render_mode != "human":
            print("Render mode not human, skipping render.")
            return

        if not hasattr(self, "_fig") or getattr(self, "_fig", None) is None:
            self._fig, self._ax = plt.subplots()

        # Draw current frame
        self._ax.clear()
        plt.sca(self._ax)
        self._draw_graph()

        # Always append frame to GIF buffer (no interactive display)
        self._fig.canvas.draw()
        width, height = self._fig.canvas.get_width_height()
        frame = None
        canvas = self._fig.canvas
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

        if not hasattr(self, "_gif_frames"):
            self._gif_frames = []

        self._gif_frames.append(frame.copy())

    def _save_render(self):
        print("+++++++++++++saving render+++++++++++++")
        fps = max(1, int(self.metadata.get("render_fps", 2)))
        if hasattr(self, "_gif_frames") and self._gif_frames:
            from PIL import Image

            frames = [Image.fromarray(f) for f in self._gif_frames]
            if frames:
                base_dir = "." if self.folder is None else self.folder
                os.makedirs(base_dir, exist_ok=True)
                self._gif_path = os.path.join(
                    base_dir, f"ctp_render_{self.env_id}_{self.ep_counter}.gif"
                )
                frames[0].save(
                    self._gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(1000 / fps),
                    loop=0,
                )

    def close(self):
        """Finalize rendering resources and write GIF if needed."""

        # Close matplotlib figure if present
        import matplotlib.pyplot as plt

        if hasattr(self, "_fig") and getattr(self, "_fig", None) is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    @staticmethod
    def _objective_function(**kwargs) -> float:
        return -kwargs["total_cost"].item()

    def objective_function(self, **kwargs) -> float:
        return self._objective_function(**kwargs)

    def action_masks(self):
        # Can select any edge that is traversable or deterministic
        valid_edges = (
            (self.edge_status[self.current_location] == 0)
            | (self.edge_status[self.current_location] == 2)
        ) & (self.adj.to_dense().numpy()[self.current_location])
        return pad_to_max_nodes(valid_edges, self.max_nodes)

    @staticmethod
    def expert_policy(obs: dict[str, np.ndarray], *args, **kwargs) -> np.ndarray:
        return

    @staticmethod
    def pre_transform(data: GraphProblemData) -> GraphProblemData:
        """Add the oracle policy value to the data."""
        edge_mask = to_dense_adj(
            data.edge_index,
            edge_attr=((data.edge_realisation == 2) | (data.edge_realisation == 0))
            * data.A,
            max_num_nodes=data.num_nodes,
        )[0].numpy()
        G = nx.from_numpy_array(edge_mask, edge_attr="weight")
        lengths = nx.single_source_dijkstra_path_length(
            G, np.argmax(data.s).item(), weight="weight"
        )
        obj = -lengths.get(np.argmax(data.g).item(), np.inf)
        data.expert_objective = th.tensor(obj, dtype=th.float32)
        return data
