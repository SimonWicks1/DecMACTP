import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gnarl.envs.alg_env import PhasedNodeSelectEnv
from gnarl.envs.generate.graph_generator import GraphGenerator
import torch as th
from gnarl.util.graph_format import unpad_array
from gnarl.util.algorithms import (
    pad_to_max_nodes,
    create_depth_counter,
    bfs,
    all_nodes_to_source,
    bellman_ford,
    mst_prim,
    check_valid_mst_predecessors,
    check_valid_dfs_solution,
)
from gnarl.util.graph_data import GraphProblemData
from torch_geometric.utils import to_dense_adj


class BFSEnv(PhasedNodeSelectEnv):
    """
    A custom Gymnasium environment for simulating the BFS algorithm.
    """

    def __init__(
        self,
        max_nodes: int,
        graph_generator: GraphGenerator,
        **kwargs,
    ):
        super().__init__(
            max_nodes=max_nodes,
            num_phases=2,
            graph_generator=graph_generator,
        )

    def _init_observation_space(
        self,
    ) -> tuple[gym.spaces.Dict, dict[str, tuple[str, str, str]]]:
        obs_space = spaces.Dict(
            {
                # outputs
                "predecessors_ptr": spaces.Box(
                    low=0,
                    high=self.max_nodes - 1,
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
                # state
                "reach": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
            }
        )
        spec = {
            "predecessors_ptr": ("state", "node", "pointer"),
            "reach": ("state", "node", "mask"),
        }
        return obs_space, spec

    def _get_observation(self) -> dict:
        return {
            "predecessors_ptr": pad_to_max_nodes(self.predecessors, self.max_nodes),
            "reach": pad_to_max_nodes(self.reach, self.max_nodes),
        }

    def _reset_state(self, seed=None, options=None):
        # Initialise state variables
        if hasattr(self.graph_data, "s"):
            self.reach = self.graph_data.s.numpy().astype(np.int32)
        else:
            self.reach = np.zeros(self.max_nodes, dtype=np.int32)
        self.predecessors = np.arange(self.graph_data.num_nodes, dtype=np.int32)

        if hasattr(self.graph_data, "s"):
            self.solution_depths = create_depth_counter(
                np.ones(self.graph_data.num_nodes, dtype=int),
                bfs(self.graph_data.s, self.adj),
            )

        # For caching
        self._is_success_inputs = None

        return {}

    @staticmethod
    def get_max_episode_steps(n: int, e: int) -> int:
        return 2 * (n - 1)

    def is_success(self) -> bool:
        if not hasattr(self.graph_data, "s"):
            return False

        state = (
            tuple(self.reach.tolist()),
            tuple(self.predecessors.tolist()),
            tuple(self.graph_data.s.tolist()),
            tuple(self.solution_depths.tolist()),
        )
        if self._is_success_inputs == state:
            return self._is_success_cache

        # Compute success criterion
        depths = create_depth_counter(self.reach, self.predecessors)
        result = bool(np.all(depths == self.solution_depths))

        # Cache
        self._is_success_inputs = state
        self._is_success_cache = result
        return result

    def is_terminal(self) -> bool:
        return self.is_success()

    def _step_env(self, action: int) -> dict:
        if self.current_phase != 1:
            return {}

        sel_node = self.last_selected[0]
        neighbour = action

        # Set the predecessor for the selected node
        if self.adj[sel_node, neighbour]:
            self.reach[sel_node] = 1
            self.reach[neighbour] = 1
            self.predecessors[neighbour] = sel_node

        return {}

    def action_masks(self):
        if self.current_phase == 0:
            # All nodes are selectable in phase 0
            return pad_to_max_nodes(
                np.ones((self.graph_data.num_nodes,)), self.max_nodes
            )
        elif self.current_phase == 1:
            # Only neighbours of the last selected node are selectable in phase 1
            return pad_to_max_nodes(
                self.adj[self.last_selected[0]].to_dense().numpy(),
                self.max_nodes,
            )
        raise ValueError("Invalid phase for action masks.")

    @staticmethod
    def expert_policy(obs: dict[str, np.ndarray], *args, **kwargs) -> np.ndarray:
        def check_closed(mask, adj):
            closed = np.zeros((len(mask),), dtype=bool)
            for i in range(len(mask)):
                if mask[i] == 1:
                    neighbours = np.where(adj[i] == 1)[0]
                    open_neighbours = neighbours[mask[neighbours] == 0]
                    if len(open_neighbours) == 0:
                        closed[i] = 1
            return closed

        def phase_2_policy(o, i, sel_node):
            neighbours = np.where(o["adj"][i][sel_node] == 1)[0]
            neighbours = neighbours[neighbours != sel_node]
            if len(neighbours) == 0:
                return None
            usable = neighbours[o["reach"][i][neighbours] == 0]

            if len(usable) == 0:
                return None

            # Return the usable neighbours with equal probabilities
            node_probs = np.zeros((len(o["reach"][i]),), dtype=np.float32)
            node_probs[usable] = 1.0 / len(usable)
            return node_probs

        def phase_1_policy(o, i):
            closed = check_closed(o["reach"][i], o["adj"][i])
            open_nodes = ~closed * o["reach"][i]
            if open_nodes.sum() == 0:
                raise ValueError("No open nodes found for phase 1.")

            depth_counter = create_depth_counter(
                o["reach"][i], o["predecessors_ptr"][i]
            )

            # Get the minimum depth of an open node
            minimum_depth = np.min(depth_counter[open_nodes == 1])

            # Find the index of the open node with the smallest depth
            eligible_nodes = np.where(
                (open_nodes == 1) & (depth_counter == minimum_depth)
            )[0]
            if len(eligible_nodes) == 0:
                raise ValueError(
                    "No eligible nodes found for depth counter calculation."
                )

            # Return the eligible nodes with equal probabilities
            node_probs = np.zeros((len(o["reach"][i]),), dtype=np.float32)
            node_probs[eligible_nodes] = 1.0 / len(
                eligible_nodes
            )  # equal probabilities
            return node_probs

        def single_policy(o, i):
            if o["phase"][i] == 0:  # phase 1
                return phase_1_policy(o, i)

            elif o["phase"][i] == 1:  # phase 2
                neighbour = phase_2_policy(o, i, np.argmax(o["last_selected_0"][i]))
                if neighbour is not None and neighbour.sum() > 0:
                    return neighbour
                else:
                    raise ValueError("No valid neighbours found for phase 2.")

            else:
                raise ValueError("Invalid phase for action selection.")

        actions = np.array(
            [single_policy(obs, i) for i in range(len(obs["last_selected_0"]))]
        )
        return actions


class DFSEnv(BFSEnv):
    """
    Adapts the BFS environment to simulate the DFS algorithm.
    Only changes the conditions for a terminal/successful state.
    """

    def is_success(self) -> bool | None:
        state = (
            tuple(self.reach.tolist()),
            tuple(self.predecessors.tolist()),
            self.adj.to_dense().numpy().tobytes(),
        )
        if self._is_success_inputs == state:
            return self._is_success_cache

        depths = create_depth_counter(self.reach, self.predecessors)
        max_node = self.adj._indices().max().item() + 1
        if np.any(np.array(depths)[:max_node] < 0):
            result = False
        else:
            result = check_valid_dfs_solution(self.adj, self.predecessors)

        # Cache
        self._is_success_inputs = state
        self._is_success_cache = result
        return result

    def is_terminal(self) -> bool:
        return self.is_success() == True

    @staticmethod
    def expert_policy(obs: dict[str, np.ndarray], *args, **kwargs) -> np.ndarray:

        def get_node_colour(mask, adj):
            colour = mask.copy()  # 0=white, 1=grey, 2=black
            colour[adj.shape[0] :] = 2
            for i in range(len(mask)):
                if mask[i] == 1:
                    neighbours = np.where(adj[i] == 1)[0]
                    open_neighbours = neighbours[mask[neighbours] == 0]
                    if len(open_neighbours) == 0:
                        colour[i] = 2
            return colour

        def phase_2_policy(o, i, sel_node):
            neighbours = np.where(o["adj"][i][sel_node] == 1)[0]
            neighbours = neighbours[neighbours != sel_node]
            if len(neighbours) == 0:
                if o["reach"][i][sel_node] == 0:
                    node_probs = np.zeros((len(o["reach"][i]),), dtype=np.float32)
                    node_probs[sel_node] = 1.0
                    return node_probs
                else:
                    return None

            usable = neighbours[o["reach"][i][neighbours] == 0]

            if len(usable) == 0:
                if o["reach"][i][sel_node] == 0:
                    # If the selected node is white, we can select it
                    usable = [int(sel_node)]
                else:
                    return None

            # Return the usable neighbours with equal probabilities
            node_probs = np.zeros((len(o["reach"][i]),), dtype=np.float32)
            node_probs[usable] = 1.0 / len(usable)
            return node_probs

        def phase_1_policy(o, i):
            adj = unpad_array(o["adj"][i])
            colour = get_node_colour(o["reach"][i], adj)
            depth_counter = create_depth_counter(colour != 0, o["predecessors_ptr"][i])

            if np.all(colour == 2):
                raise ValueError("No open nodes found for phase 1.")

            # Get the maximum depth of an open node
            maximum_depth = np.max(depth_counter[colour != 2])

            # Find the index of the open node with the greatest depth
            # Prioritises grey nodes (1) over white nodes (0)
            eligible_nodes = np.where((colour == 1) & (depth_counter == maximum_depth))[
                0
            ]
            if len(eligible_nodes) == 0:
                eligible_nodes = np.where(
                    (colour == 0) & (depth_counter == maximum_depth)
                )[0]
                if len(eligible_nodes) == 0:
                    raise ValueError(
                        "No eligible nodes found for depth counter calculation."
                    )

            # Return the eligible nodes with equal probabilities
            node_probs = np.zeros((len(colour),), dtype=np.float32)
            node_probs[eligible_nodes] = 1.0 / len(
                eligible_nodes
            )  # equal probabilities
            return node_probs

        def single_policy(o, i):
            if o["phase"][i] == 0:  # phase 1
                return phase_1_policy(o, i)

            elif o["phase"][i] == 1:  # phase 2
                neighbour = phase_2_policy(o, i, np.argmax(o["last_selected_0"][i]))
                if neighbour is not None and neighbour.sum() > 0:
                    return neighbour
                else:
                    raise ValueError("No valid neighbours found for phase 2.")

            else:
                raise ValueError("Invalid phase for action selection.")

        actions = np.array(
            [single_policy(obs, i) for i in range(len(obs["last_selected_0"]))]
        )
        return actions


class BellmanFordEnv(PhasedNodeSelectEnv):
    """
    A custom Gymnasium environment for simulating the Bellman-Ford algorithm.
    """

    def __init__(
        self,
        max_nodes: int,
        graph_generator: GraphGenerator,
        **kwargs,
    ):
        super().__init__(
            max_nodes=max_nodes, num_phases=2, graph_generator=graph_generator
        )

    def _init_observation_space(
        self,
    ) -> tuple[gym.spaces.Dict, dict[str, tuple[str, str, str]]]:
        obs_space = spaces.Dict(
            {  # outputs
                "predecessors_ptr": spaces.Box(
                    low=0,
                    high=self.max_nodes - 1,
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
                # state
                "distances": spaces.Box(
                    low=0,
                    high=self.max_nodes * self.max_nodes,
                    shape=(self.max_nodes,),
                    dtype=np.float64,
                ),
                "mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
            }
        )
        spec = {
            "predecessors_ptr": ("state", "node", "pointer"),
            "distances": ("state", "node", "scalar"),
            "mask": ("state", "node", "mask"),
        }
        return obs_space, spec

    def _get_observation(self) -> dict:
        return {
            "distances": pad_to_max_nodes(self.distances, self.max_nodes),
            "predecessors_ptr": pad_to_max_nodes(self.predecessors, self.max_nodes),
            "mask": pad_to_max_nodes(self.msk, self.max_nodes),
        }

    def _reset_state(self, seed=None, options=None):
        # Initialise state variables
        self.distances = np.full(self.graph_data.num_nodes, 0.0, dtype=np.float64)
        self.msk = self.graph_data.s.numpy().astype(np.int32)
        self.predecessors = np.arange(self.graph_data.num_nodes, dtype=np.int32)

        self.weights = th.sparse_coo_tensor(
            self.graph_data.edge_index,
            self.graph_data.A,
            (self.graph_data.num_nodes, self.graph_data.num_nodes),
        )

        # For termination
        if not hasattr(self.graph_data, "pi"):
            solution_predecessors, _ = bellman_ford(self.graph_data.s, self.weights)
        else:
            solution_predecessors = self.graph_data.pi
        self.solution_distances = all_nodes_to_source(
            solution_predecessors, self.graph_data.s, self.adj, self.weights
        )

        self._is_success_inputs = None

        return {}

    @staticmethod
    def get_max_episode_steps(n: int, e: int) -> int:
        return 2 * (n - 1) * e  # Worst-case complexity

    @property
    def max_episode_steps(self) -> int:
        if hasattr(self.graph_data, "expert_objective"):
            # 200% of expert policy length. Technically can prevent the agent from solving
            # the problem, but significantly shortens evaluation time.
            return 2 * self.num_phases * -self.graph_data.expert_objective.item()
        return self.get_max_episode_steps(
            self.graph_data.num_nodes, self.graph_data.num_edges
        )

    def is_success(self) -> bool | None:
        state = (
            tuple(self.predecessors.tolist()),
            tuple(self.graph_data.s.tolist()),
            self.adj.to_dense().numpy().tobytes(),
            self.weights.to_dense().numpy().tobytes(),
        )
        if self._is_success_inputs == state:
            return self._is_success_cache

        # Compute success criterion
        distances = all_nodes_to_source(
            self.predecessors, self.graph_data.s, self.adj, self.weights
        )
        result = bool(np.all(np.array(distances) == np.array(self.solution_distances)))

        # Cache
        self._is_success_inputs = state
        self._is_success_cache = result
        return result

    def is_terminal(self) -> bool:
        return self.is_success() == True

    def _step_env(self, action: int) -> dict:
        if self.current_phase != 1:
            return {}

        sel_node = self.last_selected[0]
        neighbour = action

        # Set the predecessor for the selected node
        if self.adj[sel_node, neighbour] == 1:
            weight = self.weights[sel_node, neighbour].item()
            self.distances[neighbour] = self.distances[sel_node] + weight
            self.msk[neighbour] = 1
            self.predecessors[neighbour] = sel_node

        return {}

    def get_reward(self, **kwargs) -> float:
        return -1  # override to track the number of steps taken, not used in training

    def action_masks(self):
        if self.current_phase == 0:
            # Nodes in the mask are selectable in phase 0
            return pad_to_max_nodes(self.msk, self.max_nodes)
        elif self.current_phase == 1:
            # Only neighbours of the last selected node are selectable in phase 1
            edges = self.adj[self.last_selected[0]].to_dense().numpy().copy()
            edges[self.last_selected[0]] = 0
            return pad_to_max_nodes(edges, self.max_nodes)
        raise ValueError("Invalid phase for action masks.")

    @staticmethod
    def pre_transform(data: GraphProblemData) -> GraphProblemData:
        """Add the expert policy value to the data."""
        _, num_steps = bellman_ford(
            data.s, to_dense_adj(data.edge_index, edge_attr=data.A)[0]
        )
        obj = -num_steps
        data.expert_objective = th.tensor(obj, dtype=th.float32)
        return data

    @staticmethod
    def _expert_policy_probabilities(obs: dict[str, np.ndarray], i: int) -> np.ndarray:
        def generate_start_indices(mask, start_at):
            active_indices = np.where(mask == 1)[0]
            start_index = np.argmax(start_at)

            output = np.zeros((len(active_indices), len(mask)), dtype=int)
            for i, idx in enumerate(active_indices):
                output[i, active_indices[(start_index + i) % len(active_indices)]] = 1

            return output

        def phase_2_policy(o, i, sel_node):
            neighbours = np.where(o["adj"][i][sel_node] == 1)[0]
            neighbours = neighbours[neighbours != sel_node]
            if len(neighbours) == 0:
                return None
            neighbour_distances = (
                o["distances"][i][sel_node] + o["A"][i][sel_node][neighbours]
            )
            usable = (neighbour_distances < o["distances"][i][neighbours]) + (
                o["mask"][i][neighbours] == 0
            )
            if sum(usable) == 0:
                return None
            node_probs = np.zeros((len(o["mask"][i]),), dtype=np.float32)
            node_probs[neighbours[usable]] = 1.0 / sum(usable)
            return node_probs

        def phase_1_policy(o, i):
            possible_nodes = []
            start_node = (
                o["last_selected_0"][i]
                if o["last_selected_0"][i].sum() > 0
                else o["s"][i]
            )
            try_nodes = generate_start_indices(o["mask"][i], start_node)
            for sel_node_arr in try_nodes:
                sel_node = np.argmax(sel_node_arr)

                if o["mask"][i][sel_node] == 0:
                    continue

                if phase_2_policy(o, i, sel_node) is None:
                    continue

                possible_nodes.append(sel_node)
            if not possible_nodes:
                raise ValueError(
                    "No valid node found for phase 1. Check the input data."
                )
            possible_nodes = list(set(possible_nodes))  # remove duplicates
            node_probs = np.zeros((len(o["mask"][i]),), dtype=np.float32)
            for node in possible_nodes:
                node_probs[node] = 1.0 / len(possible_nodes)
            return node_probs

        if obs["phase"][i] == 0:  # phase 1
            return phase_1_policy(obs, i)

        elif obs["phase"][i] == 1:  # phase 2
            neighbour = phase_2_policy(obs, i, np.argmax(obs["last_selected_0"][i]))
            if neighbour is not None and neighbour.sum() > 0:
                return neighbour
            else:
                raise ValueError("No valid neighbours found for phase 2.")

        else:
            raise ValueError("Invalid phase for action selection.")

    @staticmethod
    def _expert_policy_demonstrations(obs: dict[str, np.ndarray], i: int) -> np.ndarray:
        probs = BellmanFordEnv._expert_policy_probabilities(obs, i)
        return np.array([np.random.choice(len(probs), p=probs)])

    @staticmethod
    def expert_policy(obs: dict[str, np.ndarray], *args, **kwargs) -> np.ndarray:
        """Expert policy for the TSP environment.

        Returns action demonstrations rather than probabilities.
        """

        def single_policy(o, i):
            method = kwargs.get("method", "probabilities")
            if method == "probabilities":
                return BellmanFordEnv._expert_policy_probabilities(o, i)
            elif method == "demonstrations":
                return BellmanFordEnv._expert_policy_demonstrations(o, i)

        actions = np.array(
            [single_policy(obs, i) for i in range(len(obs["last_selected_0"]))]
        )

        return actions


class MSTPrimEnv(PhasedNodeSelectEnv):
    def __init__(
        self,
        max_nodes: int,
        graph_generator: GraphGenerator,
        **kwargs,
    ):
        super().__init__(
            max_nodes=max_nodes, num_phases=2, graph_generator=graph_generator
        )

    def _init_observation_space(
        self,
    ) -> tuple[gym.spaces.Dict, dict[str, tuple[str, str, str]]]:
        obs_space = spaces.Dict(
            {  # outputs
                "predecessors_ptr": spaces.Box(
                    low=0,
                    high=self.max_nodes - 1,
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
                # state
                "key": spaces.Box(
                    low=0,
                    high=self.max_nodes * self.max_nodes,
                    shape=(self.max_nodes,),
                    dtype=np.float64,
                ),
                "mark": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
                "in_queue": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes,),
                    dtype=np.int32,
                ),
            }
        )
        spec = {
            "predecessors_ptr": ("state", "node", "pointer"),
            "key": ("state", "node", "scalar"),
            "mark": ("state", "node", "mask"),
            "in_queue": ("state", "node", "mask"),
        }
        return obs_space, spec

    def _get_observation(self) -> dict:
        return {
            "key": pad_to_max_nodes(self.key, self.max_nodes),
            "predecessors_ptr": pad_to_max_nodes(self.predecessors, self.max_nodes),
            "mark": pad_to_max_nodes(self.mark, self.max_nodes),
            "in_queue": pad_to_max_nodes(self.in_queue, self.max_nodes),
        }

    def _reset_state(self, seed=None, options=None):
        # Initialise state variables
        self.key = np.zeros(self.graph_data.num_nodes, dtype=np.float64)
        self.mark = np.zeros(self.graph_data.num_nodes, dtype=np.int32)
        self.in_queue = self.graph_data.s.numpy().astype(np.int32)
        self.predecessors = np.arange(self.graph_data.num_nodes, dtype=np.int32)
        self.key[self.graph_data.s] = 0.0

        self.weights = to_dense_adj(
            self.graph_data.edge_index, edge_attr=self.graph_data.A
        )[0]

        if hasattr(self.graph_data, "pi"):
            self.solution_predecessors = self.graph_data.pi
        else:
            self.solution_predecessors, _ = mst_prim(
                self.weights[
                    : self.graph_data.num_nodes, : self.graph_data.num_nodes
                ].numpy(),
                np.argmax(self.graph_data.s),
            )
        # For caching
        self._is_success_inputs = None

        return {}

    @staticmethod
    def get_max_episode_steps(n: int, e: int) -> int:
        return 2 * n * n  # Worst-case complexity

    def is_success(self) -> bool | None:
        state = (
            tuple(self.key.tolist()),
            tuple(self.predecessors.tolist()),
            tuple(self.graph_data.s.tolist()),
            tuple(self.mark.tolist()),
            tuple(self.in_queue.tolist()),
        )
        if self._is_success_inputs == state:
            return self._is_success_cache

        # Compute success criterion
        result = check_valid_mst_predecessors(
            self.solution_predecessors[: self.graph_data.num_nodes],
            self.predecessors[: self.graph_data.num_nodes],
            self.adj.to_dense().numpy(),
            self.weights.numpy(),
        )

        # Cache
        self._is_success_inputs = state
        self._is_success_cache = result
        return result

    def is_terminal(self) -> bool:
        return self.is_success() == True

    def _step_env(self, action: int) -> dict:
        if self.current_phase != 1:
            self.mark[action] = 1
            self.in_queue[action] = 0
            return {}

        in_tree_node = self.last_selected[0]
        out_tree_node = action

        # Set the predecessor for the selected node
        if self.adj[in_tree_node, out_tree_node] == 1 and self.mark[out_tree_node] == 0:
            weight = self.weights[in_tree_node, out_tree_node]
            if self.in_queue[out_tree_node] == 0 or weight < self.key[out_tree_node]:
                self.predecessors[out_tree_node] = in_tree_node
                self.key[out_tree_node] = weight
                self.in_queue[out_tree_node] = 1

        return {}

    def action_masks(self):
        if self.current_phase == 0:
            # Nodes in the in_queue are selectable in phase 0
            possible_nodes = self.in_queue.copy()
            if self.last_selected[0] is not None:
                possible_nodes[self.last_selected[0]] = 1
            return pad_to_max_nodes(possible_nodes, self.max_nodes)
        elif self.current_phase == 1:
            # Only neighbours of the last selected node are selectable in phase 1
            edges = self.adj[self.last_selected[0]].to_dense().numpy().copy()
            edges[self.last_selected[0]] = 0
            return pad_to_max_nodes(edges, self.max_nodes)
        raise ValueError("Invalid phase for action masks.")

    @staticmethod
    def pre_transform(data: GraphProblemData) -> GraphProblemData:
        """Add the expert policy value to the data."""
        pi, num_steps = mst_prim(
            to_dense_adj(data.edge_index, edge_attr=data.A)[0].numpy(),
            np.argmax(data.s),
        )
        obj = -num_steps
        data.expert_objective = th.tensor(obj, dtype=th.float32)
        data.pi = pi
        return data

    @staticmethod
    def _expert_policy_probabilities(obs: dict[str, np.ndarray], i: int) -> np.ndarray:
        def phase_2_policy(o, i, sel_node, max_nodes):
            neighbours = np.where(o["adj"][i][sel_node][:max_nodes] == 1)[0]
            neighbours = neighbours[neighbours != sel_node]
            if len(neighbours) == 0:
                return None
            neighbour_distances = o["A"][i][sel_node][neighbours]
            unmarked = o["mark"][i][neighbours] == 0
            not_in_queue = o["in_queue"][i][neighbours] == 0
            keys = o["key"][i][neighbours]
            usable = ((neighbour_distances < keys) + not_in_queue) * unmarked
            if sum(usable) == 0:
                return None
            node_probs = np.zeros((len(o["mark"][i]),), dtype=np.float32)
            node_probs[neighbours[usable]] = 1.0 / sum(usable)
            return node_probs

        def phase_1_policy(o, i, max_nodes):
            if o["last_selected_0"][i].sum() > 0:
                last_used_node = np.argmax(o["last_selected_0"][i])
                if phase_2_policy(o, i, last_used_node, max_nodes) is not None:
                    return o["last_selected_0"][i]

            possible_nodes = np.argsort(
                o["key"][i][:max_nodes] + (1.0 - o["in_queue"][i][:max_nodes]) * 1e9
            )

            for node in possible_nodes:
                if o["in_queue"][i][node] == 0:
                    continue
                if phase_2_policy(o, i, node, max_nodes) is None:
                    continue
                node_probs = np.zeros((len(o["mark"][i]),), dtype=np.float32)
                node_probs[node] = 1.0
                return node_probs

            raise ValueError("No valid node found for phase 1. Check the input data.")

        max_nodes = unpad_array(obs["adj"][i]).shape[0]
        if obs["phase"][i] == 0:  # phase 1
            return phase_1_policy(obs, i, max_nodes)

        elif obs["phase"][i] == 1:  # phase 2
            neighbour = phase_2_policy(
                obs, i, np.argmax(obs["last_selected_0"][i]), max_nodes
            )
            if neighbour is not None and neighbour.sum() > 0:
                return neighbour
            else:
                raise ValueError("No valid neighbours found for phase 2.")

        else:
            raise ValueError("Invalid phase for action selection.")

    @staticmethod
    def _expert_policy_demonstrations(obs: dict[str, np.ndarray], i: int) -> np.ndarray:
        probs = MSTPrimEnv._expert_policy_probabilities(obs, i)
        return np.array([np.random.choice(len(probs), p=probs)])

    @staticmethod
    def expert_policy(obs: dict[str, np.ndarray], *args, **kwargs) -> np.ndarray:
        """Expert policy for the MST environment.

        Returns action demonstrations rather than probabilities.
        """

        def single_policy(o, i):
            method = kwargs.get("method", "probabilities")
            if method == "probabilities":
                return MSTPrimEnv._expert_policy_probabilities(o, i)
            elif method == "demonstrations":
                return MSTPrimEnv._expert_policy_demonstrations(o, i)

        actions = np.array(
            [single_policy(obs, i) for i in range(len(obs["last_selected_0"]))]
        )

        return actions
