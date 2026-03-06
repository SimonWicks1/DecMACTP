import abc
from typing import Any, Optional
import numpy as np
import networkx as nx
from clrs._src import specs
from .specs import SPECS
from gnarl.util.classes import get_clean_kwargs
from gnarl.util.graph_data import GraphProblemData, map_data_to_inputs


class Sampler(abc.ABC):
    """Base class for sampling graph data.

    Args:
        spec: The algorithm spec.
        seed: RNG seed.
        num_nodes: Number of nodes in the graph.
        graph_generator: Type of graph generator to use, e.g., 'er', 'ba', etc.
        graph_generator_kwargs: Additional arguments for the graph generator.
        **kwargs: Additional arguments for the sampler.
    """

    def __init__(
        self,
        spec: specs.Spec,
        seed: int,
        num_nodes: int,
        graph_generator: Optional[str] = None,
        graph_generator_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        self._rng = np.random.default_rng(seed)
        self._num_nodes = num_nodes
        self._graph_generator = graph_generator
        self._graph_generator_kwargs = graph_generator_kwargs or {}
        self._spec = spec
        self._kwargs = kwargs

    def next(self) -> GraphProblemData:
        data = self._sample_data(**self._kwargs)
        return map_data_to_inputs(data, self._spec)

    def _create_graph(
        self, n, weighted, directed, low=0.0, high=1.0, **kwargs
    ) -> dict[str, np.ndarray]:
        """
        Create a graph using the specified generator and parameters.

        Args:
            n (int): Number of nodes in the graph.
            weighted (bool): Whether to assign weights to edges.
            directed (bool): Whether the graph is directed.
            low (float): Lower bound for edge weights.
            high (float): Upper bound for edge weights.
            **kwargs: Additional parameters for the graph generator.
        Returns:
            dict[str, np.ndarray]: A dictionary containing graph features.
        """

        graph = {}
        if self._graph_generator == "er":
            graph["adj"] = self._random_er_graph(n=n, **kwargs)

        elif self._graph_generator == "erloop":
            res = self._random_erloop_graph(
                n=n, weighted=weighted, low=low, high=high, **kwargs
            )
            # 解包返回值
            if isinstance(res, tuple):
                graph["adj"] = res[0]
                graph["A"] = res[1]
            else:
                graph["adj"] = res

        elif self._graph_generator == "erloopmin":
            # [修改] 传递权重参数，并允许函数返回 (adj, A) 元组
            res = self._random_erloopmin_graph(
                n=n, weighted=weighted, low=low, high=high, **kwargs
            )
            if isinstance(res, tuple):
                graph["adj"] = res[0]
                graph["A"] = res[1]
            else:
                graph["adj"] = res

        elif self._graph_generator == "erloopzero":
            # [修改] 传递权重参数，并允许函数返回 (adj, A) 元组
            res = self._random_erloopzero_graph(
                n=n, weighted=weighted, low=low, high=high, **kwargs
            )
            if isinstance(res, tuple):
                graph["adj"] = res[0]
                graph["A"] = res[1]
            else:
                graph["adj"] = res

        elif self._graph_generator == "ws":
            assert not directed, "Directed graphs not supported."
            graph["adj"] = self._watt_strogatz_graph(n=n, **kwargs)

        elif self._graph_generator == "complete":
            graph["adj"] = self._complete_graph(n=n, **kwargs)

        elif self._graph_generator == "ba":
            assert not directed, "Directed graphs not supported."
            graph["adj"] = self._barabasi_albert_graph(n=n, **kwargs)

        elif self._graph_generator == "coordinate":
            assert not directed, "Directed graphs not supported."
            mat, coordinates = self._coordinate_graph(n=n, **kwargs)
            graph["adj"] = mat
            graph["xc"] = coordinates[:, 0]
            graph["yc"] = coordinates[:, 1]
            if weighted:
                weights = np.array(
                    [
                        [
                            np.linalg.norm(coordinates[i] - coordinates[j])
                            for j in range(mat.shape[0])
                        ]
                        for i in range(mat.shape[1])
                    ]
                )
                weights = weights / np.max(weights)  # Normalize weights to [0, 1]
                graph["A"] = mat.astype(float) * weights

        else:
            raise ValueError(f"Unknown graph generator {self._graph_generator}.")

        if weighted and "A" not in graph:
            weights = -self._rng.uniform(low=-high, high=-low, size=(n, n))
            if not directed:
                weights *= np.transpose(weights)
                weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
            weights = graph["adj"].astype(float) * weights
            graph["A"] = weights

        elif not weighted and "A" not in graph and "A" in self._spec:
            graph["A"] = graph["adj"].astype(float)

        if kwargs.get("node_weights", False):
            graph["nw"] = -self._rng.uniform(low=-high, high=-low, size=n)

        return graph

    @abc.abstractmethod
    def _sample_data(self, *args, **kwargs) -> dict[str, np.ndarray]:
        pass

    def _select_parameter(self, parameter, parameter_range=None, integer=False):
        if parameter_range is not None:
            assert len(parameter_range) == 2
            if integer:
                return self._rng.randint(*parameter_range)
            else:
                return self._rng.uniform(*parameter_range)
        if isinstance(parameter, list) or isinstance(parameter, tuple):
            return self._rng.choice(parameter)
        else:
            return parameter

    def _random_er_graph(
        self,
        n,
        p=None,
        p_range=None,
        directed=False,
        connected=True,
        *args,
        **kwargs,
    ):
        """Random Erdos-Renyi graph."""
        # To satisfy the Sparse graph spec, we need to ensure that the graph is connected: p = c * ln(n) / n
        # p = self._select_parameter(p, p_range)

        if p_range is not None:
            c = self._rng.uniform(p_range[0], p_range[1])
            p = c * np.log(n) / n
            p = min(max(p, 0.0), 1.0) # 边界保护
        elif p is None:
            p = 0.1 #
        # print("============ Generating ER graph, p =", p, "============")
        
        while True:
            g = nx.erdos_renyi_graph(n, p, directed=directed)
            if connected:
                # ensure that the graph is connected
                if not nx.is_connected(g):
                    continue
            return nx.to_numpy_array(g)
        
    def _random_erloop_graph(
        self,
        n,
        p=None,
        p_range=None,
        directed=False,
        connected=True,
        weighted=False,  # 新增: 接收是否加权的标志
        low=0.0,         # 新增: 权重下界
        high=1.0,        # 新增: 权重上界
        *args,
        **kwargs,
    ):
        """Random Erdos-Renyi graph with self-loops cost set to 1."""
        p = self._select_parameter(p, p_range)
        # print("============ Generating ER graph with self-loops, p =", p, "============")
        while True:
            # 1. 生成基础拓扑结构
            g = nx.erdos_renyi_graph(n, p, directed=directed)
            
            # 添加自环：每个节点加一条自环
            for node in range(n):
                g.add_edge(node, node)
                
            if connected:
                # ensure that the graph is connected
                if not nx.is_connected(g):
                    continue
            
            # 获取邻接矩阵 (0/1)
            adj = nx.to_numpy_array(g)

            # 2. 如果需要加权，在此处处理权重逻辑
            if weighted:
                # 生成基础随机权重，范围映射到 [low, high]
                # 逻辑与 _create_graph 中的一致
                raw_weights = -self._rng.uniform(low=-high, high=-low, size=(n, n))
                
                if not directed:
                    # 保证权重矩阵对称
                    raw_weights *= np.transpose(raw_weights)
                    raw_weights = np.sqrt(raw_weights + 1e-3)
                
                # 应用权重到存在的边上
                A = adj.astype(float) * raw_weights
                
                # [核心修改] 强制将自环 (对角线) 的权重设为 1.0
                np.fill_diagonal(A, 1.0)
                
                # 返回 (拓扑, 权重) 元组，供 _create_graph 解包使用
                return adj, A

            # 如果不加权，直接返回邻接矩阵 (默认值即为 1.0)
            return adj
        
    def _random_erloopmin_graph(
        self,
        n,
        p=None,
        p_range=None,
        directed=False,
        connected=True,
        weighted=False,  # 新增参数
        low=0.0,         # 新增参数
        high=1.0,        # 新增参数
        *args,
        **kwargs,
    ):
        """Random Erdos-Renyi graph with self-loops and specific weight rules."""
        p = self._select_parameter(p, p_range)
        
        # 1. 生成拓扑结构
        while True:
            g = nx.erdos_renyi_graph(n, p, directed=directed)
            # 添加自环
            for node in range(n):
                g.add_edge(node, node)
            
            if connected:
                if not nx.is_connected(g):
                    continue
            
            # 转换为邻接矩阵
            adj = nx.to_numpy_array(g)
            
            # 2. 如果需要加权，在此处处理逻辑
            if weighted:
                # 生成基础权重 (使用与 _create_graph 相同的分布逻辑)
                # 范围映射逻辑：-uniform(-high, -low) -> [low, high]
                raw_weights = -self._rng.uniform(low=-high, high=-low, size=(n, n))
                
                if not directed:
                    # 保证对称性
                    raw_weights *= np.transpose(raw_weights)
                    raw_weights = np.sqrt(raw_weights + 1e-3)
                
                # 应用到现有的边上
                A = adj.astype(float) * raw_weights
                
                # [核心逻辑] 要求 1: 自环权重 = 相邻边的最小权重
                for i in range(n):
                    # 获取第 i 行的权重
                    row = A[i]
                    
                    # 创建掩码：连接的边 (row > 0) 且 不是自环 (index != i)
                    # 注意：前面已经把 A 计算出来了，非连接处为 0
                    neighbor_mask = (row > 0)
                    neighbor_mask[i] = False 
                    
                    if np.any(neighbor_mask):
                        # 取邻居权重的最小值
                        min_w = np.min(row[neighbor_mask])
                        A[i, i] = min_w
                    else:
                        # 极端情况（如孤立点，但在 connected=True 时不会发生），保持原随机值
                        pass
                
                return adj, A
            # 如果不加权，只返回 adj
            return adj
    
    def _random_erloopzero_graph(
        self,
        n,
        p=None,
        p_range=None,
        directed=False,
        connected=True,
        weighted=False,  # 新增参数
        low=0.0,         # 新增参数
        high=1.0,        # 新增参数
        *args,
        **kwargs,
    ):
        """Random Erdos-Renyi graph with self-loops and specific weight rules."""
        p = self._select_parameter(p, p_range)
        
        # 1. 生成拓扑结构
        while True:
            g = nx.erdos_renyi_graph(n, p, directed=directed)
            # 添加自环
            for node in range(n):
                g.add_edge(node, node)
            
            if connected:
                if not nx.is_connected(g):
                    continue
            
            # 转换为邻接矩阵
            adj = nx.to_numpy_array(g)
            
            # 2. 如果需要加权，在此处处理逻辑
            if weighted:
                # 生成基础权重 (使用与 _create_graph 相同的分布逻辑)
                # 范围映射逻辑：-uniform(-high, -low) -> [low, high]
                raw_weights = -self._rng.uniform(low=-high, high=-low, size=(n, n))
                
                if not directed:
                    # 保证对称性
                    raw_weights *= np.transpose(raw_weights)
                    raw_weights = np.sqrt(raw_weights + 1e-3)
                
                # 应用到现有的边上
                A = adj.astype(float) * raw_weights
                
                # [核心修改] 强制将自环 (对角线) 的权重设为 0.0
                np.fill_diagonal(A, 0.0)
                
                return adj, A
            # 如果不加权，只返回 adj
            return adj

    def _watt_strogatz_graph(self, n, k, *args, p=None, p_range=None, **kwargs):
        """Watts-Strogatz graph."""
        k = self._select_parameter(k)
        p = self._select_parameter(p, p_range)
        g = nx.connected_watts_strogatz_graph(n, k, p)
        mat = nx.to_numpy_array(g)
        return mat

    def _complete_graph(self, n, *args, **kwargs):
        """Complete graph."""
        mat = np.ones((n, n))
        return mat

    def _coordinate_graph(self, n, *args, **kwargs):
        """Coordinate graph."""
        coordinates = (
            self._rng.randint(0, 1001, size=(n, 2)) / 1000.0
        )  # Scale for Concorde solver
        mat = np.ones((n, n))
        return mat, coordinates

    def _barabasi_albert_graph(self, n, M=None, M_range=None, *args, **kwargs):
        """Barabasi-Albert graph."""
        M = self._select_parameter(M, M_range, integer=True)
        g = nx.barabasi_albert_graph(n, M)
        mat = nx.to_numpy_array(g)
        return mat


class DfsSampler(Sampler):
    """DFS sampler."""

    def _sample_data(self):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=True,
            acyclic=False,
            weighted=False,
            **self._graph_generator_kwargs,
        )
        return graph_data


class BfsSampler(Sampler):
    """BFS sampler."""

    def _sample_data(self):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=False,
            **self._graph_generator_kwargs,
        )
        graph_data["s"] = self._rng.choice(graph_data["adj"].shape[0])
        return graph_data


class BellmanFordSampler(Sampler):
    """Bellman-Ford sampler."""

    def _sample_data(self, low=0.0, high=1.0):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=True,
            low=low,
            high=high,
            **self._graph_generator_kwargs,
        )
        graph_data["s"] = self._rng.choice(graph_data["adj"].shape[0])
        return graph_data


class TspSampler(Sampler):
    """TSP sampler for travelling salesperson problem."""

    def _sample_data(self):
        if self._graph_generator != "coordinate":
            raise ValueError(
                "TSP sampler requires coordinate graph generator. "
                f"Got {self._graph_generator} instead."
            )
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=True,
            **self._graph_generator_kwargs,
        )
        graph_data["s"] = self._rng.choice(graph_data["adj"].shape[0])
        return graph_data


class MVCSampler(Sampler):
    """MVC sampler for minimum vertex cover."""

    def _sample_data(self):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=False,
            node_weights=True,
            **self._graph_generator_kwargs,
        )
        return graph_data


class RGCSampler(Sampler):
    """RGC sampler for robust graph construction."""

    def _sample_data(self):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=False,
            **self._graph_generator_kwargs,
        )
        graph_data["tau"] = np.array([self._graph_generator_kwargs.get("tau", 0.05)])
        return graph_data


class CTPSampler(Sampler):

    def _generate_realisation(self):
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=True,
            **self._graph_generator_kwargs,
        )
        stochastic_edge_prop = self._graph_generator_kwargs.get("prop_stochastic", 0.25)
        edges = np.argwhere(np.triu(graph_data["adj"]) > 0)
        num_stochastic = int(stochastic_edge_prop * edges.shape[0])
        stochastic_edges_idx = self._rng.choice(
            edges.shape[0], size=num_stochastic, replace=False
        )
        stochastic_edges = edges[stochastic_edges_idx]
        # convert to symmetric square matrix
        stochastic_edge_matrix = np.zeros_like(graph_data["adj"])
        edge_prob_matrix = np.zeros_like(graph_data["adj"])
        edge_realisation_matrix = np.zeros_like(graph_data["adj"])
        for edge in stochastic_edges:
            stochastic_edge_matrix[edge[0], edge[1]] = 1
            stochastic_edge_matrix[edge[1], edge[0]] = 1
            edge_prob = self._rng.uniform(
                self._graph_generator_kwargs.get("low", 0.0),
                self._graph_generator_kwargs.get("high", 1.0),
            )
            edge_prob_matrix[edge[0], edge[1]] = edge_prob
            edge_prob_matrix[edge[1], edge[0]] = edge_prob
            edge_realisation = self._rng.binomial(1, edge_prob)
            if edge_realisation == 0:
                edge_realisation_matrix[edge[0], edge[1]] = 2  # traversable
                edge_realisation_matrix[edge[1], edge[0]] = 2  # traversable
            else:
                edge_realisation_matrix[edge[0], edge[1]] = 3  # blocked
                edge_realisation_matrix[edge[1], edge[0]] = 3  # blocked

        graph_data["stochastic_edges"] = stochastic_edge_matrix
        graph_data["edge_probs"] = edge_prob_matrix
        graph_data["edge_realisation"] = edge_realisation_matrix
        graph_data["s"] = self._rng.choice(graph_data["adj"].shape[0])
        non_start_nodes = list(
            set(range(graph_data["adj"].shape[0])) - set([graph_data["s"]])
        )
        graph_data["g"] = self._rng.choice(non_start_nodes)

        return graph_data

    def _sample_data(self, **kwargs):
        while True:
            graph_data = self._generate_realisation()

            # Ensure that there is a path from s to g in the realisation
            edge_mask = graph_data["adj"] * (
                (graph_data["edge_realisation"] == 2)
                | (graph_data["edge_realisation"] == 0)
            )
            G = nx.from_numpy_array(edge_mask)
            reachable = nx.single_source_dijkstra_path(
                G, graph_data["s"], weight="weight"
            )
            if graph_data["g"] in reachable.keys():
                return graph_data

class MACTPSampler(Sampler):
    """
    Multi-Agent Canadian Traveller Problem Sampler.
    Supports multiple starts (s) and goals (g) with relaxed connectivity verification.
    Requirement: 
    For every g in G, there exists at least one s in S such that there is a path from s to g.
    For every s in S, there exists at least one g in G such that there is a path from s to g.
    """

    def _generate_realisation(self):
        # Extract multi-agent parameters from kwargs
        num_starts = self._kwargs.get("num_starts", 1)
        num_goals = self._kwargs.get("num_goals", 1)
        
        graph_data = self._create_graph(
            self._num_nodes,
            directed=False,
            acyclic=False,
            weighted=True,
            **self._graph_generator_kwargs,
        )
        
        # Generate stochastic edges (same as CTPSampler)
        stochastic_edge_prop = self._graph_generator_kwargs.get("prop_stochastic", 0.25)
        # Edit by Xiao
        # 排除自环成为随机边
        # 原代码: edges = np.argwhere(np.triu(graph_data["adj"]) > 0)
        # 修改后: 添加 k=1 参数。这会选取主对角线之上的元素，从而物理上排除了自环 (对角线)。
        edges = np.argwhere(np.triu(graph_data["adj"], k=1) > 0)
        # End of Edit by Xiao
        num_stochastic = int(stochastic_edge_prop * edges.shape[0])
        stochastic_edges_idx = self._rng.choice(
            edges.shape[0], size=num_stochastic, replace=False
        )
        stochastic_edges = edges[stochastic_edges_idx]
        
        # Convert to symmetric square matrix
        stochastic_edge_matrix = np.zeros_like(graph_data["adj"])
        edge_prob_matrix = np.zeros_like(graph_data["adj"])
        edge_realisation_matrix = np.zeros_like(graph_data["adj"])
        
        for edge in stochastic_edges:
            stochastic_edge_matrix[edge[0], edge[1]] = 1
            stochastic_edge_matrix[edge[1], edge[0]] = 1
            edge_prob = self._rng.uniform(
                self._graph_generator_kwargs.get("low", 0.0),
                self._graph_generator_kwargs.get("high", 1.0),
            )
            edge_prob_matrix[edge[0], edge[1]] = edge_prob
            edge_prob_matrix[edge[1], edge[0]] = edge_prob
            edge_realisation = self._rng.binomial(1, edge_prob)
            if edge_realisation == 0:
                edge_realisation_matrix[edge[0], edge[1]] = 2  # traversable
                edge_realisation_matrix[edge[1], edge[0]] = 2  # traversable
            else:
                edge_realisation_matrix[edge[0], edge[1]] = 3  # blocked
                edge_realisation_matrix[edge[1], edge[0]] = 3  # blocked

        graph_data["stochastic_edges"] = stochastic_edge_matrix
        graph_data["edge_probs"] = edge_prob_matrix
        graph_data["edge_realisation"] = edge_realisation_matrix
        
        # Edit by Xiao
        # Main difference from CTPSampler: multiple starts and goals
        # Select multiple starts and goals
        all_nodes = list(range(graph_data["adj"].shape[0]))
        
        # Ensure starts and goals don't overlap and we have enough nodes
        if num_starts + num_goals > len(all_nodes):
            raise ValueError(
                f"Cannot select {num_starts} starts and {num_goals} goals from {len(all_nodes)} nodes. "
                f"Total required: {num_starts + num_goals}"
            )
        
        # Randomly select starts
        starts = self._rng.choice(all_nodes, size=num_starts, replace=False)
        remaining_nodes = list(set(all_nodes) - set(starts))
        
        # Randomly select goals from remaining nodes
        goals = self._rng.choice(remaining_nodes, size=num_goals, replace=False)
        
        graph_data["s"] = starts
        graph_data["g"] = goals

        # For non-goal nodes, delete self-loops by setting diagonal to 0
        for i in range(graph_data["adj"].shape[0]):
            if i not in goals:
                graph_data["adj"][i, i] = 0
        
        return graph_data

    def _sample_data(self, **kwargs):
        num_starts = kwargs.get("num_starts", self._kwargs.get("num_starts", 1))
        num_goals  = kwargs.get("num_goals",  self._kwargs.get("num_goals", 1))

        # print("++++++++++++number of starts:", num_starts, "number of goals:", num_goals)
        # print("++++++++++++++++++Sampling MACTP data++++++++++++++++++")
        while True:
            graph_data = self._generate_realisation()

            # Build the traversable graph (non-stochastic edges + traversable stochastic edges)
            edge_mask = graph_data["adj"] * (
                (graph_data["edge_realisation"] == 2)  # traversable stochastic edges
                | (graph_data["edge_realisation"] == 0)  # non-stochastic edges
            )
            G = nx.from_numpy_array(edge_mask)
            
            # Check connectivity: for every goal g, there exists at least one start s with a path to g
            all_reachable = True
            # unreachable_goals = []
            
            for goal in graph_data["g"]:
                goal_reachable = False
                # Check if any start can reach this goal
                for start in graph_data["s"]:
                    try:
                        # Check if there's a path from this start to the goal
                        if nx.has_path(G, start, goal):
                            goal_reachable = True
                            break
                    except nx.NodeNotFound:
                        continue
                
                if not goal_reachable:
                    all_reachable = False
                    # unreachable_goals.append(goal)
                    break  # No need to check other goals if one is unreachable
            
            for start in graph_data["s"]:
                start_reachable = False
                # Check if any goal can reach from this start
                for goal in graph_data["g"]:
                    try:
                        # Check if there's a path from this start to the goal
                        if nx.has_path(G, start, goal):
                            start_reachable = True
                            break
                    except nx.NodeNotFound:
                        continue
                
                if not start_reachable:
                    all_reachable = False
                    # unreachable_goals.append(goal)
                    break  # No need to check other goals if one is unreachable
            
            # print("Selected starts:", graph_data["s"])
            if all_reachable:
                return graph_data
        
def build_sampler(
    name: str,
    seed: int,
    num_nodes: int,
    graph_generator: str,
    graph_generator_kwargs: Optional[dict[str, Any]] = None,
    **kwargs,
) -> tuple[Sampler, specs.Spec]:
    """Builds a sampler for the specified algorithm.

    Args:
        name (str): The name of the algorithm.
        seed (int): The seed for random number generation.
        num_nodes (int): The number of nodes in the graph.
        graph_generator (str): The type of graph generator to use.
        graph_generator_kwargs (Optional[dict[str, Any]]): Additional arguments for the graph generator.
        **kwargs: Additional arguments for the sampler.
    Returns:
        tuple[Sampler, specs.Spec]: The sampler instance and the algorithm spec.
    """

    if name not in SPECS or name not in SAMPLERS:
        raise NotImplementedError(f"No implementation of algorithm {name}.")

    spec = SPECS[name]
    sampler_class = SAMPLERS[name]
    # clean_kwargs = get_clean_kwargs(
    #     sampler_class._sample_data, warn=True, kwargs=kwargs
    # )
    # Edit by Xiao
    clean_kwargs = get_clean_kwargs(sampler_class.__init__, warn=True, kwargs=kwargs)

    # print("At build_sampler ++++++++++++clean_kwargs:", clean_kwargs)
    sampler = sampler_class(
        spec,
        seed=seed,
        num_nodes=num_nodes,
        graph_generator=graph_generator,
        graph_generator_kwargs=graph_generator_kwargs,
        **clean_kwargs,
    )
    return sampler, spec


SAMPLERS = {
    "dfs": DfsSampler,
    "bfs": BfsSampler,
    "bellman_ford": BellmanFordSampler,
    "mst_prim": BellmanFordSampler,
    "tsp": TspSampler,
    "mvc": MVCSampler,
    "rgc": RGCSampler,
    "ctp": CTPSampler,
    "mactp": MACTPSampler,
}