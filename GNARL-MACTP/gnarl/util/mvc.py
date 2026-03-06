import numpy as np
from functools import lru_cache
import pulp
import networkx as nx


@lru_cache(maxsize=32)
def min_weighted_vertex_cover_cached(
    edges: tuple[tuple[int, int], ...], node_weights: tuple[float, ...]
) -> tuple[float, list[int]]:
    """
    Finds the minimum weighted vertex cover of a graph using integer linear programming (ILP).

    Args:
        edges (tuple[tuple[int, int], ...]): A tuple of edges in the graph.
        node_weights (tuple[float]): A tuple of weights for each node in the graph.

    Returns:
        tuple: A tuple containing:
            - float: The total weight of the minimum vertex cover.
            - list: The list of nodes in the minimum vertex cover.
    """
    num_nodes = len(node_weights)
    edge_index = (np.array([e[0] for e in edges]), np.array([e[1] for e in edges]))
    node_weights_np = np.array(node_weights)

    # ILP logic (moved from min_weighted_vertex_cover_ilp)
    edges_list = list(zip(edge_index[0], edge_index[1]))
    prob = pulp.LpProblem("WeightedMinVertexCover", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(num_nodes)]
    prob += pulp.lpSum([node_weights_np[i] * x[i] for i in range(num_nodes)])
    for u, v in edges_list:
        prob += x[u] + x[v] >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    cover = []
    for i in range(num_nodes):
        val = pulp.value(x[i])
        if val is not None and not isinstance(val, pulp.LpVariable):
            if float(val) > 0.5:
                cover.append(i)
    total_weight = float(sum(node_weights_np[i] for i in cover))
    return total_weight, cover


def min_weighted_vertex_cover(
    edge_index: tuple[np.ndarray, np.ndarray], node_weights: np.ndarray
) -> tuple[float, list[int]]:
    edges = tuple((a, b) for a, b in zip(edge_index[0], edge_index[1]))
    node_weights = tuple(node_weights.tolist())
    return min_weighted_vertex_cover_cached(edges, node_weights)


@lru_cache(maxsize=32)
def mvc_approx_cached(
    edges: tuple[tuple[int, int], ...], node_weights: tuple[float, ...], epsilon: float
) -> tuple[float, list[int]]:

    G = nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from(range(len(node_weights)), weight=node_weights)
    deleted_nodes = []

    for node, weight in enumerate(node_weights):
        G.nodes[node]["w_p"] = weight
    while G.edges():
        for edge in G.edges():
            G[edge[0]][edge[1]]["delta"] = min(
                G.nodes[edge[0]]["w_p"] / G.degree(edge[0]),
                G.nodes[edge[1]]["w_p"] / G.degree(edge[1]),
            )
        unique_nodes = list(G.nodes())
        for v in unique_nodes:
            G.nodes[v]["w_p"] = G.nodes[v]["w_p"] - sum(
                [G[a][b]["delta"] for a, b in G.edges(v)]
            )
            if G.nodes[v]["w_p"] <= epsilon * node_weights[v]:
                deleted_nodes.append(v)
                G.remove_node(v)
    return sum([node_weights[v] for v in deleted_nodes]), deleted_nodes


def min_weighted_vertex_cover_approx(
    edge_index: tuple[np.ndarray, np.ndarray],
    node_weights: np.ndarray,
    epsilon: float = 0.1,
) -> tuple[float, list[int]]:
    edges = tuple((a, b) for a, b in zip(edge_index[0], edge_index[1]))
    node_weights = tuple(node_weights.tolist())
    return mvc_approx_cached(edges, node_weights, epsilon)
