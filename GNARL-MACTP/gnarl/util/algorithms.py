import numpy as np
import torch as th
from functools import lru_cache
from collections import deque


def traverse_to_source_memoised(
    node: int,
    predecessors: np.ndarray,
    s: np.ndarray,
    A: th.Tensor,
    costs: list[float | None],
) -> list[float | None]:
    """Traverse from a node to the source, calculating cumulative costs along the way.
    Uses memoization to store already computed costs for efficiency.

    Args:
        node: The starting node for traversal.
        predecessors: Array where predecessors[i] is the parent of node i.
        s: Array where s[i] is 1 if node i is a source, else 0 (may be multiple sources).
        A: Adjacency matrix with edge weights.
        costs: List to store computed costs, where costs[i] is the cost from node i to the source.
    Returns:
        Updated costs list with computed costs for the traversed nodes.
    """
    if costs[node] is not None:
        return costs

    if s[node] == 1:
        costs[node] = 0.0
        return costs

    visited = {i for i, c in enumerate(costs) if c is not None}

    path = []
    path_costs = []
    current = node
    cumulative_cost = 0.0

    while s[current] == 0:
        if current in path:  # loop detected
            for n in path:
                costs[n] = len(predecessors) * 1.0
                return costs

        if current in visited:
            cumulative_cost = costs[current]
            break

        pred = predecessors[current]
        path_costs.append(A[pred, current].item())
        path.append(current)
        current = pred

    # finalise cost for all nodes in path
    for i in reversed(range(len(path))):
        n = path[i]
        cumulative_cost += path_costs[i]
        costs[n] = cumulative_cost

    return costs


@lru_cache(maxsize=200)
def _all_nodes_to_source_cached(
    predecessors: tuple, s: tuple, adj: tuple, A: tuple
) -> list[float]:
    # convert back to numpy arrays and tensors
    predecessors_arr = np.array(predecessors)
    s_arr = np.array(s)
    adj_tensor = th.tensor(adj)
    A_tensor = th.tensor(A)

    n_nodes = int((th.argwhere(adj_tensor > 0)).max().item()) + 1
    costs: list[float | None] = [None for _ in range(n_nodes)]

    for node in range(n_nodes):
        costs = traverse_to_source_memoised(
            node, predecessors_arr, s_arr, A_tensor, costs
        )

    return costs  # type: ignore


def all_nodes_to_source(
    predecessors: np.ndarray, s: np.ndarray, adj: th.Tensor, A: th.Tensor
) -> list[float]:
    """Compute the cost from each node to the nearest source node using the predecessors array.
    Uses memoization to cache results for efficiency.

    Args:
        predecessors: Array where predecessors[i] is the parent of node i.
        s: Array where s[i] is 1 if node i is a source, else 0 (may be multiple sources).
        adj: Adjacency matrix (sparse or dense) indicating graph structure.
        A: Adjacency matrix (sparse or dense) with edge weights.
    Returns:
        List of costs where costs[i] is the cost from node i to the nearest source.
    """
    # convert to tuples for caching
    pred_tuple = tuple(predecessors.tolist())
    s_tuple = tuple(s.tolist())

    # handle sparse tensors
    if th.is_tensor(adj):
        if adj.is_sparse:
            adj_tuple = tuple(tuple(row) for row in adj.to_dense().numpy())
        else:
            adj_tuple = tuple(tuple(row) for row in adj.numpy())
    else:
        adj_tuple = tuple(tuple(row) for row in adj)

    if th.is_tensor(A):
        if A.is_sparse:
            A_tuple = tuple(tuple(row) for row in A.to_dense().numpy())
        else:
            A_tuple = tuple(tuple(row) for row in A.numpy())
    else:
        A_tuple = tuple(tuple(row) for row in A)
    return _all_nodes_to_source_cached(pred_tuple, s_tuple, adj_tuple, A_tuple)


def create_depth_counter(mask, predecessors):
    """Create a depth counter for nodes in a forest defined by the predecessors array.
    Nodes not in the forest (mask=0) have depth -1, roots have depth 0, and other nodes have depth equal to their distance from the root.

    Args:
        mask: Array where mask[i] is 1 if node i is in the forest, else 0.
        predecessors: Array where predecessors[i] is the parent of node i.
    Returns:
        Array where depth_counter[i] is the depth of node i in the forest, or -1 if not in the forest.
    """
    # convert arrays to hashable tuples for caching
    mask_tuple = tuple(mask.tolist()) if hasattr(mask, "tolist") else tuple(mask)
    pred_tuple = (
        tuple(predecessors.tolist())
        if hasattr(predecessors, "tolist")
        else tuple(predecessors)
    )
    return create_depth_counter_cached(mask_tuple, pred_tuple)


@lru_cache(maxsize=200)
def create_depth_counter_cached(mask: tuple, predecessors: tuple):
    # convert back to numpy arrays
    mask_arr = np.array(mask)
    predecessors_arr = np.array(predecessors)

    depth_counter = np.ones((len(mask_arr),), dtype=int) * -1
    for i in range(len(mask_arr)):
        if mask_arr[i] == 1 and predecessors_arr[i] == i:
            depth_counter[i] = 0
    num_unassigned = np.sum(depth_counter == -1)
    while num_unassigned > 0:
        for i in range(len(mask_arr)):
            if depth_counter[i] == -1 and mask_arr[i] == 1:
                pred_depth = depth_counter[predecessors_arr[i]]
                if pred_depth == -1:  # no predecessor depth, skip
                    continue
                depth_counter[i] = pred_depth + 1
        if num_unassigned == np.sum(depth_counter == -1):
            return depth_counter  # no change, finished
        num_unassigned = np.sum(depth_counter == -1)
    return depth_counter


def pad_to_max_nodes(arr: np.ndarray, max_nodes: int) -> np.ndarray:
    """Pad a 1D array with zeros to a specified maximum length.
    If the array is already at or above the maximum length, it is returned unchanged.
    """
    if arr.shape[0] >= max_nodes:
        return arr
    return np.pad(
        arr,
        (0, max(0, max_nodes - arr.shape[0])),
        mode="constant",
        constant_values=0,
    )


def bfs(s: np.ndarray, adj: th.Tensor) -> np.ndarray:
    """Perform a BFS traversal from the source node. Returns a list of predecessors for each node."""
    # Convert arrays to hashable tuples for caching
    s_tuple = tuple(s.tolist())

    # Handle sparse tensors properly
    if th.is_tensor(adj):
        if adj.is_sparse:
            # For sparse tensors, convert to dense first
            adj_tuple = tuple(tuple(row) for row in adj.to_dense().numpy())
        else:
            adj_tuple = tuple(tuple(row) for row in adj.numpy())
    else:
        adj_tuple = tuple(tuple(row) for row in adj)
    return bfs_cached(s_tuple, adj_tuple)


@lru_cache(maxsize=200)
def bfs_cached(s: tuple, adj: tuple) -> np.ndarray:
    # Convert back to numpy arrays and tensors
    s_arr = np.array(s)
    adj_tensor = th.tensor(adj)

    reach = np.zeros_like(s_arr, dtype=np.int32)
    pi = np.arange(len(s_arr), dtype=np.int32)
    reach[np.argmax(s_arr)] = 1
    while True:
        prev_reach = np.copy(reach)
        for i in range(len(s_arr)):
            for j in range(len(s_arr)):
                if adj_tensor[i, j] > 0 and prev_reach[i] == 1:
                    if pi[j] == j and j != np.argmax(s_arr):
                        pi[j] = i
                    reach[j] = 1
        if np.all(reach == prev_reach):
            break

    return pi


def bellman_ford(s: np.ndarray, A: th.Tensor) -> tuple[np.ndarray, int]:
    """Execute the Bellman-Ford algorithm given the source node.
    Returns a list of predecessors for each node and the number of edge updates made."""
    # Convert arrays to hashable tuples for caching
    s_tuple = tuple(s.tolist())

    # Handle sparse tensors properly
    if th.is_tensor(A):
        if A.is_sparse:
            # For sparse tensors, convert to dense first
            A_tuple = tuple(tuple(row) for row in A.to_dense().numpy())
        else:
            A_tuple = tuple(tuple(row) for row in A.numpy())
    else:
        A_tuple = tuple(tuple(row) for row in A)
    return bellman_ford_cached(s_tuple, A_tuple)


@lru_cache(maxsize=200)
def bellman_ford_cached(s: tuple, A: tuple) -> tuple[np.ndarray, int]:
    # Convert back to numpy arrays and tensors
    s_arr = np.array(s)
    A_tensor = th.tensor(A)

    d = np.zeros(A_tensor.shape[0])
    pi = np.arange(A_tensor.shape[0])
    msk = np.zeros(A_tensor.shape[0])
    d[np.argmax(s_arr)] = 0
    msk[np.argmax(s_arr)] = 1
    num_steps = 0
    while True:
        prev_d = np.copy(d)
        prev_msk = np.copy(msk)
        for u in range(A_tensor.shape[0]):
            for v in range(A_tensor.shape[0]):
                if prev_msk[u] == 1 and A_tensor[u, v] != 0:
                    if msk[v] == 0 or prev_d[u] + A_tensor[u, v] < d[v]:
                        d[v] = prev_d[u] + A_tensor[u, v]
                        pi[v] = u
                        num_steps += 1
                    msk[v] = 1
        if np.all(d == prev_d):
            break

    return pi, num_steps


def mst_prim(A: np.ndarray, s: int) -> tuple[np.ndarray, int]:
    """Execute Prim's algorithm to find the Minimum Spanning Tree (MST) of a graph.
    Returns a list of predecessors for each node in the MST and the number of edge updates made.
    """
    key = np.zeros(A.shape[0])
    mark = np.zeros(A.shape[0])
    in_queue = np.zeros(A.shape[0])
    pi = np.arange(A.shape[0])
    key[s] = 0
    in_queue[s] = 1
    num_steps = 0

    for _ in range(A.shape[0]):
        u = np.argsort(key + (1.0 - in_queue) * 1e9)[0]  # drop-in for extract-min
        if in_queue[u] == 0:
            break
        mark[u] = 1
        in_queue[u] = 0
        for v in range(A.shape[0]):
            if A[u, v] != 0:
                if mark[v] == 0 and (in_queue[v] == 0 or A[u, v] < key[v]):
                    pi[v] = u
                    key[v] = A[u, v]
                    in_queue[v] = 1
                    num_steps += 1

    return pi, num_steps


def has_cycle_in_component_graph(adj_dict: dict, num_nodes: int) -> bool:
    """
    Checks for cycles in a directed graph using DFS.
    `adj_dict` is a dictionary-based adjacency list.
    """
    path = set()
    visited = set()

    def visit(u: int) -> bool:
        visited.add(u)
        path.add(u)
        for v in adj_dict.get(u, []):
            if v in path:
                return True
            if v not in visited:
                if visit(v):
                    return True
        path.remove(u)
        return False

    for i in range(num_nodes):
        if i not in visited:
            if visit(i):
                return True
    return False


def check_valid_dfs_solution(adj: th.Tensor, predecessors: np.ndarray) -> bool:
    """
    Determines if a given spanning forest could have been produced by a DFS.

    The algorithm is based on a recursive definition of a valid DFS forest. It checks two properties:
    1.  Inter-Tree Acyclicity: The relationship between trees (or subtrees at recursive steps) must be acyclic.
        If there's a graph edge from a node in tree Ta to a node in tree Tb, a valid DFS must explore Ta before Tb.
        These dependencies must not form a cycle.
    2.  Recursive Validity: Each tree in the forest must itself be a valid DFS tree with respect to the
        subgraph induced by its own nodes. This is verified by recursively applying the same logic to the
        forest formed by the subtrees of its children.

    Args:
        adj: A sparse torch.Tensor (torch.sparse_coo_tensor) representing the graph's adjacency matrix.
        predecessors: A numpy.ndarray where predecessors[i] is the parent of node i. A root's predecessor is itself.

    Returns:
        True if the forest is a valid DFS forest, False otherwise.
    """
    num_nodes = len(predecessors)
    if num_nodes == 0:
        return True

    # Coalesce to handle potential duplicate edges and ensure sorted indices
    adj = adj.coalesce()
    u_indices, v_indices = adj.indices()
    all_edges = list(zip(u_indices.tolist(), v_indices.tolist()))
    max_node = max(u_indices.max().item(), v_indices.max().item()) + 1
    predecessors = predecessors.copy()[:max_node]

    # Build an adjacency list for the forest from the predecessors array
    children_map = [[] for _ in range(num_nodes)]
    for i, p in enumerate(predecessors):
        if i != p:
            if not (0 <= p < num_nodes):
                raise ValueError(f"Invalid predecessor {p} for node {i}")
            children_map[p].append(i)

    # Memoization for the recursive calls, using tuple of mask as key
    memo = {}

    def is_valid_forest_recursive(nodes_mask_tuple: tuple) -> bool:
        """
        Recursively checks if the sub-forest on the given nodes is a valid DFS forest.
        """
        if nodes_mask_tuple in memo:
            return memo[nodes_mask_tuple]

        nodes_mask = np.array(nodes_mask_tuple, dtype=bool)
        active_nodes_indices = np.where(nodes_mask)[0]

        if active_nodes_indices.size <= 1:
            return True

        # Identify roots of the current sub-forest. A node is a sub-root if its
        # predecessor is outside the current set of active nodes.
        sub_roots = [
            node
            for node in active_nodes_indices
            if predecessors[node] == node or not nodes_mask[predecessors[node]]
        ]

        if not sub_roots and active_nodes_indices.size > 0:
            return False

        # Build and check the component graph for cycles if there's more than one tree.
        if len(sub_roots) > 1:
            node_to_sub_root = {}
            for node in active_nodes_indices:
                curr = node
                while curr not in sub_roots:
                    curr = predecessors[curr]
                node_to_sub_root[node] = curr

            root_to_idx = {root: i for i, root in enumerate(sub_roots)}
            num_trees = len(sub_roots)
            comp_adj = {i: set() for i in range(num_trees)}

            for u, v in all_edges:
                if nodes_mask[u] and nodes_mask[v]:
                    root_u, root_v = node_to_sub_root.get(u), node_to_sub_root.get(v)
                    if root_u is not None and root_v is not None and root_u != root_v:
                        comp_adj[root_to_idx[root_u]].add(root_to_idx[root_v])

            comp_adj_list_vals = {k: list(v) for k, v in comp_adj.items()}
            if has_cycle_in_component_graph(comp_adj_list_vals, num_trees):
                memo[nodes_mask_tuple] = False
                return False

        # Recursively check the validity of the forest formed by the children of each sub-root.
        for root in sub_roots:
            children_forest_mask = np.zeros(num_nodes, dtype=bool)
            q = deque([c for c in children_map[root] if nodes_mask[c]])

            if not q:
                continue

            visited_q = set(q)
            while q:
                u = q.popleft()
                children_forest_mask[u] = True
                for v in children_map[u]:
                    if nodes_mask[v] and v not in visited_q:
                        visited_q.add(v)
                        q.append(v)

            if np.any(children_forest_mask):
                if not is_valid_forest_recursive(tuple(children_forest_mask)):
                    memo[nodes_mask_tuple] = False
                    return False

        memo[nodes_mask_tuple] = True
        return True

    initial_mask = tuple(np.ones(num_nodes, dtype=bool))
    return is_valid_forest_recursive(initial_mask)


def _verify_spanning_tree_properties_and_weight(
    edge_set: set, n_nodes: int, A: np.ndarray, adj: np.ndarray
) -> float | None:
    if len(edge_set) != n_nodes - 1:
        return None

    adj_list_tree = [[] for _ in range(n_nodes)]
    total_weight = 0.0

    for u, v in edge_set:
        if not adj[u, v]:
            return None

        adj_list_tree[u].append(v)
        adj_list_tree[v].append(u)
        total_weight += A[u, v]

    visited = [False] * n_nodes
    queue = deque([0])
    visited[0] = True
    visited_count = 1

    while queue:
        u = queue.popleft()
        for v in adj_list_tree[u]:
            if not visited[v]:
                visited[v] = True
                visited_count += 1
                queue.append(v)

    if visited_count != n_nodes:
        return None

    return total_weight


def _get_edge_set_from_input(
    input_data, n_nodes: int, adj: np.ndarray, input_type: str
) -> set | None:
    """
    Helper to convert different input formats (predecessor array or mask) to a canonical edge set.
    Also performs initial validity checks specific to the input format.
    """
    edge_set = set()
    if input_type == "predecessor":
        for i in range(n_nodes):
            pred_i = input_data[i]
            if pred_i != i:
                if not (0 <= pred_i < n_nodes):
                    return None  # Invalid predecessor index
                u, v = i, pred_i
                if not adj[u, v]:
                    return None  # Edge does not exist in original graph
                edge_set.add(tuple(sorted((u, v))))
    elif input_type == "mask":
        for u in range(n_nodes):
            for v in range(u + 1, n_nodes):
                if input_data[u, v]:
                    if not adj[u, v]:
                        return None  # Edge specified in mask does not exist in original graph
                    edge_set.add((u, v))
    else:
        return None  # Invalid input type

    return edge_set


def check_valid_mst_predecessors(
    solution_predecessors: np.ndarray,
    proposed_predecessors: np.ndarray,
    adj: np.ndarray,
    A: np.ndarray,
) -> bool:
    """
    Verify if the proposed predecessors form a valid Minimum Spanning Tree (MST)
    that matches the weight of the solution MST defined by its predecessors.
    """
    n_nodes = A.shape[0]

    solution_edges_set = _get_edge_set_from_input(
        solution_predecessors, n_nodes, adj, "predecessor"
    )
    proposed_edges_set = _get_edge_set_from_input(
        proposed_predecessors, n_nodes, adj, "predecessor"
    )

    if solution_edges_set is None or proposed_edges_set is None:
        return False

    solution_mst_weight = _verify_spanning_tree_properties_and_weight(
        solution_edges_set, n_nodes, A, adj
    )
    proposed_mst_weight = _verify_spanning_tree_properties_and_weight(
        proposed_edges_set, n_nodes, A, adj
    )

    if solution_mst_weight is None or proposed_mst_weight is None:
        return False

    return np.isclose(proposed_mst_weight, solution_mst_weight)
