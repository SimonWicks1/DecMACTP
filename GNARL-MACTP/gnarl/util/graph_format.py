import torch as th
from torch_geometric.data import Data, Batch
import numpy as np


def matrix_features_to_batch(
    node_features: th.Tensor,
    edge_features: th.Tensor,
    graph_features: th.Tensor,
    adj_matrix: th.Tensor,
) -> Batch:
    """Convert the matrix features to a PyTorch Geometric Batch object.

    Args:
        node_features (th.Tensor): b x n x f_n matrix of node features
        edge_features (th.Tensor): b x n x n x f_e matrix of edge features
        graph_features (th.Tensor): b x f_g vector of graph features
        adj_matrix (th.Tensor): b x n x n binary adjacency matrix

    Returns:
        Batch: PyTorch Geometric Batch object
    """

    data_list = []
    for b in range(node_features.size(0)):
        edge_index = th.nonzero(adj_matrix[b], as_tuple=False).t()
        edge_attr = edge_features[b][edge_index[0], edge_index[1]]
        has_edge = (adj_matrix[b].sum(dim=0) > 0) | (adj_matrix[b].sum(dim=1) > 0)
        node_features_b = node_features[b][has_edge]
        data = Data(
            x=node_features_b,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_attr=graph_features[b].unsqueeze(0),
        )
        data_list.append(data)
    return Batch.from_data_list(data_list)


def unpad_array(array: np.ndarray) -> np.ndarray:
    """Removes trailing zeros from a square 2D array.

    Args:
        array (np.ndarray): The input array to unpad.

    Returns:
        np.ndarray: The unpadded array.
    """
    if array.ndim != 2:
        raise ValueError("Input must be a 2D array")

    non_zero_rows = np.where(np.any(array != 0, axis=1))[0]
    non_zero_cols = np.where(np.any(array != 0, axis=0))[0]

    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return np.zeros((1, 1), dtype=array.dtype)

    last_row = non_zero_rows.max() + 1
    last_col = non_zero_cols.max() + 1
    max_dim = max(last_row, last_col)

    return array[:max_dim, :max_dim]
