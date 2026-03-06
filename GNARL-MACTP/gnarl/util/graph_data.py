import numpy as np
import torch as th
import gymnasium as gym
from torch_geometric.utils import to_dense_adj
import warnings
from torch_geometric.data import Data
from clrs._src import specs
from typing import Any


class GraphProblemData(Data):
    """A data object for a graph problem."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def GraphProblemData_to_dense(
    data: GraphProblemData,
    spec: dict,
    stage: str,
    obs_space: gym.spaces.Dict,
    max_nodes: int | None,
) -> dict[str, np.ndarray]:
    def get_dense_feature(key: str) -> th.Tensor:
        if key == "adj":
            return (
                to_dense_adj(data.edge_index, max_num_nodes=max_nodes)
                .squeeze(0)
                .to(th.int32)
            )
        if key not in data:
            raise ValueError(f"Key '{key}' not found in GraphProblemData.")

        feature = data[key].clone()
        if spec[key][1] == "edge" or ():
            return to_dense_adj(
                data.edge_index, edge_attr=feature, max_num_nodes=max_nodes
            ).squeeze(0)
        if spec[key][1] == "node":
            if spec[key][2] == "pointer":
                # turn the feature back into a node label
                feature = th.where(
                    to_dense_adj(
                        data.edge_index, edge_attr=feature, max_num_nodes=max_nodes
                    ).squeeze(0)
                )[1]

            # pad the feature to max_nodes if necessary
            if feature.size(0) < max_nodes:
                padding = th.zeros(max_nodes - feature.size(0))
                feature = th.cat([feature, padding], dim=0).to(th.int64)
        elif spec[key][1] == "graph":
            if spec[key][2] == "categorical":
                max_categories = spec[key][3]
                if feature.size(-1) < max_categories:
                    padding = th.zeros(max_categories - feature.size(-1))
                    feature = th.cat([feature, padding], dim=-1).to(th.int64)
            if feature.dim() == 0:  # scalar tensor
                feature = feature.unsqueeze(0)  # make it (1,)
        return feature

    feature_keys = [
        key for key in obs_space.spaces.keys() if key in spec and spec[key][0] == stage
    ]

    dense_features = {key: get_dense_feature(key).numpy() for key in feature_keys}
    return dense_features


def infer_type(dp_type, data):
    if dp_type in [specs.Type.MASK_ONE, specs.Type.MASK]:
        return data.astype(np.int32)
    elif dp_type in [
        specs.Type.CATEGORICAL,
        specs.Type.POINTER,
        specs.Type.PERMUTATION_POINTER,
    ]:
        return data.astype(np.int64)
    elif dp_type in [specs.Type.SCALAR]:
        return data.astype(np.float64)
    return data.astype(np.float32)


def verify_sparseness(data, edge_index, data_name):
    """Verify that the n by n data is sparse (meaning that it only contains values for edges)."""
    edge_mask = np.zeros_like(data, dtype=bool)
    edge_mask[edge_index[0], edge_index[1]] = True

    if np.any(data[~edge_mask] > 0):
        warnings.warn(
            f"The data '{data_name}' is not sparse. It contains values for non-edges. Offending values: {data[~edge_mask]}",
            UserWarning,
        )


def to_torch(value):
    if isinstance(value, np.ndarray):
        return th.from_numpy(value)
    elif isinstance(value, th.Tensor):
        return value
    return th.tensor(value)


def dense_to_GraphProblemData(
    info: dict[str, Any], spec: specs.Spec
) -> GraphProblemData:
    data_dict = {}
    attrs = {}

    # Convert adjacency matrix to edge index
    edge_index = th.tensor(info["adj"]).to_sparse().indices()
    data_dict["edge_index"] = edge_index

    # Parse remaining inputs
    for k, s in spec.items():
        v = info[k]
        if k == "adj":
            continue
        elif k == "A":
            A = v.astype(np.float64)
            # # Edit by Xiao: ensure self-loops in weights matrix
            # np.fill_diagonal(A, 1.0)  # ensure self-loops
            # This is a shitty way to ensure self-loops in weights matrix
            data_dict["A"] = A[edge_index[0], edge_index[1]]
        elif s[1] == "edge":
            verify_sparseness(v, edge_index, k)
            data_dict[k] = infer_type(s[2], v[edge_index[0], edge_index[1]])
        else:
            data_dict[k] = infer_type(s[2], v)
        if s[0] not in attrs:
            attrs[s[0]] = []
        attrs[s[0]].append(k)

    data_dict = {k: to_torch(v) for k, v in data_dict.items()}
    data = GraphProblemData(**data_dict)
    for k, v in attrs.items():
        setattr(data, k, v)
    data.num_nodes = info["adj"].shape[0]
    return data


def map_data_to_inputs(data, spec):
    input_only_spec = {k: v for k, v in spec.items() if v[0] == specs.Stage.INPUT}

    n = data["adj"].shape[0]
    next_probe = {}
    for k, v in input_only_spec.items():
        if k == "pos":
            data[k] = np.copy(np.arange(n)) * 1.0 / n
        if k not in data:
            raise ValueError(
                f"Data does not contain key {k} required by spec {v}. "
                "Please check the data and spec."
            )
        s = data[k].copy() if isinstance(data[k], np.ndarray) else data[k]
        if v[2] == specs.Type.MASK_ONE:
            if not isinstance(s, np.ndarray):
                s = np.eye(n, dtype=np.int32)[s]
        next_probe[k] = s

    if "adj" not in next_probe:
        next_probe["adj"] = data["adj"]

    return dense_to_GraphProblemData(next_probe, input_only_spec)
