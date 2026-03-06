#!/usr/bin/env python3

from gnarl.util.envs import make_train_env, make_eval_env
import yaml
import os
import argparse
from gnarl.util.bc import get_bc_experience, complete_config
import networkx as nx
import csv
from torch_geometric.data import Data
import torch as th
import numpy as np
import pickle
from gnarl.util.mvc import min_weighted_vertex_cover_approx

tsp_data_path = "data/tsp_large"
rgc_data_path = "data/graph-construction-datasets"
mvc_data_path = "data/mvc"


def read_graphml_with_ordered_int_labels(filepath):
    instance = nx.readwrite.read_graphml(filepath)
    num_nodes = len(instance.nodes)
    relabel_map = {str(i): i for i in range(num_nodes)}
    nx.relabel_nodes(instance, relabel_map, copy=False)

    G = nx.Graph()
    G.add_nodes_from(sorted(instance.nodes(data=True)))
    G.add_edges_from(instance.edges(data=True))

    return G


def nx_to_pyg_data_rgc(graph):
    initial_edges = nx.to_numpy_array(graph)
    edge_index = np.array(np.where(initial_edges > 0))
    initial_edge_feature = initial_edges[edge_index[0], edge_index[1]]

    data = Data(
        edge_index=th.tensor(edge_index, dtype=th.int64),
        num_nodes=initial_edges.shape[0],
        initial_edges=th.tensor(initial_edge_feature, dtype=th.int32),
        tau=th.tensor([0.05]),
    )

    return data


def read_objective_functions(method, num_nodes, i, source_path):
    filepath = os.path.join(
        source_path,
        method,
        f"{num_nodes}-{i}.txt",
    )
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        objective_value = {k: float(v) for k, v in next(reader).items()}
    return objective_value


def transform_rgc_data(config):
    destination_path = "graph_data/rgc/"
    split_max = {"train": 10000, "val": 10100, "test": 10200}
    seed = {
        "train": config["train_data"]["seed"],
        "val": config["val_data"]["seed"],
        "test": config["test_data"]["seed"],
    }

    for method, shortname in [
        ("random_network", "er_p=0.2"),
        ("barabasi_albert", "ba_M=2"),
    ]:
        for num_nodes in range(20, 101, 10):
            for i in range(max(split_max.values())):
                split = "train"
                k = i
                if i >= split_max["train"]:
                    split = "val" if i < split_max["val"] else "test"
                    k = (
                        i - split_max["train"]
                        if split == "val"
                        else i - split_max["val"]
                    )

                filepath = os.path.join(
                    rgc_data_path, method, f"{num_nodes}-{i}.graphml"
                )
                filepath = os.path.abspath(filepath)
                # check if the file exists
                if not os.path.exists(filepath):
                    continue
                graph = read_graphml_with_ordered_int_labels(filepath)
                data = nx_to_pyg_data_rgc(graph)

                objective_value = read_objective_functions(
                    method, num_nodes, i, rgc_data_path
                )
                data.init_random_removal = objective_value["random_removal"]
                data.init_targeted_removal = objective_value["targeted_removal"]

                dest_path = os.path.join(
                    destination_path,
                    f"{shortname}_tau=0.05_seed={seed[split]}",
                    f"num_nodes_{num_nodes}",
                    "processed",
                    split,
                )
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)

                th.save(data, os.path.join(dest_path, f"data_{k}.pt"))


def transform_tsp_data(seed=47):
    tsp_dest_path = f"graph_data/tsp/coordinate__seed={seed}"
    subdir = [
        d
        for d in os.listdir(tsp_data_path)
        if os.path.isdir(os.path.join(tsp_data_path, d))
    ]
    for subdir_name in subdir:
        print(f"Processing {subdir_name}...")
        n = int(subdir_name.split("_")[-1])
        path = os.path.join(tsp_data_path, subdir_name, "processed")

        for split in ["train", "val", "test"]:
            split_path = os.path.join(path, f"{split}_{n}")
            if not os.path.exists(split_path):
                split_path = os.path.join(path, split)
                if not os.path.exists(split_path):
                    print(f"Skipping non-existent split: {split_path}")
                    continue
            for f in os.listdir(split_path):
                if not f.endswith(".pt"):
                    continue
                data = th.load(os.path.join(split_path, f), weights_only=False)
                if not isinstance(data, Data):
                    print("Skipping non-Data object:", f)
                    continue
                dest = os.path.join(tsp_dest_path, subdir_name, "processed", split)
                if not os.path.exists(dest):
                    os.makedirs(dest)
                data2 = Data()
                data2.edge_index = data.edge_index
                data2.xc = data.xc
                data2.yc = data.yc
                data2.num_nodes = data.num_nodes
                data2.A = data.edge_attr
                data2.expert_objective = -data.optimal_value
                data2.s = data.start_route
                data2.inputs = ["xc", "yc", "A", "s"]
                th.save(
                    data2,
                    os.path.join(dest, f),
                )


def recover_from_bipartite(B):
    # Find the set of original nodes (bipartite=1)
    original_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]
    G = nx.Graph()
    G.add_nodes_from(original_nodes)
    # Edge nodes are those with bipartite=0
    edge_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    for e in edge_nodes:
        neighbors = list(B.neighbors(e))
        if len(neighbors) == 2:
            u, v = neighbors
            G.add_edge(u, v)
    # Recover weights
    weights = B.graph.get("weights", [])
    G.graph["weights"] = weights
    return G


def nx_to_pyg_data_mvc(graph: nx.Graph) -> Data:
    edges = nx.to_numpy_array(graph)
    edge_index = np.array(np.where(edges > 0))
    nw = graph.graph["weights"]

    data = Data(
        nw=th.tensor(nw, dtype=th.float32),
        edge_index=th.tensor(edge_index, dtype=th.int64),
        num_nodes=edges.shape[0],
    )

    return data


def transform_train_val_data(config, input_path, type_mapping, dest_mapping):
    train_seed = config["train_data"]["seed"]
    val_seed = config["val_data"]["seed"]
    graph_type = config["train_data"]["graph_generator"]
    with open(
        os.path.join(
            input_path,
            f"train_{type_mapping[graph_type]}.pkl",
        ),
        "rb",
    ) as f:
        data = pickle.load(f)
    train_data = data[:1000]
    val_data = data[1000:]
    for split, split_data, seed in [
        ("train", train_data, train_seed),
        ("val", val_data, val_seed),
    ]:
        num_nodes = list(config["train_data"]["node_samples"].keys())[0]
        dest_path = f"graph_data/mvc/{dest_mapping[graph_type]}_seed={seed}"
        dest = os.path.join(
            dest_path,
            f"num_nodes_{num_nodes}",
            "processed",
            split,
        )
        if not os.path.exists(dest):
            os.makedirs(dest)
        for i, B in enumerate(split_data):
            G = recover_from_bipartite(B)
            data = nx_to_pyg_data_mvc(G)
            obj, _ = min_weighted_vertex_cover_approx(
                (data.edge_index[0].numpy(), data.edge_index[1].numpy()), data.nw
            )
            data.expert_objective = th.tensor(-obj, dtype=th.float32)
            th.save(data, os.path.join(dest, f"data_{i}.pt"))


def transform_test_data(config, input_path, type_mapping, dest_mapping):
    test_seed = config["test_data"]["seed"]
    graph_type = config["test_data"]["graph_generator"]
    pfx = "test_" + type_mapping[graph_type] + "_"
    test_files = [f for f in os.listdir(input_path) if f.startswith(pfx)]
    for test_file in test_files:
        with open(os.path.join(input_path, test_file), "rb") as f:
            test_data = pickle.load(f)
        for split, split_data, seed in [
            ("test", test_data, test_seed),
        ]:
            num_nodes = int(test_file.split("_")[-1].split(".")[0])
            dest_path = f"graph_data/mvc/{dest_mapping[graph_type]}_seed={seed}"
            dest = os.path.join(
                dest_path,
                f"num_nodes_{num_nodes}",
                "processed",
                split,
            )
            if not os.path.exists(dest):
                os.makedirs(dest)
            for i, B in enumerate(split_data):
                G = recover_from_bipartite(B)
                data = nx_to_pyg_data_mvc(G)
                obj, _ = min_weighted_vertex_cover_approx(
                    (data.edge_index[0].numpy(), data.edge_index[1].numpy()), data.nw
                )
                data.expert_objective = th.tensor(-obj, dtype=th.float32)
                th.save(data, os.path.join(dest, f"data_{i}.pt"))


def generate_data_for_config(config_path):
    """Generate data for a single config file."""
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found")
        return False

    print(f"Processing config: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config = complete_config(config)

    algorithm = config["algorithm"]

    if algorithm == "tsp":
        # The data exists, extract it
        # Otherwise generate it as normal
        if not os.path.exists(tsp_data_path):
            print(f"Warning: No data found for TSP problem. Generating new data...")
            print(f"Generating training data for {config_path}")
            train_env = make_train_env(config, "")
            print(f"Generating validation data for {config_path}")
            make_eval_env(config, "", "val")
            print(f"Generating test data for {config_path}")
            make_eval_env(config, "", "test")
        else:
            print("Transforming existing TSP data...")
            transform_tsp_data(seed=config["train_data"]["seed"])

    elif algorithm == "rgc":
        # If the data exists, extract it
        # Otherwise generate it as normal
        if not os.path.exists(rgc_data_path):
            print(f"Warning: No data found for RGC problem. Generating new data...")
            print(f"Generating training data for {config_path}")
            train_env = make_train_env(config, "")
            print(f"Generating validation data for {config_path}")
            make_eval_env(config, "", "val")
            print(f"Generating test data for {config_path}")
            make_eval_env(config, "", "test")
        else:
            print("Transforming existing RGC data...")
            transform_rgc_data(config)

    elif algorithm == "mvc":
        if not os.path.exists(mvc_data_path):
            print(f"Warning: No data found for MVC problem. Generating new data...")
            print(f"Generating training data for {config_path}")
            train_env = make_train_env(config, "")
            print(f"Generating validation data for {config_path}")
            make_eval_env(config, "", "val")
            print(f"Generating test data for {config_path}")
            make_eval_env(config, "", "test")
        else:
            print("Transforming existing MVC data...")
            type_mapping = {"ba": "ordinary"}
            dest_mapping = {"ba": "ba_M_range=[1, 10]"}
            transform_train_val_data(config, mvc_data_path, type_mapping, dest_mapping)
            transform_test_data(config, mvc_data_path, type_mapping, dest_mapping)

    else:
        print(f"Generating validation data for {config_path}")
        make_eval_env(config, "", "val")
        print(f"Generating test data for {config_path}")
        make_eval_env(config, "", "test")

        print(f"Generating training data for {config_path}")
    train_env = make_train_env(config, "")

    if "BC" in config:
        print(f"Generating BC experience for {config_path}")
        get_bc_experience(
            config["BC"]["data_path"],
            train_env,
            config,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate training/validation/test data for a single config file"
    )
    parser.add_argument("config_path", help="Path to the config YAML file")

    args = parser.parse_args()

    generate_data_for_config(args.config_path)


if __name__ == "__main__":
    main()
