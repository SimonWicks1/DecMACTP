import numpy as np
import hashlib

try:
    from third_party.relnet.objective_functions.objective_functions_ext import *

    HAS_CPP_EXT = True
except ImportError as e:
    HAS_CPP_EXT = False
    import warnings

    warnings.warn(
        f"C++ extension not available. Robustness objective functions will be disabled. Error: {e}",
        UserWarning,
    )


def extract_kwargs(kwargs):
    num_mc_sims = 20
    random_seed = 42
    if "num_mc_sims" in kwargs:
        num_mc_sims = kwargs["num_mc_sims"]
    if "random_seed" in kwargs:
        random_seed = kwargs["random_seed"]
    return num_mc_sims, random_seed


def get_graph_hash(num_nodes, edge_pairs, size=32):
    if size == 32:
        hash_instance = hashlib.md5()
        hash_bytes = 4  # 32 bits = 4 bytes
    elif size == 64:
        hash_instance = hashlib.sha256()
        hash_bytes = 8  # 64 bits = 8 bytes
    else:
        raise ValueError("only 32 or 64-bit hashes supported.")

    hash_instance.update(np.zeros(num_nodes).tobytes())
    hash_instance.update(edge_pairs.tobytes())

    # Take only the first hash_bytes from the digest to get the desired bit size
    digest = hash_instance.digest()[:hash_bytes]
    graph_hash = int.from_bytes(digest, byteorder="big", signed=False)

    return graph_hash


class CriticalFractionRandom(object):
    name = "random_removal"
    upper_limit = 1.0

    @staticmethod
    def compute(num_nodes, num_edges, edge_pairs, **kwargs):
        if not HAS_CPP_EXT:
            raise RuntimeError(
                "C++ extension not available. Install boost libraries and recompile."
            )

        num_mc_sims, random_seed = extract_kwargs(kwargs)
        graph_hash = get_graph_hash(num_nodes, edge_pairs)
        frac = critical_fraction_random(
            num_nodes, num_edges, edge_pairs, num_mc_sims, graph_hash, random_seed
        )
        return frac


class CriticalFractionTargeted(object):
    name = "targeted_removal"
    upper_limit = 1.0

    @staticmethod
    def compute(num_nodes, num_edges, edge_pairs, **kwargs):
        if not HAS_CPP_EXT:
            raise RuntimeError(
                "C++ extension not available. Install boost libraries and recompile."
            )

        num_mc_sims, random_seed = extract_kwargs(kwargs)
        graph_hash = get_graph_hash(num_nodes, edge_pairs)
        frac = critical_fraction_targeted(
            num_nodes, num_edges, edge_pairs, num_mc_sims, graph_hash, random_seed
        )
        return frac
