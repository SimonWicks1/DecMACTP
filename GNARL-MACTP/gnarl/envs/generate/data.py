from torch_geometric.data import Dataset
import os.path as osp
from tqdm import tqdm
import torch as th
from .sampler import build_sampler

from gnarl.util.classes import dict2string


class GraphProblemDataset(Dataset):
    """A dataset for graph problems.
    Dataset will be saved in the following structure:
    root/algorithm/graph_generator_{graph_generator_kwargs}_seed=seed/num_nodes_{num_nodes}/processed/split/data_i.pt

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str): One of 'train', 'val', or 'test'.
        algorithm (str): The algorithm name, e.g., 'bfs', 'dfs', etc.
        num_nodes (int): Number of nodes in each graph.
        num_samples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        graph_generator (str): Type of graph generator to use, e.g., 'er', 'ba', etc.
        graph_generator_kwargs (dict, optional): Additional arguments for the graph generator.
        **kwargs: Additional arguments passed to the parent Dataset class.
    """

    def __init__(
        self,
        root: str,
        split: str,  # train, val, or test
        algorithm: str,
        num_nodes: int,
        num_samples: int,
        seed: int,
        graph_generator: str,
        graph_generator_kwargs: dict | None = None,
        # Edit by Xiao
        # The number of start and goal nodes for MACTP
        num_starts: int | None = None,
        num_goals: int | None = None,
        **kwargs,
    ):
        self.split = split
        self.algorithm = algorithm
        self.num_nodes = num_nodes
        self.num_samples = num_samples
        self.seed = seed
        self.graph_generator = graph_generator
        self.graph_generator_kwargs = graph_generator_kwargs or {}


        # Edit by Xiao
        # Pass num_starts and num_goals to the sampler
        self.sampler_kwargs = {}
        if num_starts is not None:
            self.sampler_kwargs["num_starts"] = num_starts
        if num_goals is not None:
            self.sampler_kwargs["num_goals"] = num_goals

        # print("At GraphProblemDataset ++++++++++++number of starts:", num_starts, "number of goals:", num_goals)
        # print("Sampler kwargs:", self.sampler_kwargs)
        name = f"{graph_generator}_{dict2string(graph_generator_kwargs)}_seed={seed}"
        root = osp.join(root, algorithm, name)

        self.sampler, self.specs = build_sampler(
            self.algorithm,
            self.seed,
            self.num_nodes,
            self.graph_generator,
            self.graph_generator_kwargs,
            **self.sampler_kwargs,  # 传递多智能体参数
        )

        super().__init__(root, **kwargs)

    @property
    def processed_file_names(self):
        return [f"data_{i}.pt" for i in range(self.num_samples)]

    @property
    def processed_dir(self):
        return osp.join(
            self.root, f"num_nodes_{self.num_nodes}", "processed", self.split
        )

    def process(self):
        """Process the raw graph data into the final format."""

        pbar = tqdm(range(self.num_samples))
        i = 0
        while i < self.num_samples:
            data = self.sampler.next()

            if self.pre_filter and not self.pre_filter(data):  # filter out
                continue

            processed = self.pre_transform(data) if self.pre_transform else data
            th.save(processed, osp.join(self.processed_dir, f"data_{i}.pt"))
            pbar.update(1)
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Get the data object at index idx."""
        d = th.load(osp.join(self.processed_dir, f"data_{idx}.pt"), weights_only=False)
        if self.transform is not None:
            return self.transform(d)
        return d
