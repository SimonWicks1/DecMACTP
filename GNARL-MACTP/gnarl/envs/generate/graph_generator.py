import abc
from typing import Iterator, Optional
import numpy as np
from .data import GraphProblemDataset
from gnarl.util.graph_data import GraphProblemData


class GraphGenerator(abc.ABC):
    """Abstract base class for generating GraphProblemData."""

    def __init__(self, graph_spec: dict = None):
        """
        Initializes the GraphGenerator with an optional specification.

        Args:
            spec: Optional specification for the graph generator.
        """
        self.graph_spec = graph_spec

    @abc.abstractmethod
    def generate(self) -> Iterator[GraphProblemData]:
        """Yields an infinite sequence of GraphProblemData."""
        pass

    @abc.abstractmethod
    def seed(self, seed: int) -> None:
        """Seeds the generator for reproducibility."""
        pass


class RandomSetGraphGenerator(GraphGenerator):
    """A generator that randomly yields a graph from a set of datasets.

    Args:
        datasets (list[GraphProblemDataset]): The datasets to sample from.
        seed (int): The seed for random number generation.
    """

    def __init__(self, datasets: list[GraphProblemDataset], seed: int):
        super().__init__(datasets[0].specs)
        self._datasets = datasets
        self.datasets_cumulative_sizes = np.cumsum([len(ds) for ds in datasets])
        self._rng = np.random.default_rng(seed)

    def seed(self, seed: int) -> None:
        """Reseed the generator for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def generate(self) -> Iterator[GraphProblemData]:
        """Yields random graph instances from the datasets indefinitely."""
        while True:
            idx = int(self._rng.random() * self.datasets_cumulative_sizes[-1])
            dataset_idx = np.searchsorted(
                self.datasets_cumulative_sizes, idx, side="right"
            )
            local_idx = idx - (
                self.datasets_cumulative_sizes[dataset_idx - 1]
                if dataset_idx > 0
                else 0
            )
            yield self._datasets[dataset_idx].get(local_idx)


class FixedSetGraphGenerator(GraphGenerator):
    """A generator that yields a fixed set of graph instances.
    The graphs are yielded in a round-robin fashion, with order determined by the seed, shuffled after each loop.

    Args:
        dataset (GraphProblemDataset): The dataset containing the graph instances.
        seed (int): The seed for random number generation.
        reshuffle (bool): If True, reshuffles the order of graphs after each complete iteration
            through the dataset. Defaults to False.
        subset (Optional[list[int]]): If provided, only uses the specified indices from the dataset.
            If None, uses all indices in the dataset.
    """

    def __init__(
        self,
        dataset: GraphProblemDataset,
        seed: int,
        reshuffle: bool = False,
        subset: Optional[list[int]] = None,
    ):
        super().__init__(dataset.specs)
        self._dataset = dataset
        self._rng = np.random.default_rng(seed)
        if subset is not None:
            self._indices = np.array(subset)
        else:
            self._indices = np.arange(len(self._dataset))
        self._reshuffle = reshuffle
        if self._reshuffle:
            self._rng.shuffle(self._indices)
        self._current_index = 0

    def seed(self, seed: int) -> None:
        """Reseed the generator and reshuffle if necessary."""
        self._rng = np.random.default_rng(seed)
        self._current_index = 0
        if self._reshuffle:
            self._indices = np.arange(len(self._dataset))
            self._rng.shuffle(self._indices)

    def generate(self) -> Iterator[GraphProblemData]:
        """Yields graph instances in a round-robin fashion, reshuffling if specified."""
        while True:
            yield self._dataset.get(self._indices[self._current_index])
            self._current_index = self._current_index + 1
            if self._current_index >= len(self._indices):
                self._current_index = 0
                if self._reshuffle:
                    self._rng.shuffle(self._indices)
