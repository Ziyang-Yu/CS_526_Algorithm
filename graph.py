# sssp_barrier/graph.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import random


@dataclass
class Graph:
    n: int
    edges_out: List[List[Tuple[int, float]]] = field(default_factory=list)

    def __post_init__(self):
        if not self.edges_out:
            self.edges_out = [[] for _ in range(self.n)]

    def add_edge(self, u: int, v: int, w: float):
        assert 0 <= u < self.n and 0 <= v < self.n
        assert w >= 0.0
        self.edges_out[u].append((v, w))

    @property
    def m(self) -> int:
        return sum(len(adj) for adj in self.edges_out)


def random_directed_graph(
    n: int,
    avg_out_degree: int = 4,
    weight_range: Tuple[float, float] = (1.0, 10.0),
    seed: int | None = None,
) -> Graph:
    """
    Generate a connected-ish directed graph with non-negative real weights.
    Not guaranteed strongly connected; good enough for experiments.
    """
    if seed is not None:
        random.seed(seed)

    g = Graph(n)
    low, high = weight_range

    # Ensure at least a directed spanning tree from 0
    for v in range(1, n):
        u = random.randrange(0, v)
        w = random.uniform(low, high)
        g.add_edge(u, v, w)

    # Add extra random edges
    extra_edges = n * avg_out_degree - (n - 1)
    for _ in range(max(0, extra_edges)):
        u = random.randrange(0, n)
        v = random.randrange(0, n)
        if u == v:
            continue
        w = random.uniform(low, high)
        g.add_edge(u, v, w)

    return g
