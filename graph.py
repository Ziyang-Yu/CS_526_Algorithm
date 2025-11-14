# sssp_barrier/graph.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import random
import numpy as np


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
    case_type: str = "normal",
) -> Graph:
    """
    Generate a connected-ish directed graph with non-negative real weights.
    Optimized with numpy for faster generation.
    Not guaranteed strongly connected; good enough for experiments.
    
    Args:
        n: number of vertices
        avg_out_degree: average out-degree
        weight_range: (min_weight, max_weight) tuple
        seed: random seed
        case_type: "normal", "best", or "worst"
            - "best": uniform small weights, simple structure
            - "worst": extreme weight distribution, complex structure
            - "normal": standard random weights
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)  # Keep for compatibility

    g = Graph(n)
    low, high = weight_range

    # Best case: uniform small weights, simple tree-like structure
    if case_type == "best":
        weight_range_actual = (low, low + (high - low) * 0.3)
        w_low, w_high = weight_range_actual
        
        # Spanning tree edges - vectorized
        if n > 1:
            # For each v in [1, n), pick random u in [0, v)
            v_indices = np.arange(1, n)
            u_indices = np.array([np.random.randint(0, v) for v in v_indices])
            weights = np.random.uniform(w_low, w_high, size=n-1)
            for u, v, w in zip(u_indices, v_indices, weights):
                g.add_edge(int(u), int(v), float(w))

        # Extra random edges - vectorized
        extra_edges = max(0, n * avg_out_degree - (n - 1))
        if extra_edges > 0:
            u_all = np.random.randint(0, n, size=extra_edges * 2)
            v_all = np.random.randint(0, n, size=extra_edges * 2)
            mask = u_all != v_all
            u_filtered = u_all[mask][:extra_edges]
            v_filtered = v_all[mask][:extra_edges]
            weights = np.random.uniform(w_low, w_high, size=len(u_filtered))
            for u, v, w in zip(u_filtered, v_filtered, weights):
                g.add_edge(int(u), int(v), float(w))
    
    # Worst case: extreme weight distribution, complex structure
    elif case_type == "worst":
        # Spanning tree with small weights
        if n > 1:
            v_indices = np.arange(1, n)
            u_indices = np.array([np.random.randint(0, v) for v in v_indices])
            weights = np.random.uniform(low, low + (high - low) * 0.2, size=n-1)
            for u, v, w in zip(u_indices, v_indices, weights):
                g.add_edge(int(u), int(v), float(w))

        # Extra edges with extreme weights
        extra_edges = max(0, n * avg_out_degree - (n - 1))
        if extra_edges > 0:
            u_all = np.random.randint(0, n, size=extra_edges * 2)
            v_all = np.random.randint(0, n, size=extra_edges * 2)
            mask = u_all != v_all
            u_filtered = u_all[mask][:extra_edges]
            v_filtered = v_all[mask][:extra_edges]
            
            # Alternate between small and large weights
            i_arr = np.arange(len(u_filtered))
            small_mask = (i_arr % 3) == 0
            weights = np.zeros(len(u_filtered))
            weights[small_mask] = np.random.uniform(low, low + (high - low) * 0.1, size=small_mask.sum())
            weights[~small_mask] = np.random.uniform(low + (high - low) * 0.7, high, size=(~small_mask).sum())
            
            for u, v, w in zip(u_filtered, v_filtered, weights):
                g.add_edge(int(u), int(v), float(w))
    
    # Normal case: standard random weights
    else:
        # Spanning tree edges - vectorized
        if n > 1:
            v_indices = np.arange(1, n)
            u_indices = np.array([np.random.randint(0, v) for v in v_indices])
            weights = np.random.uniform(low, high, size=n-1)
            for u, v, w in zip(u_indices, v_indices, weights):
                g.add_edge(int(u), int(v), float(w))

        # Extra random edges - vectorized
        extra_edges = max(0, n * avg_out_degree - (n - 1))
        if extra_edges > 0:
            u_all = np.random.randint(0, n, size=extra_edges * 2)
            v_all = np.random.randint(0, n, size=extra_edges * 2)
            mask = u_all != v_all
            u_filtered = u_all[mask][:extra_edges]
            v_filtered = v_all[mask][:extra_edges]
            weights = np.random.uniform(low, high, size=len(u_filtered))
            for u, v, w in zip(u_filtered, v_filtered, weights):
                g.add_edge(int(u), int(v), float(w))

    return g
