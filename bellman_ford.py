# sssp_barrier/bellman_ford.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from graph import Graph


def bellman_ford_sssp(graph: Graph, src: int = 0) -> Tuple[List[float], int, int]:
    """
    Classic Bellman-Ford algorithm for SSSP, optimized with numpy.
    Works with non-negative weights (though can handle negative weights too).
    Returns:
        dist: list of distances
        relax_count: number of relax operations
        heap_ops: always 0 (Bellman-Ford doesn't use a heap, but kept for consistency)
    """
    n = graph.n
    INF = float("inf")
    dist = np.full(n, INF, dtype=np.float64)
    dist[src] = 0.0

    relax_count = 0
    heap_ops = 0  # Bellman-Ford doesn't use a heap

    # Relax all edges (n-1) times
    for _ in range(n - 1):
        changed = False
        for u in range(n):
            if dist[u] == INF:
                continue
            for v, w in graph.edges_out[u]:
                relax_count += 1
                new_dist = dist[u] + w
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    changed = True
        
        # Early termination if no changes
        if not changed:
            break

    return dist.tolist(), relax_count, heap_ops

