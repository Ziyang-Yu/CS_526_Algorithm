# sssp_barrier/dijkstra.py
from __future__ import annotations
from typing import List, Tuple
import heapq
from graph import Graph


def dijkstra_sssp(graph: Graph, src: int = 0) -> Tuple[List[float], int, int]:
    """
    Classic Dijkstra with binary heap.
    Returns:
        dist: list of distances
        relax_count: number of relax operations
        heap_ops: number of heap push/pop ops (cache proxy)
    """
    n = graph.n
    INF = float("inf")
    dist = [INF] * n
    dist[src] = 0.0

    heap: List[Tuple[float, int]] = [(0.0, src)]
    visited = [False] * n

    relax_count = 0
    heap_ops = 1  # initial push

    while heap:
        d_u, u = heapq.heappop(heap)
        heap_ops += 1
        if visited[u]:
            continue
        visited[u] = True

        if d_u > dist[u]:
            continue

        for v, w in graph.edges_out[u]:
            relax_count += 1
            nd = d_u + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
                heap_ops += 1

    return dist, relax_count, heap_ops
