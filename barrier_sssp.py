# sssp_barrier/barrier_sssp.py
from __future__ import annotations
from typing import List, Tuple, Dict, Set
import heapq
from graph import Graph


INF = float("inf")


class BarrierSSSP:
    """
    Practical implementation inspired by:
      'Breaking the Sorting Barrier for Directed Single-Source Shortest Paths'
    It implements:
      - FindPivots (Lemma 3.2)
      - Recursive BMSSP (Algorithm 3)
    with a simpler heap-based partial priority queue (not the full Lemma 3.3 DS).
    """

    def __init__(
        self,
        graph: Graph,
        src: int = 0,
        enable_pivots: bool = True,
        log_k_power: float = 1.0 / 3.0,
        log_t_power: float = 2.0 / 3.0,
    ):
        self.g = graph
        self.src = src
        self.n = graph.n

        # Dist estimates d̂[·]
        self.dist = [INF] * self.n
        self.pred = [-1] * self.n
        self.dist[src] = 0.0

        # Metrics
        self.relax_count = 0
        self.heap_ops = 0

        # Params
        import math

        # Use natural log; asymptotics only depend on powers anyway.
        logn = max(1.0, math.log(self.n + 1))
        self.k = max(1, int(logn ** log_k_power))
        self.t = max(1, int(logn ** log_t_power))

        self.enable_pivots = enable_pivots

    # --- Helpers ---------------------------------------------------------

    def _relax(self, u: int, v: int, w: float) -> bool:
        """Relax edge (u, v). Return True if dist[v] was decreased."""
        self.relax_count += 1
        cand = self.dist[u] + w
        if cand < self.dist[v]:
            self.dist[v] = cand
            self.pred[v] = u
            return True
        return False

    def _is_complete(self, v: int) -> bool:
        """
        In the paper, 'complete' means dist_hat[v] == true distance.
        Here we don't know true distances, but we treat 'complete' as:
        - a node that has already been processed in the current BMSSP level.
        We will track this via explicit sets, not a separate boolean.
        """
        # We won't use this directly; instead pass explicit sets of complete nodes.
        raise NotImplementedError

    # --- Base-case: simplified Algorithm 2 -------------------------------

    def _base_case(self, B: float, S: List[int]) -> Tuple[float, List[int]]:
        """
        Base case l = 0: S is a singleton {x}, which is considered complete.
        Run a Dijkstra-like expansion from x but stop after k+1 visited nodes
        with distance < B, as in Algorithm 2 of the paper.
        """
        assert len(S) == 1
        x = S[0]

        # Local heap of (dist, vertex)
        heap: List[Tuple[float, int]] = []
        visited: Set[int] = set()
        U0: List[int] = []

        # We may revisit x if dist changed; for simplicity, push it now.
        heapq.heappush(heap, (self.dist[x], x))
        self.heap_ops += 1

        while heap and len(U0) < self.k + 1:
            d_u, u = heapq.heappop(heap)
            self.heap_ops += 1

            if u in visited:
                continue
            visited.add(u)
            U0.append(u)

            # Relax outgoing edges from u but keep under B
            for v, w in self.g.edges_out[u]:
                if self.dist[u] + w < B:
                    if self._relax(u, v, w):
                        heapq.heappush(heap, (self.dist[v], v))
                        self.heap_ops += 1

        if len(U0) <= self.k:
            # Successful execution: all nodes with d(v) < B reachable via x are done
            return B, U0
        else:
            # Partial: set B' to max distance among U0 and keep only those < B'
            B_prime = max(self.dist[v] for v in U0)
            U = [v for v in U0 if self.dist[v] < B_prime]
            return B_prime, U

    # --- FindPivots (Lemma 3.2) -----------------------------------------

    def _find_pivots(self, B: float, S: List[int]) -> Tuple[List[int], List[int]]:
        """
        FindPivots(B, S) from Lemma 3.2.
        We maintain W, expanding up to k steps. If |W| > k|S|, we return P = S.
        Otherwise we build the forest F and choose pivots as roots with >= k nodes.
        """
        # We treat all nodes in S as "complete" at this moment.
        W: Set[int] = set(S)
        W_prev: Set[int] = set(S)

        # K-step BFS-like relax
        for _ in range(self.k):
            W_curr: Set[int] = set()
            for u in W_prev:
                for v, w in self.g.edges_out[u]:
                    # Relax with strict bound B
                    if self.dist[u] + w <= self.dist[v]:
                        changed = self._relax(u, v, w)
                        if changed and self.dist[v] < B:
                            W_curr.add(v)
            W |= W_curr
            W_prev = W_curr

            if len(W) > self.k * len(S):
                # Too big, just revert to P = S
                return list(S), list(W)

        # Build forest F over W: edges (u,v) where both in W and dist[v] = dist[u] + w
        # In theory, paths are unique; we approximate via equality with small tolerance.
        EPS = 1e-9
        children: Dict[int, List[int]] = {u: [] for u in W}
        roots: Set[int] = set(W)

        for u in W:
            du = self.dist[u]
            for v, w in self.g.edges_out[u]:
                if v in W:
                    if abs(self.dist[v] - (du + w)) < EPS:
                        children[u].append(v)
                        if v in roots:
                            roots.remove(v)

        # DFS each tree to count sizes
        def subtree_size(u: int, visited: Set[int]) -> int:
            visited.add(u)
            sz = 1
            for ch in children.get(u, []):
                if ch not in visited:
                    sz += subtree_size(ch, visited)
            return sz

        P: List[int] = []
        visited: Set[int] = set()
        for r in roots:
            if r in visited:
                continue
            sz = subtree_size(r, visited)
            if r in S and sz >= self.k:
                P.append(r)

        return P, list(W)

    # --- Simplified partial priority queue --------------------------------

    class _PartialPQ:
        """
        Simple heap-based partial priority queue:
          - insert(key, value)
          - pull(M) -> (value_threshold, keys)
        Not the advanced block structure in Lemma 3.3, but OK for experiments.
        """

        def __init__(self, parent: "BarrierSSSP"):
            self.parent = parent
            self.heap: List[Tuple[float, int]] = []

        def insert(self, v: int, val: float):
            heapq.heappush(self.heap, (val, v))
            self.parent.heap_ops += 1

        def empty(self) -> bool:
            return not self.heap

        def pull(self, M: int, B: float) -> Tuple[float, List[int]]:
            """
            Return at most M vertices with smallest values.
            Also return Bi = next value in the heap (or B if heap empty).
            """
            if not self.heap:
                return B, []

            S: List[int] = []
            vals: List[float] = []
            for _ in range(M):
                if not self.heap:
                    break
                val, v = heapq.heappop(self.heap)
                self.parent.heap_ops += 1
                S.append(v)
                vals.append(val)

            if not self.heap:
                Bi = B
            else:
                Bi = self.heap[0][0]

            return Bi, S

    # --- BMSSP main recursion --------------------------------------------

    def _bmssp(
        self,
        l: int,
        B: float,
        S: List[int],
        complete_set: Set[int],
    ) -> Tuple[float, List[int]]:
        """
        BMSSP(l, B, S) in Algorithm 3 (simplified in Python).
        complete_set: vertices already known "complete" w.r.t. current global dist.
        Returns (B', U) where U is the set of vertices newly completed here.
        """
        if not S:
            return B, []

        if l == 0:
            # Base case
            return self._base_case(B, S)

        # 1) Find pivots and W
        if self.enable_pivots:
            P, W = self._find_pivots(B, S)
        else:
            # Ablation: no pivot shrinking; treat S as pivots, W empty
            P, W = list(S), list(S)

        # 2) Partial PQ
        pq = BarrierSSSP._PartialPQ(self)
        for x in P:
            pq.insert(x, self.dist[x])

        # Track new completed verts on this call
        U_all: List[int] = []
        B0 = min(self.dist[x] for x in P) if P else B
        B_prime_global = B

        M = max(1, 2 ** max(0, l - 1) * self.t)  # very rough analog of 2^{(l-1)t}

        # For simplicity, we treat all vertices in W as potential to be added at the end
        W_set: Set[int] = set(W)

        while len(U_all) < self.k * (2 ** l) * self.t and not pq.empty():
            Bi, Si = pq.pull(M, B)
            if not Si:
                break

            # Recursively process this batch
            B_prime_i, Ui = self._bmssp(l - 1, Bi, Si, complete_set)
            B_prime_global = min(B_prime_global, B_prime_i)
            U_all.extend(Ui)

            # Mark completed
            for u in Ui:
                complete_set.add(u)

            # Relax outgoing edges of Ui and push neighbors
            for u in Ui:
                for v, w in self.g.edges_out[u]:
                    old = self.dist[v]
                    if self._relax(u, v, w) and self.dist[v] < B:
                        # if (d̂[u] + w) in [Bi, B) insert into PQ.
                        if self.dist[v] >= B_prime_i:
                            pq.insert(v, self.dist[v])
                        # else: belongs to lower segment; will be covered by recursion naturally

            if B_prime_global < B:
                # partial execution, we allow stopping early
                break

        # Finally include vertices from W that now satisfy dist < B'
        U_final = list(U_all)
        for x in W_set:
            if self.dist[x] < B_prime_global and x not in complete_set:
                complete_set.add(x)
                U_final.append(x)

        return B_prime_global, U_final

    # --- Public API -------------------------------------------------------

    def run(self) -> Tuple[List[float], int, int]:
        """
        Run the barrier-style SSSP from src.
        Returns:
            dist: distance array
            relax_count
            heap_ops
        """
        # Sanity: assume all nodes reachable; if not, INF remains.
        # initial complete set contains only src in spirit
        complete_set: Set[int] = set([self.src])

        # Top level parameters (following paper: l ≈ log n / t)
        import math

        max_l = max(1, int(math.ceil(math.log(self.n + 1) / max(1.0, self.t))))
        self._bmssp(max_l, B=INF, S=[self.src], complete_set=complete_set)
        return self.dist, self.relax_count, self.heap_ops


def barrier_sssp(
    graph: Graph,
    src: int = 0,
    enable_pivots: bool = True,
) -> Tuple[List[float], int, int]:
    """
    Convenience function.
    """
    algo = BarrierSSSP(graph, src=src, enable_pivots=enable_pivots)
    return algo.run()
