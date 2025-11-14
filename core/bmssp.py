
from core.pivots import find_pivots
from core.basecase import base_case
from core.pq_partial import PartialPQ

def bmssp(graph, S, B, d_hat, l, k, t):
    if l == 0:
        return base_case(graph, S, B, d_hat, k)

    P, W = find_pivots(graph, S, B, d_hat, k)
    M = max(1, int((2 ** ((l-1)*t))))
    pq = PartialPQ(M, B)

    for x in P:
        pq.insert(x, d_hat[x])
    U = set()
    prev_bound = min([d_hat[x] for x in P]) if P else B

    while pq.data and len(U) < k * M:
        S_i, B_i = pq.pull()
        Bp_i, U_i = bmssp(graph, set(S_i), B_i, d_hat, l-1, k, t)
        U |= U_i
        prev_bound = min(prev_bound, Bp_i)
    return prev_bound, U
