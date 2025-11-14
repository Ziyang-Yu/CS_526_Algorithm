
import math
from core.bmssp import bmssp

def sssp_break_sorting(graph, n, source):
    d_hat = {u: float('inf') for u in graph}
    d_hat[source] = 0

    k = int((math.log(n)) ** (1/3))
    t = int((math.log(n)) ** (2/3))
    L = max(1, int(math.ceil(math.log(n) / max(1,t))))

    bmssp(graph, {source}, float('inf'), d_hat, L, k, t)
    return d_hat
