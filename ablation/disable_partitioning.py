
from baseline.bellman_ford import bellman_ford
def sssp_no_partition(graph, n, source):
    return bellman_ford(graph, n, source)
