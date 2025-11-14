
import random

def gen_graph(n, avg_deg=4):
    graph = {i: [] for i in range(n)}
    m = n * avg_deg
    for _ in range(m):
        u = random.randrange(n)
        v = random.randrange(n)
        if u != v:
            w = random.uniform(0.1, 10.0)
            graph[u].append((v, w))
    return graph
