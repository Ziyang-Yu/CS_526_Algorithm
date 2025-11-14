
def bellman_ford(graph, n, source):
    dist = {u: float('inf') for u in graph}
    dist[source] = 0
    for _ in range(n-1):
        improved = False
        for u in graph:
            for v,w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    improved = True
        if not improved: break
    return dist
