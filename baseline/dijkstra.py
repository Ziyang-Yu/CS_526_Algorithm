
import heapq
def dijkstra(graph, source):
    dist = {u: float('inf') for u in graph}
    dist[source] = 0
    pq = [(0,source)]
    while pq:
        d,u = heapq.heappop(pq)
        if d!= dist[u]: continue
        for v,w in graph[u]:
            if d+w < dist[v]:
                dist[v] = d+w
                heapq.heappush(pq,(dist[v],v))
    return dist
