
import heapq

def base_case(graph, S, B, d_hat, k):
    x = list(S)[0]
    pq = [(d_hat[x], x)]
    U0 = {x}
    while pq and len(U0) < k+1:
        dist, u = heapq.heappop(pq)
        if dist != d_hat[u]:
            continue
        for v,w in graph[u]:
            if dist+w < d_hat[v] and dist+w < B:
                d_hat[v] = dist+w
                heapq.heappush(pq, (d_hat[v], v))
                U0.add(v)
    if len(U0) <= k:
        return B, U0
    Bp = max(d_hat[u] for u in U0)
    return Bp, {u for u in U0 if d_hat[u] < Bp}
