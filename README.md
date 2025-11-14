# Breaking the Sorting Barrier for Directed SSSP – Experimental Code

This folder provides a practical Python implementation inspired by  
**"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths" (Duan et al., 2025)**.

Files:
- `graph.py` – graph representation & random graph generator
- `dijkstra.py` – classic Dijkstra baseline
- `barrier_sssp.py` – barrier-style SSSP (BMSSP + FindPivots, simplified)
- `experiments.py` – experiment harness


# CSV Column Descriptions:
- algo: Algorithm name - "dijkstra" (classic Dijkstra baseline), "barrier_pivots" (Barrier SSSP with pivots), "barrier_no_pivots" (Barrier SSSP ablation without pivots)
- variant: Algorithm variant identifier - "baseline" (baseline), "paper_method" (paper method), "ablation_no_pivots" (ablation without pivots)
- n: Number of vertices in the graph
- m: Number of edges in the graph
- base_seed: Base seed value for random number generator
- noise_std: Standard deviation for edge weight perturbation
- noise_repeats: Number of noise perturbation repeats on the same graph
- latency_mean: Mean execution time in seconds - average across multiple noise repeats
- latency_var: Variance of execution time - measures algorithm sensitivity to noise
- latency_best: Best execution time in seconds - minimum across multiple repeats
- latency_worst: Worst execution time in seconds - maximum across multiple repeats
- mean_peak_mem_kb: Mean peak memory usage in KB - average across multiple repeats
- mean_throughput_edges_per_sec: Mean throughput in edges per second - average of m / latency
- mean_cache_proxy: Mean cache access proxy metric - average of (relax_count + heap_ops), used as a rough estimate of cache accesses

## Running experiments

```bash
python -m sssp_barrier.experiments
