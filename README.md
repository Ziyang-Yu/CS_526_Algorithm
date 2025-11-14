# Breaking the Sorting Barrier for Directed SSSP – Experimental Code

This folder provides a practical Python implementation inspired by  
**"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths" (Duan et al., 2025)**.

Files:
- `graph.py` – graph representation & random graph generator
- `dijkstra.py` – classic Dijkstra baseline
- `bellman_ford.py` – classic Bellman-Ford baseline
- `barrier_sssp.py` – barrier-style SSSP (BMSSP + FindPivots, simplified)
- `experiments.py` – experiment harness
- `visualize.py` – visualization script for generating performance plots


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
python experiments.py
```

This will generate `sssp_experiments.csv` with results from:
- 4 algorithms: Dijkstra, Bellman-Ford, Barrier SSSP (with/without pivots)
- 5 graph sizes: n = 200, 400, 800, 2000, 5000
- 2 graph types: sparse (avg_out_degree=2), dense (avg_out_degree=8)
- 3 case types: best, normal, worst
- 10 noise repeats per configuration

## Generating visualizations

After running experiments, generate performance plots:

```bash
python visualize.py
```

This creates a `plots/` directory with:
- Performance comparisons by graph size (n)
- Case type comparisons (best/normal/worst)
- Throughput comparisons
- Memory usage comparisons
- Comprehensive summary plots

**Requirements:**
```bash
pip install matplotlib numpy
```

Note: The visualization script uses only standard library (csv module) and matplotlib, so pandas is not required.
