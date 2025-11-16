# Breaking the Sorting Barrier for Directed SSSP – Experimental Code

This folder provides a practical Python implementation inspired by  
**"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths" (Duan et al., 2025)**.

Files:
- `graph.py` – graph representation & random graph generator
- `dijkstra.py` – classic Dijkstra baseline
- `bellman_ford.py` – classic Bellman-Ford baseline
- `barrier_sssp.py` – barrier-style SSSP (BMSSP + FindPivots, simplified + randomized pivots)
- `experiments.py` – experiment harness
- `visualize.py` – visualization script for generating performance plots


## Barrier SSSP details & randomized pivots

The `BarrierSSSP` implementation in `barrier_sssp.py` follows the BMSSP + FindPivots framework from the paper, but uses:
- a simpler heap-based partial priority queue instead of the full Lemma 3.3 data structure, and  
- a **randomized pivot selection** mechanism to improve robustness in worst-case configurations.

Key ideas:
- **Pivots (`FindPivots`)**:  
  For a current frontier set `S`, the algorithm expands a neighborhood `W` (up to `k` steps) and builds a forest of shortest-path trees inside `W`. Roots of sufficiently large trees (size ≥ `k`) are chosen as pivots.
- **Randomized pivot sampling**:  
  Instead of always considering all vertices in `S` as pivot candidates, the implementation first **randomly samples a subset of `S`** as candidates, then only picks roots that fall inside this sampled subset.  
  This helps avoid consistently selecting “bad pivots” on adversarial or highly structured graphs, while keeping the overall asymptotic behavior similar.

Important parameters:
- `k` and `t`:  
  Derived from \(\log n\) via powers `log_k_power` and `log_t_power`, and control:
  - the local work in base cases (`k`), and  
  - the recursion depth / block size (`t`).
- `pivot_sample_ratio` (new):  
  - A float in \([0, 1]\) controlling the **fraction of `S` sampled as pivot candidates** in each `FindPivots` call.  
  - Default is `0.5`, i.e., roughly half of `S` are used as candidates.  
  - Values outside \([0, 1]\) are clipped. At run time, the effective sample size is at least 1 and at most \(|S|\), so when the ratio is large or `|S|` is small, the behavior gracefully falls back to using all of `S` (deterministic).

Empirical observations (from the provided `sssp_experiments.csv`):
- Across all tested random graphs and sizes, the `barrier_pivots` variant (with randomized pivots) achieves on average about **0.8×** the throughput of the `barrier_no_pivots` ablation. The extra structure (FindPivots + recursion) introduces non-trivial overhead on small/medium graphs.
- For the largest graphs tested (`n = 5000`), the situation reverses: `barrier_pivots` is typically **20–30% faster** in terms of mean edges-per-second throughput compared to `barrier_no_pivots`, with the biggest gains observed on dense and worst-case configurations.
- These numbers are indicative for the current implementation and random graph model; re-running `experiments.py` after tuning parameters (e.g., `pivot_sample_ratio`, `log_k_power`, `log_t_power`) may shift the exact values but should preserve the qualitative trend: **pivots help more as the problem size and difficulty grow**.

Usage:
- The convenience function:
  - `barrier_sssp(graph, src=0, enable_pivots=True)`  
    internally constructs `BarrierSSSP` with default parameters, including `pivot_sample_ratio=0.5`.
- For finer control (e.g., changing `pivot_sample_ratio`), instantiate the class directly:

```python
from barrier_sssp import BarrierSSSP

algo = BarrierSSSP(
    graph,
    src=0,
    enable_pivots=True,
    pivot_sample_ratio=0.3,  # e.g., sample 30% of S as pivot candidates
)
dist, relax_count, heap_ops = algo.run()
```


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
