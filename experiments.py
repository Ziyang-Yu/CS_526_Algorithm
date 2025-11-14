# sssp_barrier/experiments.py
from __future__ import annotations
import time
import csv
import math
import tracemalloc
from typing import Dict, Any, List, Tuple
import random

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

from graph import random_directed_graph, Graph
from dijkstra import dijkstra_sssp
from barrier_sssp import barrier_sssp
from bellman_ford import bellman_ford_sssp


def perturb_graph_weights(g: Graph, noise_std: float, seed: int | None = None) -> Graph:
    """Optimized with numpy for faster weight perturbation."""
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    
    new_g = Graph(g.n)
    for u in range(g.n):
        if len(g.edges_out[u]) > 0:
            # Vectorized noise generation
            weights = np.array([w for _, w in g.edges_out[u]], dtype=np.float64)
            noise = np.random.normal(0.0, noise_std, size=len(weights))
            new_weights = np.maximum(0.0, weights + noise)
            
            for (v, _), new_w in zip(g.edges_out[u], new_weights):
                new_g.add_edge(u, v, float(new_w))
    return new_g


def run_single(
    algo_name: str,
    graph: Graph,
    src: int = 0,
) -> Dict[str, Any]:
    """
    Run a single algorithm on a graph and return metrics.
    """
    # Start memory tracking
    tracemalloc.start()
    t0 = time.perf_counter()

    if algo_name == "dijkstra":
        dist, relax_count, heap_ops = dijkstra_sssp(graph, src=src)
    elif algo_name == "barrier_pivots":
        dist, relax_count, heap_ops = barrier_sssp(graph, src=src, enable_pivots=True)
    elif algo_name == "barrier_no_pivots":
        dist, relax_count, heap_ops = barrier_sssp(graph, src=src, enable_pivots=False)
    elif algo_name == "bellman_ford":
        dist, relax_count, heap_ops = bellman_ford_sssp(graph, src=src)
    else:
        raise ValueError(f"Unknown algo_name={algo_name}")

    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latency = t1 - t0
    m = graph.m
    throughput = m / latency if latency > 0 else float("inf")

    # Use relax_count + heap_ops as a very rough "cache access proxy"
    cache_fingerprint_proxy = relax_count + heap_ops

    return {
        "algo": algo_name,
        "n": graph.n,
        "m": graph.m,
        "latency_sec": latency,
        "peak_mem_kb": peak / 1024.0,
        "throughput_edges_per_sec": throughput,
        "relax_count": relax_count,
        "heap_ops": heap_ops,
        "cache_proxy": cache_fingerprint_proxy,
    }


def aggregate_noise_sensitivity(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Given multiple runs of the same (algo, n, m) under different noise seeds,
    compute variance, best, worst latency – a simple "noise sensitivity" view.
    """
    if not records:
        return {}

    latencies = [r["latency_sec"] for r in records]
    mean_lat = sum(latencies) / len(latencies)
    var_lat = sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)
    best_lat = min(latencies)
    worst_lat = max(latencies)

    return {
        "latency_mean": mean_lat,
        "latency_var": var_lat,
        "latency_best": best_lat,
        "latency_worst": worst_lat,
    }


def run_experiments(
    output_csv: str = "sssp_experiments.csv",
    n_values: List[int] | None = None,
    base_seed: int = 42,
    noise_std: float = 0.5,
    noise_repeats: int = 10,
):
    """
    Run experiments for:
      - Dijkstra (baseline)
      - Bellman-Ford (classic baseline)
      - Barrier SSSP with pivots (paper-style)
      - Barrier SSSP without pivots (ablation)
    Across different graph sizes, graph densities (sparse/dense), and noisy perturbations.

    Writes a CSV with columns that make it easy to see improvements.
    """
    if n_values is None:
        n_values = [200, 400, 800, 2000, 5000]

    algos = ["dijkstra", "bellman_ford", "barrier_pivots", "barrier_no_pivots"]
    
    # Two graph types: sparse (low avg_out_degree) and dense (high avg_out_degree)
    graph_types = [
        ("sparse", 2),   # sparse graph: avg_out_degree = 2
        ("dense", 8),    # dense graph: avg_out_degree = 8
    ]
    
    # Three case types: best, normal, worst
    case_types = ["best", "normal", "worst"]

    fieldnames = [
        "algo",
        "variant",
        "graph_type",
        "case_type",
        "n",
        "m",
        "base_seed",
        "noise_std",
        "noise_repeats",
        "latency_mean",
        "latency_var",
        "latency_best",
        "latency_worst",
        "mean_peak_mem_kb",
        "mean_throughput_edges_per_sec",
        "mean_cache_proxy",
    ]

    rows: List[Dict[str, Any]] = []

    # Calculate total number of configurations for progress bar
    total_configs = len(n_values) * len(graph_types) * len(case_types) * len(algos)
    
    # Create a list of all configurations to iterate with progress bar
    configs = []
    for n in n_values:
        for graph_type_name, avg_out_degree in graph_types:
            for case_type in case_types:
                for algo_name in algos:
                    configs.append((n, graph_type_name, avg_out_degree, case_type, algo_name))
    
    # Cache graphs to avoid regenerating the same graph multiple times
    graph_cache: Dict[Tuple[int, str, int, str], Graph] = {}
    
    # Use tqdm for progress bar
    for n, graph_type_name, avg_out_degree, case_type, algo_name in tqdm(
        configs, 
        desc="Running experiments",
        unit="config",
        total=total_configs
    ):
        # Generate base graph (only once per n/graph_type/case_type combination)
        graph_key = (n, graph_type_name, avg_out_degree, case_type)
        if graph_key not in graph_cache:
            graph_cache[graph_key] = random_directed_graph(
                n=n,
                avg_out_degree=avg_out_degree,
                weight_range=(1.0, 10.0),
                seed=base_seed,
                case_type=case_type,
            )
        
        base_graph = graph_cache[graph_key]
        
        all_records: List[Dict[str, Any]] = []

        for i in range(noise_repeats):
            seed = base_seed + i
            g_pert = perturb_graph_weights(base_graph, noise_std=noise_std, seed=seed)
            rec = run_single(algo_name, g_pert, src=0)
            rec["noise_run_id"] = i
            all_records.append(rec)

        # Aggregate
        agg = aggregate_noise_sensitivity(all_records)
        mean_peak_mem_kb = sum(r["peak_mem_kb"] for r in all_records) / len(all_records)
        mean_throughput = sum(r["throughput_edges_per_sec"] for r in all_records) / len(
            all_records
        )
        mean_cache_proxy = sum(r["cache_proxy"] for r in all_records) / len(all_records)

        # Determine variant
        if algo_name == "dijkstra":
            variant = "baseline"
        elif algo_name == "bellman_ford":
            variant = "baseline_bf"
        elif algo_name == "barrier_pivots":
            variant = "paper_method"
        else:  # barrier_no_pivots
            variant = "ablation_no_pivots"

        row = {
            "algo": algo_name,
            "variant": variant,
            "graph_type": graph_type_name,
            "case_type": case_type,
            "n": n,
            "m": base_graph.m,
            "base_seed": base_seed,
            "noise_std": noise_std,
            "noise_repeats": noise_repeats,
            "latency_mean": agg["latency_mean"],
            "latency_var": agg["latency_var"],
            "latency_best": agg["latency_best"],
            "latency_worst": agg["latency_worst"],
            "mean_peak_mem_kb": mean_peak_mem_kb,
            "mean_throughput_edges_per_sec": mean_throughput,
            "mean_cache_proxy": mean_cache_proxy,
        }

        rows.append(row)

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    # Example usage:
    run_experiments(output_csv="sssp_experiments.csv")
    print("Experiments done, results saved to sssp_experiments.csv")
