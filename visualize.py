"""
Visualization script for SSSP experiments.
Generates performance comparison plots similar to academic paper style.
"""
from __future__ import annotations
import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import os
from collections import defaultdict

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Algorithm colors and markers
ALGO_STYLE = {
    'dijkstra': {'color': '#000000', 'marker': 'o', 'linestyle': '-', 'label': 'Dijkstra'},
    'bellman_ford': {'color': '#FF6B6B', 'marker': '^', 'linestyle': '--', 'label': 'Bellman-Ford'},
    'barrier_pivots': {'color': '#4ECDC4', 'marker': 's', 'linestyle': '-', 'label': 'Barrier (Pivots)'},
    'barrier_no_pivots': {'color': '#FFE66D', 'marker': '*', 'linestyle': '-.', 'label': 'Barrier (No Pivots)'},
}


class DataFrame:
    """Simple DataFrame-like class for data manipulation."""
    def __init__(self, data: List[Dict]):
        self.data = data
        self.columns = list(data[0].keys()) if data else []
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.data]
        # Boolean indexing
        if isinstance(key, list) and len(key) == len(self.data):
            return DataFrame([row for row, mask in zip(self.data, key) if mask])
        return None
    
    def __len__(self):
        return len(self.data)
    
    def sort_values(self, column: str):
        sorted_data = sorted(self.data, key=lambda x: float(x[column]))
        return DataFrame(sorted_data)


def load_data(csv_path: str = "sssp_experiments.csv") -> DataFrame:
    """Load experiment data from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ['n', 'm', 'base_seed', 'noise_repeats', 
                       'latency_mean', 'latency_var', 'latency_best', 'latency_worst',
                       'mean_peak_mem_kb', 'mean_throughput_edges_per_sec', 'mean_cache_proxy']:
                if key in row:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            
            # Convert latency from seconds to milliseconds
            row['latency_mean_ms'] = row.get('latency_mean', 0) * 1000
            row['latency_best_ms'] = row.get('latency_best', 0) * 1000
            row['latency_worst_ms'] = row.get('latency_worst', 0) * 1000
            data.append(row)
    
    return DataFrame(data)


def plot_performance_by_n(df: DataFrame, graph_type: str, case_type: str, save_path: Optional[str] = None):
    """
    Plot latency vs n for different algorithms.
    Similar to the style in the provided images.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'SSSP Performance: {graph_type.capitalize()} Graphs, {case_type.capitalize()} Case', 
                 fontsize=14, fontweight='bold')
    
    # Filter data
    filtered_data = [row for row in df.data 
                    if row['graph_type'] == graph_type and row['case_type'] == case_type]
    data = DataFrame(filtered_data)
    
    for idx, metric in enumerate(['latency_mean_ms', 'latency_best_ms', 'latency_worst_ms']):
        ax = axes[idx]
        
        for algo in ['dijkstra', 'bellman_ford', 'barrier_pivots', 'barrier_no_pivots']:
            algo_data_list = [row for row in data.data if row['algo'] == algo]
            algo_data = DataFrame(algo_data_list)
            if len(algo_data) > 0:
                algo_data = algo_data.sort_values('n')
                style = ALGO_STYLE[algo]
                n_vals = algo_data['n']
                metric_vals = algo_data[metric]
                ax.plot(n_vals, metric_vals, 
                       color=style['color'], marker=style['marker'], 
                       linestyle=style['linestyle'], label=style['label'],
                       linewidth=2, markersize=6)
        
        ax.set_xlabel('Number of Vertices (n)', fontweight='bold')
        ax.set_ylabel('Time (msec)', fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        metric_name = metric.replace('_ms', '').replace('_', ' ').title()
        ax.set_title(f'({chr(97+idx)}) {metric_name}', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_comparison_by_case_type(df: DataFrame, graph_type: str, save_path: Optional[str] = None):
    """
    Compare algorithms across different case types (best/normal/worst).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'SSSP Performance: {graph_type.capitalize()} Graphs - Case Type Comparison', 
                 fontsize=14, fontweight='bold')
    
    filtered_data = [row for row in df.data if row['graph_type'] == graph_type]
    data = DataFrame(filtered_data)
    case_types = ['best', 'normal', 'worst']
    
    for idx, case_type in enumerate(case_types):
        ax = axes[idx]
        case_data_list = [row for row in data.data if row['case_type'] == case_type]
        case_data = DataFrame(case_data_list)
        case_data = case_data.sort_values('n')
        
        for algo in ['dijkstra', 'bellman_ford', 'barrier_pivots', 'barrier_no_pivots']:
            algo_data_list = [row for row in case_data.data if row['algo'] == algo]
            algo_data = DataFrame(algo_data_list)
            if len(algo_data) > 0:
                algo_data = algo_data.sort_values('n')
                style = ALGO_STYLE[algo]
                ax.plot(algo_data['n'], algo_data['latency_mean_ms'], 
                       color=style['color'], marker=style['marker'], 
                       linestyle=style['linestyle'], label=style['label'],
                       linewidth=2, markersize=6)
        
        ax.set_xlabel('Number of Vertices (n)', fontweight='bold')
        ax.set_ylabel('Time (msec)', fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='best')
        ax.set_title(f'({chr(97+idx)}) {case_type.capitalize()} Case', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_throughput_comparison(df: DataFrame, save_path: Optional[str] = None):
    """
    Plot throughput (edges/sec) comparison across algorithms.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SSSP Throughput Comparison (Edges/Second)', fontsize=14, fontweight='bold')
    
    graph_types = ['sparse', 'dense']
    case_types = ['best', 'normal', 'worst']
    
    for g_idx, graph_type in enumerate(graph_types):
        for c_idx, case_type in enumerate(case_types):
            ax = axes[g_idx, c_idx]
            filtered_data = [row for row in df.data 
                           if row['graph_type'] == graph_type and row['case_type'] == case_type]
            data = DataFrame(filtered_data)
            data = data.sort_values('n')
            
            for algo in ['dijkstra', 'bellman_ford', 'barrier_pivots', 'barrier_no_pivots']:
                algo_data_list = [row for row in data.data if row['algo'] == algo]
                algo_data = DataFrame(algo_data_list)
                if len(algo_data) > 0:
                    algo_data = algo_data.sort_values('n')
                    style = ALGO_STYLE[algo]
                    throughput_vals = [x / 1e6 for x in algo_data['mean_throughput_edges_per_sec']]
                    ax.plot(algo_data['n'], throughput_vals, 
                           color=style['color'], marker=style['marker'], 
                           linestyle=style['linestyle'], label=style['label'],
                           linewidth=2, markersize=6)
            
            ax.set_xlabel('Number of Vertices (n)', fontweight='bold')
            ax.set_ylabel('Throughput (M edges/sec)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            if g_idx == 0 and c_idx == 0:
                ax.legend(loc='best', fontsize=8)
            
            title = f'{graph_type.capitalize()}, {case_type.capitalize()}'
            ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_memory_comparison(df: DataFrame, save_path: Optional[str] = None):
    """
    Plot memory usage comparison across algorithms.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SSSP Memory Usage Comparison', fontsize=14, fontweight='bold')
    
    graph_types = ['sparse', 'dense']
    case_types = ['best', 'normal', 'worst']
    
    for g_idx, graph_type in enumerate(graph_types):
        for c_idx, case_type in enumerate(case_types):
            ax = axes[g_idx, c_idx]
            filtered_data = [row for row in df.data 
                           if row['graph_type'] == graph_type and row['case_type'] == case_type]
            data = DataFrame(filtered_data)
            data = data.sort_values('n')
            
            for algo in ['dijkstra', 'bellman_ford', 'barrier_pivots', 'barrier_no_pivots']:
                algo_data_list = [row for row in data.data if row['algo'] == algo]
                algo_data = DataFrame(algo_data_list)
                if len(algo_data) > 0:
                    algo_data = algo_data.sort_values('n')
                    style = ALGO_STYLE[algo]
                    ax.plot(algo_data['n'], algo_data['mean_peak_mem_kb'], 
                           color=style['color'], marker=style['marker'], 
                           linestyle=style['linestyle'], label=style['label'],
                           linewidth=2, markersize=6)
            
            ax.set_xlabel('Number of Vertices (n)', fontweight='bold')
            ax.set_ylabel('Peak Memory (KB)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            if g_idx == 0 and c_idx == 0:
                ax.legend(loc='best', fontsize=8)
            
            title = f'{graph_type.capitalize()}, {case_type.capitalize()}'
            ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_scalability_summary(df: DataFrame, save_path: Optional[str] = None):
    """
    Create a comprehensive summary plot showing scalability across all conditions.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('SSSP Algorithm Performance Summary', fontsize=16, fontweight='bold', y=0.995)
    
    graph_types = ['sparse', 'dense']
    case_types = ['best', 'normal', 'worst']
    
    plot_idx = 0
    for g_idx, graph_type in enumerate(graph_types):
        for c_idx, case_type in enumerate(case_types):
            ax = fig.add_subplot(gs[g_idx, c_idx])
            
            filtered_data = [row for row in df.data 
                           if row['graph_type'] == graph_type and row['case_type'] == case_type]
            data = DataFrame(filtered_data)
            data = data.sort_values('n')
            
            for algo in ['dijkstra', 'bellman_ford', 'barrier_pivots', 'barrier_no_pivots']:
                algo_data_list = [row for row in data.data if row['algo'] == algo]
                algo_data = DataFrame(algo_data_list)
                if len(algo_data) > 0:
                    algo_data = algo_data.sort_values('n')
                    style = ALGO_STYLE[algo]
                    ax.plot(algo_data['n'], algo_data['latency_mean_ms'], 
                           color=style['color'], marker=style['marker'], 
                           linestyle=style['linestyle'], label=style['label'],
                           linewidth=2, markersize=5)
            
            ax.set_xlabel('n', fontweight='bold')
            ax.set_ylabel('Time (msec)', fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            if g_idx == 0 and c_idx == 0:
                ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            
            title = f'{graph_type.capitalize()}, {case_type.capitalize()}'
            ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def generate_all_plots(csv_path: str = "sssp_experiments.csv", output_dir: str = "plots"):
    """Generate all visualization plots."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} records")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. Performance by n for each graph type and case type
    for graph_type in ['sparse', 'dense']:
        for case_type in ['best', 'normal', 'worst']:
            save_path = os.path.join(output_dir, f'performance_{graph_type}_{case_type}.png')
            plot_performance_by_n(df, graph_type, case_type, save_path)
    
    # 2. Case type comparison
    for graph_type in ['sparse', 'dense']:
        save_path = os.path.join(output_dir, f'case_comparison_{graph_type}.png')
        plot_comparison_by_case_type(df, graph_type, save_path)
    
    # 3. Throughput comparison
    save_path = os.path.join(output_dir, 'throughput_comparison.png')
    plot_throughput_comparison(df, save_path)
    
    # 4. Memory comparison
    save_path = os.path.join(output_dir, 'memory_comparison.png')
    plot_memory_comparison(df, save_path)
    
    # 5. Comprehensive summary
    save_path = os.path.join(output_dir, 'summary_all_conditions.png')
    plot_scalability_summary(df, save_path)
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "sssp_experiments.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "plots"
    
    generate_all_plots(csv_path, output_dir)

