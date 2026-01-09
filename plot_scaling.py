#!/usr/bin/env python3
"""
Plot wall time vs number of cores for calibration scaling study.
"""
import json
import os
import glob
import numpy as np

# Try to import matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for cluster
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Cannot create plot.")

def load_calibration_results(outdir="results_calibration"):
    """Load all calibration JSON files and extract wall time vs cores data."""
    pattern = os.path.join(outdir, "calib_workers*_chains*.json")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No calibration result files found in {outdir}")
    
    cores_list = []
    wall_times = []
    
    for filepath in files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            cores_list.append(data['n_workers'])
            wall_times.append(data['wall_time_sec'])
    
    # Sort by number of cores
    sorted_data = sorted(zip(cores_list, wall_times))
    cores = np.array([x[0] for x in sorted_data])
    times = np.array([x[1] for x in sorted_data])
    
    return cores, times

def plot_scaling(cores, wall_times, output_file="scaling_plot.png"):
    """Create scaling plot: wall time vs number of cores."""
    if not HAS_MATPLOTLIB:
        print("Cannot create plot: matplotlib not available")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot actual data
    plt.plot(cores, wall_times, 'o-', linewidth=2, markersize=8, label='Measured')
    
    # Plot ideal scaling (inverse relationship)
    if len(cores) > 0:
        ideal_time = wall_times[0] / cores  # Ideal: time scales inversely with cores
        plt.plot(cores, ideal_time, '--', linewidth=1.5, alpha=0.7, 
                label=f'Ideal scaling (from {cores[0]} core)')
    
    plt.xlabel('Number of Cores', fontsize=12)
    plt.ylabel('Wall Time (seconds)', fontsize=12)
    plt.title('Calibration Scaling: Wall Time vs Number of Cores', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Log scale for x-axis (optional, comment out if not desired)
    # plt.xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

def print_summary(cores, wall_times):
    """Print summary statistics."""
    print("\n" + "="*50)
    print("Scaling Study Summary")
    print("="*50)
    print(f"{'Cores':<10} {'Wall Time (s)':<15} {'Speedup':<15} {'Efficiency':<15}")
    print("-"*50)
    
    if len(wall_times) > 0:
        baseline_time = wall_times[0]
        baseline_cores = cores[0]
        
        for c, t in zip(cores, wall_times):
            speedup = baseline_time / t if t > 0 else 0
            efficiency = speedup / (c / baseline_cores) if c > 0 else 0
            print(f"{c:<10} {t:<15.2f} {speedup:<15.2f} {efficiency:<15.2%}")
    
    print("="*50)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Plot calibration scaling results")
    ap.add_argument("--outdir", type=str, default="results_calibration",
                    help="Directory containing calibration JSON files")
    ap.add_argument("--output", type=str, default="scaling_plot.png",
                    help="Output plot filename")
    args = ap.parse_args()
    
    try:
        cores, wall_times = load_calibration_results(args.outdir)
        print_summary(cores, wall_times)
        
        if HAS_MATPLOTLIB:
            plot_scaling(cores, wall_times, args.output)
        else:
            print("\nData loaded successfully but matplotlib not available for plotting.")
            print("You can plot manually using the data above.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure calibration jobs have completed and JSON files exist.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
