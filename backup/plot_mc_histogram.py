#!/usr/bin/env python3
"""
Plot histogram of Monte Carlo sampling to visualize search space exploration
and convergence to global minimum.
Compares different burn_in values: 0, 5e4, 1e5, 2e5
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
from pathlib import Path

# Import model and data loading from calibrate_parallel
import sys
sys.path.insert(0, '.')
from Optimisation_and_High_Performance_Computing_project2025.team9_OHPC_submission.src.calibrate_parallel import (
    X0, DATA_FILE, model
)


def sa_optimize_with_trace(x0, T0, sigma, f, n_iter=250000, seed=0):
    """
    SA optimization that saves ALL iterations.
    Returns: samples, mse_values
    """
    rng = np.random.default_rng(seed)
    
    x = x0.copy()
    n_params = x.shape[0]
    
    means = np.zeros(n_params)
    cov = np.diag(np.full(n_params, sigma))
    
    # Store ALL iterations for histogram
    samples = np.zeros((n_iter, n_params), dtype=float)
    mse_values = np.zeros(n_iter, dtype=float)
    
    T = T0
    
    print(f"Running {n_iter} iterations...")
    for it in range(1, n_iter + 1):
        x_old = x.copy()
        x_prop = x_old + rng.multivariate_normal(means, cov)
        
        dE = f(x_prop) - f(x_old)
        accept_prob = np.exp(-np.clip(dE / max(T, 1e-12), -100, 100))
        
        if accept_prob >= rng.random():
            x = x_prop
        
        # Update temperature
        T = T0 * (1 - it / n_iter)
        
        # Store current state
        samples[it - 1] = x.copy()
        mse_values[it - 1] = f(x)
        
        # Progress indicator
        if it % 25000 == 0:
            print(f"  Iteration {it}/{n_iter} ({(it/n_iter*100):.1f}%)")
    
    return samples, mse_values


def plot_parameter_histograms(samples, mse_values, burn_in_values, output_file="mc_histogram.png"):
    """
    Plot histograms of parameters comparing different burn_in values.
    Supports both 20-parameter (Ts, Td) and 30-parameter (T0, Ts, Td) models.
    """
    n_params = samples.shape[1]
    
    # Determine parameter structure based on number of parameters
    if n_params == 30:
        # 30-parameter structure: [T0_1..T0_10, Ts0..Ts9, Td0..Td9]
        # Select representative: T0_1, T0_2, Ts0, Ts1, Td0, Td1 (6 parameters)
        param_indices = [0, 1, 10, 11, 20, 21]  # T0_1, T0_2, Ts0, Ts1, Td0, Td1
        param_names = ['T0_1', 'T0_2', 'Ts0', 'Ts1', 'Td0', 'Td1']
    else:
        # 20-parameter structure: [Ts0, Td0, Ts1, Td1, ...]
        # Select representative parameters: first 3 phases (Ts1, Td1, Ts2, Td2, Ts3, Td3)
        param_indices = [0, 1, 2, 3, 4, 5]
        param_names = ['Ts1', 'Td1', 'Ts2', 'Td2', 'Ts3', 'Td3']
    
    n_burnins = len(burn_in_values)
    fig, axes = plt.subplots(len(param_indices), n_burnins, figsize=(4*n_burnins, 3*len(param_indices)))
    
    if n_burnins == 1:
        axes = axes.reshape(-1, 1)
    if len(param_indices) == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['steelblue', 'coral', 'lightgreen', 'gold', 'plum']
    
    for col, burn_in in enumerate(burn_in_values):
        if burn_in == 0:
            # No burn-in: use all samples
            post_burnin_samples = samples
            label_suffix = f'All ({len(samples):,})'
        else:
            # With burn-in: use samples after burn_in
            post_burnin_samples = samples[burn_in:]
            label_suffix = f'After {burn_in} ({len(post_burnin_samples):,})'
        
        for row, (param_idx, param_name) in enumerate(zip(param_indices, param_names)):
            ax = axes[row, col]
            
            # Histogram of post-burn_in samples
            ax.hist(post_burnin_samples[:, param_idx], bins=80, alpha=0.7, 
                   color=colors[col % len(colors)], edgecolor='black', 
                   linewidth=0.3, density=True, label=label_suffix)
            
            # Mark initial value
            ax.axvline(X0[param_idx], color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label='Initial')
            
            # Mark mean of post_burnin samples
            post_mean = np.mean(post_burnin_samples[:, param_idx])
            ax.axvline(post_mean, color='darkblue', linestyle='-', 
                      linewidth=1.5, alpha=0.8, label=f'Mean: {post_mean:.3f}')
            
            ax.set_xlabel(param_name, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'burn_in={burn_in}', fontsize=11)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Monte Carlo Sampling: Parameter Distributions Comparison\n'
                 f'Total iterations: {len(samples):,} | '
                 f'Final MSE: {np.mean(mse_values[-1000:]):.2f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved histogram: {output_file}")
    plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Plot Monte Carlo sampling histograms")
    ap.add_argument("--T0", type=float, default=10.0, help="Initial temperature")
    ap.add_argument("--sigma", type=float, default=1e-5, help="Jump standard deviation")
    ap.add_argument("--n_iter", type=int, default=250000, help="Total iterations")
    ap.add_argument("--burn_in_values", type=int, nargs='+', 
                   default=[0, 50000, 100000, 200000],
                   help="Burn-in values to compare (0 means no burn-in)")
    ap.add_argument("--outdir", type=str, default="results_burnin_analysis",
                   help="Output directory")
    
    args = ap.parse_args()
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = np.loadtxt(DATA_FILE, delimiter=',', skiprows=1)
    time_points = data[:, 0]
    data_points = data[:, 1]
    
    def mse(x):
        return float(np.mean((data_points - model(time_points, x)) ** 2))
    
    print(f"\nRunning Monte Carlo sampling...")
    print(f"T0={args.T0}, sigma={args.sigma}, n_iter={args.n_iter}")
    print(f"Burn-in values to compare: {args.burn_in_values}\n")
    
    # Run sampling
    samples, mse_values = sa_optimize_with_trace(
        x0=X0,
        T0=args.T0,
        sigma=args.sigma,
        f=mse,
        n_iter=args.n_iter,
        seed=0
    )
    
    print(f"\nSampling complete!")
    print(f"Final MSE: {np.mean(mse_values[-1000:]):.2f} ± {np.std(mse_values[-1000:]):.2f}")
    print(f"Minimum MSE: {np.min(mse_values):.2f}")
    
    # Validate burn_in values
    valid_burn_ins = [bi for bi in args.burn_in_values if bi < args.n_iter]
    if len(valid_burn_ins) < len(args.burn_in_values):
        print(f"Warning: Some burn_in values >= n_iter were removed")
    
    # Plot histograms
    print("\nGenerating histograms...")
    # Generate output filename based on n_iter
    if args.n_iter == 250000:
        output_filename = "mc_histogram_iter2.5e5.png"
    elif args.n_iter == 500000:
        output_filename = "mc_histogram_iter5e5.png"
    elif args.n_iter == 1000000:
        output_filename = "mc_histogram_iter1e6.png"
    elif args.n_iter == 10000000:
        output_filename = "mc_histogram_iter1e7.png"
    elif args.n_iter == 100000000:
        output_filename = "mc_histogram_iter1e8.png"
    else:
        # Convert to scientific notation for large numbers
        if args.n_iter >= 1000000:
            exp = int(np.log10(args.n_iter))
            coeff = args.n_iter / (10 ** exp)
            output_filename = f"mc_histogram_iter{coeff:.0f}e{exp}.png"
        else:
            output_filename = f"mc_histogram_iter{args.n_iter}.png"
    
    plot_parameter_histograms(samples, mse_values, valid_burn_ins,
                             output_file=str(outdir / output_filename))
    
    print(f"\nAnalysis complete! Histogram saved to {outdir}/")
    print(f"  - {output_filename}: Parameter distributions comparison")


if __name__ == "__main__":
    main()
