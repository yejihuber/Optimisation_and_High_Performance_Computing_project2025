#!/usr/bin/env python3
import argparse
import json
import os
import time
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt

# -------------------------
# Project-specific settings
# -------------------------

# 8 values for T0 (exactly as requested: 8x8=64)
T0_GRID = np.array([0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 8.0, 10.0], dtype=float)

# 8 values for sigma (log spaced usually makes most sense)
SIGMA_GRID = np.logspace(-11, -4, 8, dtype=float)

# Initial parameter array structure: [T0_1, T0_2, ..., T0_10, Ts1, Ts2, ..., Ts10, Td1, Td2, ..., Td10]
# Total: 10 T0 + 10 Ts + 10 Td = 30 parameters

T0_INITIAL = np.array([
    1878.9166666667,  # T0_1 
    1890.1666666667,  # T0_2 
    1902.0000000000,  # T0_3 
    1913.5833333333,  # T0_4 
    1923.5833333333,  # T0_5 
    1933.6666666667,  # T0_6 
    1944.0833333333,  # T0_7 
    1954.2500000000,  # T0_8 
    1964.7500000000,  # T0_9 
    1976.1666666667  # T0_10 
], dtype=float)

# Initial values: T0 (10), Ts (10), Td (10) = 30 parameters
X0 = np.concatenate([
    T0_INITIAL,            # T0 (10 values)
    np.array([0.3] * 10),  # Ts (10 values)
    np.array([5.0] * 10)   # Td (10 values)
], dtype=float)

DATA_FILE = "data_Team9.csv"


# -------------------------
# Model + loss (same names as your notebook: model, mse)
# -------------------------

def model(t, x):
    """Model function for 30-parameter optimization.
    Parameter structure: [T0_1, ..., T0_10, Ts0, ..., Ts9, Td0, ..., Td9] (30 params)
    All T0 values (T0_1 to T0_10) will be optimized (no fixed T0 value)
    """
    n_cycles = 10
    T0_optimized = x[:n_cycles]  # T0_1 to T0_10 
    Ts = x[n_cycles:2*n_cycles]  # Ts_1 to Ts_10
    Td = x[2*n_cycles:]  # Td_1 to Td_10

    # T0ARRAY is directly set to T0 values from parameter vector x 
    T0ARRAY = T0_optimized

    # Create intervals: (T0_1, T0_2), (T0_2, T0_3), ..., (T0_9, T0_10), (T0_10, inf)
    # Note: We need 10 intervals for 10 T0 values
    intervals = [(T0ARRAY[ix], T0ARRAY[ix + 1]) for ix in range(n_cycles - 1)] + [(T0ARRAY[n_cycles - 1], np.inf)]

    t = np.atleast_1d(t)
    out = np.zeros_like(t, dtype=float)

    for ix, (a, b) in enumerate(intervals):
        mask = (a <= t) & (t < b)
        out[mask] = ((t[mask] - T0ARRAY[ix]) / Ts[ix]) ** 2 * np.exp(-((t[mask] - T0ARRAY[ix]) / Td[ix]) ** 2)

    return out.item() if out.size == 1 else out


def sa_tune(x0, T0, sigma, f, n_iter=2.5e5, thinning=10, seed=0):
    rng = np.random.default_rng(seed)

    # --- minimal fix: ensure int ---
    n_iter = int(n_iter)

    x = x0.copy()
    n_params = x.shape[0]

    means = np.zeros(n_params)

    # --- minimal fix: sigma is std-dev => covariance uses sigma**2 ---
    cov = np.diag(np.full(n_params, sigma))

    n_save = int(np.ceil(n_iter / thinning)) + 1
    chain = np.zeros((n_save, n_params), dtype=float)
    losses = np.zeros(n_save, dtype=float)

    chain[0] = x
    losses[0] = f(x)

    save_i = 1
    T = float(T0)

    for it in range(1, n_iter + 1):
        x_old = x
        x_prop = x_old + rng.multivariate_normal(means, cov)

        dE = f(x_prop) - f(x_old)

        # --- minimal fix: always accept improvements; otherwise accept with exp(-dE/T) ---
        if dE <= 0 or np.exp(-np.clip(dE / max(T, 1e-12), -100, 100)) >= rng.random():
            x = x_prop

        # linear cooling schedule
        T = T0 * (1 - it / n_iter)

        # --- minimal fix: keep T positive (avoid T=0 at end) ---
        T = max(T, 1e-12)

        if it % thinning == 0:
            chain[save_i] = x
            losses[save_i] = f(x)
            save_i += 1

    return chain[:save_i], losses[:save_i]


def _worker_run_one_idx(args):
    """Worker function to run one hyperparameter combination."""
    (idx, n_iter, thinning, seed, outdir, time_points, data_points, save_plots) = args

    # Map idx -> (i, j) for an 8x8 grid
    i = idx // 8
    j = idx % 8
    T0 = float(T0_GRID[i])
    sigma = float(SIGMA_GRID[j])

    # Define loss
    def mse(x):
        return float(np.mean((data_points - model(time_points, x)) ** 2))

    t_start = time.perf_counter()
    chain, loss_hist = sa_tune(
        x0=X0,
        T0=T0,
        sigma=sigma,
        f=mse,
        n_iter=n_iter,
        thinning=thinning,
        seed=seed + idx,   # different seed per idx
    )
    t_end = time.perf_counter()

    final_x = chain[-1].tolist()
    final_mse = float(loss_hist[-1])

    # Create visualization: MSE curve over iterations (similar to notebook)
    plot_path = None
    if save_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_hist, linewidth=1.5)
        plt.xlabel('Iteration (thinned)', fontsize=12)
        plt.ylabel('MSE', fontsize=12)
        plt.title(f'Tuning idx={idx}: T0={T0}, σ={sigma:.2e}\nFinal MSE={final_mse:.6e}', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale often better for MSE

        plot_path = os.path.join(outdir, f"mse_curve_{idx:02d}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    out = {
        "idx": idx,
        "T0": T0,
        "sigma": sigma,
        "n_iter": n_iter,
        "thinning": thinning,
        "final_mse": final_mse,
        "final_x": final_x,
        "wall_time_sec": float(t_end - t_start),
    }

    out_path = os.path.join(outdir, f"tuning_{idx:02d}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    plot_msg = f" (plot: {plot_path})" if save_plots and plot_path else ""
    return f"[idx={idx}] T0={T0} sigma={sigma} final_mse={final_mse} -> {out_path}{plot_msg}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None, help="Single idx (0..63) or None for parallel mode")
    ap.add_argument("--start_idx", type=int, default=None, help="Start idx for parallel mode (inclusive)")
    ap.add_argument("--end_idx", type=int, default=None, help="End idx for parallel mode (exclusive)")
    ap.add_argument("--outdir", type=str, default="results_tuning")
    ap.add_argument("--n_iter", type=int, default=250000)
    ap.add_argument("--thinning", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save_plots", action="store_true", default=True, help="Save MSE curve plots (default: True)")
    ap.add_argument("--no_plots", dest="save_plots", action="store_false", help="Disable plot saving")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data once (shared across workers)
    data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)
    time_points = data[:, 0]
    data_points = data[:, 1]

    # Determine which indices to run
    if args.idx is not None:
        # Single index mode (backward compatible)
        idx = args.idx
        if idx < 0 or idx >= 64:
            raise ValueError("idx must be in [0, 63]")
        indices = [idx]
    elif args.start_idx is not None and args.end_idx is not None:
        # Parallel mode: run range of indices
        indices = list(range(args.start_idx, args.end_idx))
        if any(idx < 0 or idx >= 64 for idx in indices):
            raise ValueError("All indices must be in [0, 63]")
    else:
        raise ValueError("Must specify either --idx or both --start_idx and --end_idx")

    # Use SLURM CPU allocation if present, otherwise use all available cores
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))

    # Don't use more workers than indices
    n_workers = min(n_workers, len(indices))

    # Prepare worker inputs
    worker_inputs = [
        (idx, args.n_iter, args.thinning, args.seed, args.outdir, time_points, data_points, args.save_plots)
        for idx in indices
    ]

    t_total_start = time.perf_counter()

    if n_workers == 1 or len(indices) == 1:
        # Sequential execution
        results = list(map(_worker_run_one_idx, worker_inputs))
    else:
        # Parallel execution
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_run_one_idx, worker_inputs)

    t_total_end = time.perf_counter()

    # Print all results
    for result in results:
        print(result)

    print(f"\nTotal wall time: {t_total_end - t_total_start:.2f}s")
    print(f"Processed {len(indices)} combinations using {n_workers} workers")


if __name__ == "__main__":
    main()