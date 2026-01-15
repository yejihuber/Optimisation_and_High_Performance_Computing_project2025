#!/usr/bin/env python3
import argparse
import json
import os
import time
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster execution
import matplotlib.pyplot as plt

# -------------------------
# Hyperparameter grids for SA tuning
# -------------------------

# 8 initial temperature values → combined with sigma gives 8x8 = 64 runs
T0_GRID = np.array([0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 8.0, 10.0], dtype=float)

# 8 proposal scale values (log-spaced)
SIGMA_GRID = np.logspace(-11, -4, 8, dtype=float)

# -------------------------
# Initial parameter vector
# -------------------------
# Structure:
# [T0_1, ..., T0_10, Ts_1, ..., Ts_10, Td_1, ..., Td_10]
# Total: 30 parameters

T0_INITIAL = np.array([
    1878.9166666667,
    1890.1666666667,
    1902.0000000000,
    1913.5833333333,
    1923.5833333333,
    1933.6666666667,
    1944.0833333333,
    1954.2500000000,
    1964.7500000000,
    1976.1666666667
], dtype=float)

# Concatenate full initial state vector
X0 = np.concatenate([
    T0_INITIAL,            # T0 values
    np.array([0.3] * 10),  # Ts values
    np.array([5.0] * 10)   # Td values
], dtype=float)

# Input data file (same as notebook)
DATA_FILE = "data_Team9.csv"

# -------------------------
# Model and loss definition
# -------------------------

def model(t, x):
    """
    Model function used for SA tuning.
    Parameter vector x contains 10 cycles with (T0, Ts, Td).
    """
    n_cycles = 10
    T0_optimized = x[:n_cycles]
    Ts = x[n_cycles:2*n_cycles]
    Td = x[2*n_cycles:]

    # Define time intervals between successive T0 values
    intervals = [(T0_optimized[ix], T0_optimized[ix + 1]) for ix in range(n_cycles - 1)] \
                + [(T0_optimized[n_cycles - 1], np.inf)]

    t = np.atleast_1d(t)
    out = np.zeros_like(t, dtype=float)

    # Evaluate model piecewise per cycle
    for ix, (a, b) in enumerate(intervals):
        mask = (a <= t) & (t < b)
        out[mask] = ((t[mask] - T0_optimized[ix]) / Ts[ix]) ** 2 * \
                    np.exp(-((t[mask] - T0_optimized[ix]) / Td[ix]) ** 2)

    return out.item() if out.size == 1 else out


# -------------------------
# Simulated annealing routine
# -------------------------

def sa_tune(x0, T0, sigma, f, n_iter=2.5e5, thinning=10, seed=0):
    """
    Run simulated annealing for a given (T0, sigma) pair.
    Returns the thinned parameter chain and loss history.
    """
    rng = np.random.default_rng(seed)
    n_iter = int(n_iter)

    x = x0.copy()
    n_params = x.shape[0]

    # Mean-zero proposal distribution
    means = np.zeros(n_params)

    # Diagonal proposal covariance (controlled by sigma)
    cov = np.diag(np.full(n_params, sigma))

    # Allocate storage for thinned chain
    n_save = int(np.ceil(n_iter / thinning)) + 1
    chain = np.zeros((n_save, n_params), dtype=float)
    losses = np.zeros(n_save, dtype=float)

    chain[0] = x
    losses[0] = f(x)

    save_i = 1
    T = float(T0)

    for it in range(1, n_iter + 1):
        # Propose new state
        x_old = x
        x_prop = x_old + rng.multivariate_normal(means, cov)

        # Energy difference
        dE = f(x_prop) - f(x_old)

        # Metropolis acceptance step
        if dE <= 0 or np.exp(-np.clip(dE / max(T, 1e-12), -100, 100)) >= rng.random():
            x = x_prop

        # Linear cooling schedule
        T = T0 * (1 - it / n_iter)
        T = max(T, 1e-12)

        # Save thinned samples
        if it % thinning == 0:
            chain[save_i] = x
            losses[save_i] = f(x)
            save_i += 1

    return chain[:save_i], losses[:save_i]


# -------------------------
# Worker function (one grid point)
# -------------------------

def _worker_run_one_idx(args):
    """
    Runs SA for a single hyperparameter index (one T0, sigma pair).
    """
    (idx, n_iter, thinning, seed, outdir, time_points, data_points, save_plots) = args

    # Map flat index to 8x8 grid
    i = idx // 8
    j = idx % 8
    T0 = float(T0_GRID[i])
    sigma = float(SIGMA_GRID[j])

    # Define MSE loss (same as notebook)
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
        seed=seed + idx
    )
    t_end = time.perf_counter()

    final_x = chain[-1].tolist()
    final_mse = float(loss_hist[-1])

    # Optionally save MSE convergence plot
    plot_path = None
    if save_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_hist)
        plt.xlabel("Iteration (thinned)")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.title(f"Tuning idx={idx}, T0={T0}, sigma={sigma}")
        plt.grid(alpha=0.3)
        plot_path = os.path.join(outdir, f"mse_curve_{idx:02d}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    # Save results to JSON
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


# -------------------------
# Main entry point
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--start_idx", type=int, default=None)
    ap.add_argument("--end_idx", type=int, default=None)
    ap.add_argument("--outdir", type=str, default="results_tuning")
    ap.add_argument("--n_iter", type=int, default=250000)
    ap.add_argument("--thinning", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save_plots", action="store_true", default=True)
    ap.add_argument("--no_plots", dest="save_plots", action="store_false")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data once and share with workers
    data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)
    time_points = data[:, 0]
    data_points = data[:, 1]

    # Determine which tuning indices to run
    if args.idx is not None:
        indices = [args.idx]
    elif args.start_idx is not None and args.end_idx is not None:
        indices = list(range(args.start_idx, args.end_idx))
    else:
        raise ValueError("Must specify --idx or --start_idx and --end_idx")

    # Use SLURM CPU allocation if available
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    n_workers = min(n_workers, len(indices))

    worker_inputs = [
        (idx, args.n_iter, args.thinning, args.seed,
         args.outdir, time_points, data_points, args.save_plots)
        for idx in indices
    ]

    t_total_start = time.perf_counter()

    if n_workers == 1 or len(indices) == 1:
        results = list(map(_worker_run_one_idx, worker_inputs))
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_run_one_idx, worker_inputs)

    t_total_end = time.perf_counter()

    for result in results:
        print(result)

    print(f"\nTotal wall time: {t_total_end - t_total_start:.2f}s")
    print(f"Processed {len(indices)} combinations using {n_workers} workers")


if __name__ == "__main__":
    main()