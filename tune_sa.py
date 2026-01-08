#!/usr/bin/env python3
import argparse
import json
import os
import time
import numpy as np
import multiprocessing as mp

# -------------------------
# Project-specific settings
# -------------------------

# 8 values for T0 (exactly as requested: 8x8=64)
T0_GRID = np.array([0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 8.0, 10.0], dtype=float)

# 8 values for sigma (log spaced usually makes most sense)
SIGMA_GRID = np.logspace(-11, -4, 8, dtype=float)

# NOTE: your notebook parameterization uses Ts/Td pairs (10 phases -> 20 params)
X0 = np.array([0.3, 5, 0.3, 5, 0.3, 5, 0.3, 5, 0.3, 5,
               0.3, 5, 0.3, 5, 0.3, 5, 0.3, 5, 0.3, 5], dtype=float)

# Cycle start times (must be 11 values for 10 phases)
# This matches the Hathaway-like list your notebook was aiming for.
T0ARRAY = np.array([
    1878.9166666667,
    1890.1666666667,
    1902.0000000000,
    1913.5833333333,
    1923.5833333333,
    1933.6666666667,
    1944.0833333333,
    1954.2500000000,
    1964.7500000000,
    1976.1666666667,
    1986.6666666667
], dtype=float)

DATA_FILE = "data_Team9.csv"


# -------------------------
# Model + loss (same names as your notebook: model, mse)
# -------------------------

def model(t, x):
    Ts = x[::2]
    Td = x[1::2]
    intervals = [(T0ARRAY[ix], T0ARRAY[ix + 1]) for ix in range(len(T0ARRAY) - 1)]

    t = np.atleast_1d(t)
    out = np.zeros_like(t, dtype=float)

    for ix, (a, b) in enumerate(intervals):
        mask = (a <= t) & (t < b)
        out[mask] = ((t[mask] - T0ARRAY[ix]) / Ts[ix]) ** 2 * np.exp(-((t[mask] - T0ARRAY[ix]) / Td[ix]) ** 2)

    return out.item() if out.size == 1 else out


def sa_tune(x0, T0, sigma, f, n_iter=2.5e5, thinning=10, seed=0):
    rng = np.random.default_rng(seed)

    x = x0.copy()
    n_params = x.shape[0]

    means = np.zeros(n_params)
    cov = np.diag(np.full(n_params, sigma))

    n_save = int(np.ceil(n_iter / thinning)) + 1
    chain = np.zeros((n_save, n_params), dtype=float)
    losses = np.zeros(n_save, dtype=float)

    chain[0] = x
    losses[0] = f(x)

    save_i = 1
    T = T0

    for it in range(1, n_iter + 1):
        x_old = x
        x_prop = x_old + rng.multivariate_normal(means, cov)

        dE = f(x_prop) - f(x_old)
        # For tuning we use acceptance without dividing by T (your notebook style)
        if np.exp(-np.clip(dE, -100, 100)) >= rng.random():
            x = x_prop

        # linear cooling (same idea as your notebook)
        T = T0 * (1 - it / n_iter)

        if it % thinning == 0:
            chain[save_i] = x
            losses[save_i] = f(x)
            save_i += 1

    return chain[:save_i], losses[:save_i]


def _worker_run_one_idx(args):
    """Worker function to run one hyperparameter combination."""
    (idx, n_iter, thinning, seed, outdir, time_points, data_points) = args
    
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
    
    return f"[idx={idx}] T0={T0} sigma={sigma} final_mse={final_mse} -> {out_path}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None, help="Single idx (0..63) or None for parallel mode")
    ap.add_argument("--start_idx", type=int, default=None, help="Start idx for parallel mode (inclusive)")
    ap.add_argument("--end_idx", type=int, default=None, help="End idx for parallel mode (exclusive)")
    ap.add_argument("--outdir", type=str, default="results_tuning")
    ap.add_argument("--n_iter", type=int, default=6000)
    ap.add_argument("--thinning", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
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
        (idx, args.n_iter, args.thinning, args.seed, args.outdir, time_points, data_points)
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