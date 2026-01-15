#!/usr/bin/env python3
import argparse
import json
import os
import time
import numpy as np
import multiprocessing as mp

# Try to import numba (optional speedup for the model evaluation)
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator so code still runs without numba installed
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    print("Warning: numba not available. Running without JIT compilation.")


# -------------------------
# Initial parameter vector (30 parameters)
# -------------------------
# Structure:
# [T0_1..T0_10, Ts_1..Ts_10, Td_1..Td_10]

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

# Initial state: T0 (10), Ts (10), Td (10)
X0 = np.concatenate([
    T0_INITIAL,
    np.array([0.3] * 10),
    np.array([5.0] * 10)
], dtype=float)

# Input dataset used for calibration
DATA_FILE = "data_Team9.csv"


# -------------------------
# Model definition (original + numba-accelerated)
# -------------------------

def model_original(t, x):
    """
    Reference implementation of the piecewise model (no numba).
    Used for comparison / fallback.
    """
    n_cycles = 10
    T0 = x[:n_cycles]
    Ts = x[n_cycles:2*n_cycles]
    Td = x[2*n_cycles:]

    # Define time intervals between successive T0 values
    intervals = [(T0[ix], T0[ix + 1]) for ix in range(n_cycles - 1)] + [(T0[n_cycles - 1], np.inf)]

    t = np.atleast_1d(t)
    out = np.zeros_like(t, dtype=float)

    # Evaluate model piecewise across intervals
    for ix, (a, b) in enumerate(intervals):
        mask = (a <= t) & (t < b)
        out[mask] = ((t[mask] - a) / Ts[ix]) ** 2 * np.exp(-((t[mask] - a) / Td[ix]) ** 2)

    return out.item() if out.size == 1 else out


@njit(cache=True)
def _model_numba_core(t_arr, Ts, Td, T0ARRAY):
    """
    Numba-compiled core loop: evaluates the model for all time points.
    """
    n_points = len(t_arr)
    n_phases = len(Ts)  # Expected: 10
    out = np.zeros(n_points, dtype=np.float64)

    # For each time value, find its interval and evaluate the phase expression
    for i in range(n_points):
        t_val = t_arr[i]
        for ix in range(n_phases):
            a = T0ARRAY[ix]
            if ix < n_phases - 1:
                b = T0ARRAY[ix + 1]
                if a <= t_val < b:
                    diff = t_val - a
                    out[i] = (diff / Ts[ix]) ** 2 * np.exp(-(diff / Td[ix]) ** 2)
                    break
            else:
                # Last phase: t >= T0_10
                if t_val >= a:
                    diff = t_val - a
                    out[i] = (diff / Ts[ix]) ** 2 * np.exp(-(diff / Td[ix]) ** 2)
                    break
    return out


# Flag to switch between model implementations for benchmarking
USE_ORIGINAL_MODEL = False

def model(t, x):
    """
    Wrapper: uses the numba-accelerated model if available,
    otherwise falls back to the original Python implementation.
    """
    if USE_ORIGINAL_MODEL:
        return model_original(t, x)

    n_cycles = 10
    T0 = x[:n_cycles]
    Ts = x[n_cycles:2*n_cycles]
    Td = x[2*n_cycles:]

    t = np.atleast_1d(t)

    if HAS_NUMBA:
        out = _model_numba_core(t, Ts, Td, T0)
    else:
        # Fallback (same logic as original implementation)
        intervals = [(T0[ix], T0[ix + 1]) for ix in range(n_cycles - 1)] + [(T0[n_cycles - 1], np.inf)]
        out = np.zeros_like(t, dtype=float)
        for ix, (a, b) in enumerate(intervals):
            mask = (a <= t) & (t < b)
            out[mask] = ((t[mask] - a) / Ts[ix]) ** 2 * np.exp(-((t[mask] - a) / Td[ix]) ** 2)

    return out.item() if out.size == 1 else out


# -------------------------
# SA optimisation routine (runs one chain)
# -------------------------

def sa_optimize(x0, T0, sigma, f, n_iter=15000, burn_in=10000, seed=0, measure_iter_time=False):
    """
    Runs simulated annealing to sample/optimise parameters.
    Applies constraints on T0 parameters (bounds + ordering).
    Optionally measures per-iteration runtime.
    """
    rng = np.random.default_rng(seed)

    x = x0.copy()
    n_params = x.shape[0]
    n_cycles = 10  # Number of T0 parameters

    # Mean-zero multivariate normal proposal
    means = np.zeros(n_params)
    cov = np.diag(np.full(n_params, sigma))

    # Store only samples after burn-in
    out = np.zeros((n_iter - burn_in, n_params), dtype=float)

    T = T0
    keep_i = 0

    # T0 constraints: keep each T0 within ±5% of initial and maintain strict ordering
    T0_bounds = np.column_stack([
        T0_INITIAL * 0.95,
        T0_INITIAL * 1.05
    ])

    def apply_constraints(x_prop):
        """Enforce bounds and monotonic order for T0_1..T0_10."""
        x_constrained = x_prop.copy()

        # Clip each T0_i into its allowed range
        for i in range(n_cycles):
            x_constrained[i] = np.clip(x_constrained[i], T0_bounds[i, 0], T0_bounds[i, 1])

        # Repair ordering violations by shifting neighbouring values apart
        for i in range(n_cycles - 1):
            if x_constrained[i] >= x_constrained[i + 1]:
                mid = (x_constrained[i] + x_constrained[i + 1]) / 2
                x_constrained[i] = np.clip(mid - 0.1, T0_bounds[i, 0], T0_bounds[i, 1])
                x_constrained[i + 1] = np.clip(mid + 0.1, T0_bounds[i + 1, 0], T0_bounds[i + 1, 1])

        return x_constrained

    # If requested: do a warm-up, then measure iteration runtimes for a sample window
    iter_times = []
    if measure_iter_time:
        for it in range(1, min(101, n_iter + 1)):
            x_old = x
            x_prop = x_old + rng.multivariate_normal(means, cov)
            x_prop = apply_constraints(x_prop)
            dE = f(x_prop) - f(x_old)
            if np.exp(-np.clip(dE / max(T, 1e-12), -100, 100)) >= rng.random():
                x = x_prop
            T = T0 * (1 - it / n_iter)

        sample_size = min(1000, n_iter - 100)
        for sample_idx in range(sample_size):
            it = 101 + sample_idx
            iter_start = time.perf_counter()

            x_old = x
            x_prop = x_old + rng.multivariate_normal(means, cov)
            x_prop = apply_constraints(x_prop)
            dE = f(x_prop) - f(x_old)
            if np.exp(-np.clip(dE / max(T, 1e-12), -100, 100)) >= rng.random():
                x = x_prop
            T = T0 * (1 - it / n_iter)

            iter_end = time.perf_counter()
            iter_times.append(iter_end - iter_start)

            if it > burn_in:
                out[keep_i] = x
                keep_i += 1

        for it in range(101 + sample_size, n_iter + 1):
            x_old = x
            x_prop = x_old + rng.multivariate_normal(means, cov)
            x_prop = apply_constraints(x_prop)
            dE = f(x_prop) - f(x_old)
            if np.exp(-np.clip(dE / max(T, 1e-12), -100, 100)) >= rng.random():
                x = x_prop
            T = T0 * (1 - it / n_iter)
            if it > burn_in:
                out[keep_i] = x
                keep_i += 1
    else:
        # Standard run: propose → constrain → accept/reject → cool → store after burn-in
        for it in range(1, n_iter + 1):
            x_old = x
            x_prop = x_old + rng.multivariate_normal(means, cov)
            x_prop = apply_constraints(x_prop)
            dE = f(x_prop) - f(x_old)
            if np.exp(-np.clip(dE / max(T, 1e-12), -100, 100)) >= rng.random():
                x = x_prop
            T = T0 * (1 - it / n_iter)
            if it > burn_in:
                out[keep_i] = x
                keep_i += 1

    if measure_iter_time:
        return out, iter_times
    return out


# -------------------------
# Worker: run one chain and return samples
# -------------------------

def _worker_run_one_chain(args):
    """
    Runs one SA chain and returns the post-burn-in samples.
    """
    (chain_id, x0, T0, sigma, n_iter, burn_in, seed, time_points, data_points, measure_iter_time) = args

    # Define MSE loss for this dataset
    def mse(x):
        return float(np.mean((data_points - model(time_points, x)) ** 2))

    result = sa_optimize(
        x0=x0,
        T0=T0,
        sigma=sigma,
        f=mse,
        n_iter=n_iter,
        burn_in=burn_in,
        seed=seed + chain_id,
        measure_iter_time=measure_iter_time
    )

    if measure_iter_time:
        samples, iter_times = result
    else:
        samples = result
        iter_times = None

    return samples, iter_times


# -------------------------
# Main: run multiple chains in parallel and aggregate results
# -------------------------

def main():
    global USE_ORIGINAL_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--T0", type=float, required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--n_chains", type=int, default=10)
    ap.add_argument("--n_iter", type=int, default=250000)
    ap.add_argument("--burn_in", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--outdir", type=str, default="results_calibration")
    ap.add_argument("--measure_iter_time", action="store_true",
                    help="Measure single iteration timing (for performance analysis)")
    ap.add_argument("--use_original_model", action="store_true",
                    help="Use original model (without numba) for performance comparison")
    ap.add_argument("--suffix", type=str, default="",
                    help="Suffix to add to output filename (e.g., 'constraint_test')")
    args = ap.parse_args()

    # Switch model implementation if requested
    USE_ORIGINAL_MODEL = args.use_original_model

    os.makedirs(args.outdir, exist_ok=True)

    # Number of processes for multiprocessing (typically equals SLURM_CPUS_PER_TASK)
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    # Load dataset once
    data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)
    time_points = data[:, 0]
    data_points = data[:, 1]

    # Build chain-specific initial states by adding small Gaussian noise to X0
    n_params = X0.shape[0]
    rng = np.random.default_rng(args.seed)
    noise_scale = 0.01
    x0_list = np.abs(X0 + rng.normal(0, noise_scale, size=(args.n_chains, n_params)))

    # One worker input per chain
    worker_inputs = [
        (cid, x0_list[cid], args.T0, args.sigma, args.n_iter, args.burn_in,
         args.seed, time_points, data_points, args.measure_iter_time)
        for cid in range(args.n_chains)
    ]

    # Run all chains (parallel if n_workers > 1)
    t0 = time.perf_counter()

    if n_workers == 1:
        results = list(map(_worker_run_one_chain, worker_inputs))
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_run_one_chain, worker_inputs)

    t1 = time.perf_counter()
    wall = t1 - t0

    # Collect samples (and timing info if enabled)
    iter_times_all = []
    all_samples_list = []

    for r in results:
        samples, iter_times = r
        all_samples_list.append(samples)
        if args.measure_iter_time and iter_times is not None:
            iter_times_all.append(iter_times)

    # Merge samples from all chains into a single array
    all_samples = np.vstack(all_samples_list)

    # Optional: summarise iteration timing
    if iter_times_all:
        all_iter_times = np.concatenate(iter_times_all)
        avg_iter_time = np.mean(all_iter_times)
        std_iter_time = np.std(all_iter_times)
        print(f"\nPerformance Analysis:")
        print(f"  Average single iteration time: {avg_iter_time*1000:.4f} ms")
        print(f"  Std deviation: {std_iter_time*1000:.4f} ms")
        print(f"  Wall time per iteration (wall_time/n_iter): {wall/args.n_iter*1000:.4f} ms")
        print(f"  Total iterations: {args.n_iter}")
        print(f"  Measured iterations: {len(all_iter_times)}")

    # Compute “center of mass” parameter estimate using histogram-weighted bin centers
    n_params = all_samples.shape[1]
    center_of_mass = np.zeros(n_params)

    print(f"\nCollected {len(all_samples)} samples from {args.n_chains} chains")
    print(f"Total samples shape: {all_samples.shape}")
    print(f"Calculating histogram-based center of mass for {n_params} parameters...")

    for ix in range(n_params):
        counts, bin_edges = np.histogram(all_samples[:, ix], bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        if np.sum(counts) > 0:
            center_of_mass[ix] = np.sum(bin_centers * counts) / np.sum(counts)
        else:
            center_of_mass[ix] = np.mean(all_samples[:, ix])

    print(f"Center of Mass shape: {center_of_mass.shape}")

    # Evaluate final MSE at the center-of-mass parameters
    final_pred = model(time_points, center_of_mass)
    final_mse = float(np.mean((data_points - final_pred) ** 2))

    # Per-chain diagnostic: compute MSE using each chain mean
    chain_mses = []
    samples_per_chain = len(all_samples) // args.n_chains
    for chain_idx in range(args.n_chains):
        start_idx = chain_idx * samples_per_chain
        end_idx = start_idx + samples_per_chain
        chain_mean = np.mean(all_samples[start_idx:end_idx], axis=0)
        chain_mse = float(np.mean((data_points - model(time_points, chain_mean)) ** 2))
        chain_mses.append(chain_mse)
    chain_mses = np.array(chain_mses)

    # Save summary output for plotting/analysis
    out = {
        "T0": float(args.T0),
        "sigma": float(args.sigma),
        "n_chains": int(args.n_chains),
        "n_iter": int(args.n_iter),
        "burn_in": int(args.burn_in),
        "n_workers": int(n_workers),
        "wall_time_sec": float(wall),
        "wall_time_per_iter_sec": float(wall / args.n_iter),
        "final_mse": float(final_mse),
        "chain_mse_mean": float(np.mean(chain_mses)),
        "chain_mse_std": float(np.std(chain_mses)),
        "center_of_mass": center_of_mass.tolist(),
        "has_numba": HAS_NUMBA,
        "n_total_samples": int(len(all_samples)),
    }

    # Add timing stats if measured
    if args.measure_iter_time and iter_times_all:
        all_iter_times = np.concatenate(iter_times_all)
        out["avg_iter_time_sec"] = float(np.mean(all_iter_times))
        out["std_iter_time_sec"] = float(np.std(all_iter_times))
        out["min_iter_time_sec"] = float(np.min(all_iter_times))
        out["max_iter_time_sec"] = float(np.max(all_iter_times))

    # Write one JSON output per run (useful for scaling plots)
    filename = f"calib_workers{n_workers}_chains{args.n_chains}"
    if args.suffix:
        filename += f"_{args.suffix}"
    out_path = os.path.join(args.outdir, f"{filename}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved calibration result -> {out_path}")
    print(f"Workers={n_workers} wall_time={wall:.2f}s final_mse={final_mse:.6g}")


if __name__ == "__main__":
    main()