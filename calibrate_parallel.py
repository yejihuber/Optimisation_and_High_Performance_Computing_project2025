#!/usr/bin/env python3
import argparse
import json
import os
import time
import numpy as np
import multiprocessing as mp

# Try to import numba for performance optimization
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    print("Warning: numba not available. Running without JIT compilation.")

# Initial parameter array structure: [T0_1, T0_2, ..., T0_10, Ts_1, Ts_2, ..., Ts_10, Td_1, Td_2, ..., Td_10]
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


# Original model function (for comparison)
def model_original(t, x):
    """Original model implementation without numba optimization.
    Parameter structure: [T0_1, ..., T0_10, Ts0, ..., Ts9, Td0, ..., Td9] (30 params)
    All T0 values (T0_1 to T0_10) will be optimized (no fixed T0 value)
    """
    n_cycles = 10
    T0 = x[:n_cycles]  # T0_1 to T0_10 
    Ts = x[n_cycles:2*n_cycles]  # Ts_1 to Ts_10
    Td = x[2*n_cycles:]  # Td_1 to Td_10
    
    # T0ARRAY is directly set to T0 values from parameter vector x 
    T0ARRAY = T0 
    
    # Create intervals: (T0_1, T0_2), (T0_2, T0_3), ..., (T0_9, T0_10), (T0_10, inf)
    # Note: We need 10 intervals for 10 T0 values
    intervals = [(T0ARRAY[ix], T0ARRAY[ix + 1]) for ix in range(n_cycles - 1)] + [(T0ARRAY[n_cycles - 1], np.inf)]

    t = np.atleast_1d(t)
    out = np.zeros_like(t, dtype=float)

    for ix, (a, b) in enumerate(intervals):
        mask = (a <= t) & (t < b)
        out[mask] = ((t[mask] - T0ARRAY[ix]) / Ts[ix]) ** 2 * np.exp(-((t[mask] - T0ARRAY[ix]) / Td[ix]) ** 2)

    return out.item() if out.size == 1 else out


# Optimized model function with numba JIT compilation
@njit(cache=True)
def _model_numba_core(t_arr, Ts, Td, T0ARRAY):
    """Numba-optimized core computation.
    T0ARRAY contains T0_1 to T0_10 (values from parameter vector x, optimized by SA, no fixed T0).
    We have 10 intervals: (T0_1, T0_2), ..., (T0_9, T0_10), (T0_10, inf)
    """
    n_points = len(t_arr)
    n_phases = len(Ts)  # Should be 10
    out = np.zeros(n_points, dtype=np.float64)
    
    for i in range(n_points):
        t_val = t_arr[i]
        for ix in range(n_phases):
            a = T0ARRAY[ix]
            if ix < n_phases - 1:
                # Regular intervals: (T0_ix, T0_ix+1)
                b = T0ARRAY[ix + 1]
                if a <= t_val < b:
                    diff = t_val - a
                    out[i] = (diff / Ts[ix]) ** 2 * np.exp(-(diff / Td[ix]) ** 2)
                    break
            else:
                # Last interval: (T0_9, inf) - just check lower bound
                if t_val >= a:
                    diff = t_val - a
                    out[i] = (diff / Ts[ix]) ** 2 * np.exp(-(diff / Td[ix]) ** 2)
                    break
    return out


# Global flag to control which model function to use (for performance testing)
USE_ORIGINAL_MODEL = False

def model(t, x):
    """Optimized model function using numba if available.
    Parameter structure: [T0_1, ..., T0_10, Ts0, ..., Ts9, Td0, ..., Td9] (30 params)
    """
    # For performance testing: use original model if flag is set
    if USE_ORIGINAL_MODEL:
        return model_original(t, x)
    
    n_cycles = 10
    T0 = x[:n_cycles]  # T0_1 to T0_10
    Ts = x[n_cycles:2*n_cycles]  # Ts_1 to Ts_10
    Td = x[2*n_cycles:]  # Td_1 to Td_10
    
    # T0ARRAY is directly set to T0 values from parameter vector x
    T0ARRAY = T0
    
    t = np.atleast_1d(t)
    
    if HAS_NUMBA:
        out = _model_numba_core(t, Ts, Td, T0ARRAY)
    else:
        # Fallback to original implementation
        # Create intervals: (T0_1, T0_2), (T0_2, T0_3), ..., (T0_9, T0_10), (T0_10, inf)
        # Note: We need 10 intervals for 10 T0 values
        intervals = [(T0ARRAY[ix], T0ARRAY[ix + 1]) for ix in range(n_cycles - 1)] + [(T0ARRAY[n_cycles - 1], np.inf)]
        out = np.zeros_like(t, dtype=float)
        for ix, (a, b) in enumerate(intervals):
            mask = (a <= t) & (t < b)
            out[mask] = ((t[mask] - T0ARRAY[ix]) / Ts[ix]) ** 2 * np.exp(-((t[mask] - T0ARRAY[ix]) / Td[ix]) ** 2)
    
    return out.item() if out.size == 1 else out


def sa_optimize(x0, T0, sigma, f, n_iter=15000, burn_in=10000, seed=0, measure_iter_time=False):
    rng = np.random.default_rng(seed)

    x = x0.copy()
    n_params = x.shape[0]
    n_cycles = 10  # Number of T0 parameters

    means = np.zeros(n_params)
    cov = np.diag(np.full(n_params, sigma))

    out = np.zeros((n_iter - burn_in, n_params), dtype=float)

    T = T0
    keep_i = 0
    
    # Constraints for T0 parameters: maintain order and stay close to initial values
    # T0 values must satisfy: T0_1 < T0_2 < ... < T0_10
    # And stay within ±5% of initial values
    T0_bounds = np.column_stack([
        T0_INITIAL * 0.95,  # Lower bounds (5% below initial)
        T0_INITIAL * 1.05   # Upper bounds (5% above initial)
    ])
    
    def apply_constraints(x_prop):
        """Apply constraints to T0 parameters: order and bounds."""
        x_constrained = x_prop.copy()
        
        # Apply bounds to T0 parameters
        for i in range(n_cycles):
            x_constrained[i] = np.clip(x_constrained[i], T0_bounds[i, 0], T0_bounds[i, 1])
        
        # Ensure T0 values are in ascending order
        # If order is violated, adjust values to maintain order
        for i in range(n_cycles - 1):
            if x_constrained[i] >= x_constrained[i + 1]:
                # If order is violated, move both values towards their midpoint
                mid = (x_constrained[i] + x_constrained[i + 1]) / 2
                # Ensure both stay within bounds
                x_constrained[i] = np.clip(mid - 0.1, T0_bounds[i, 0], T0_bounds[i, 1])
                x_constrained[i + 1] = np.clip(mid + 0.1, T0_bounds[i + 1, 0], T0_bounds[i + 1, 1])
        
        return x_constrained
    
    # Measure single iteration time
    iter_times = []
    if measure_iter_time:
        # Warm-up iterations (first 100 iterations)
        for it in range(1, min(101, n_iter + 1)):
            x_old = x
            x_prop = x_old + rng.multivariate_normal(means, cov)
            x_prop = apply_constraints(x_prop)  # Apply T0 constraints
            dE = f(x_prop) - f(x_old)
            if np.exp(-np.clip(dE / max(T, 1e-12), -100, 100)) >= rng.random():
                x = x_prop
            T = T0 * (1 - it / n_iter)
        
        # Measure iteration time (sample 1000 iterations)
        sample_size = min(1000, n_iter - 100)
        for sample_idx in range(sample_size):
            it = 101 + sample_idx
            iter_start = time.perf_counter()
            
            x_old = x
            x_prop = x_old + rng.multivariate_normal(means, cov)
            x_prop = apply_constraints(x_prop)  # Apply T0 constraints
            dE = f(x_prop) - f(x_old)
            if np.exp(-np.clip(dE / max(T, 1e-12), -100, 100)) >= rng.random():
                x = x_prop
            T = T0 * (1 - it / n_iter)
            
            iter_end = time.perf_counter()
            iter_times.append(iter_end - iter_start)
            
            if it > burn_in:
                out[keep_i] = x
                keep_i += 1
        
        # Continue with remaining iterations
        for it in range(101 + sample_size, n_iter + 1):
            x_old = x
            x_prop = x_old + rng.multivariate_normal(means, cov)
            x_prop = apply_constraints(x_prop)  # Apply T0 constraints
            dE = f(x_prop) - f(x_old)
            if np.exp(-np.clip(dE / max(T, 1e-12), -100, 100)) >= rng.random():
                x = x_prop
            T = T0 * (1 - it / n_iter)
            if it > burn_in:
                out[keep_i] = x
                keep_i += 1
    else:
        # Normal execution without timing
        for it in range(1, n_iter + 1):
            x_old = x
            x_prop = x_old + rng.multivariate_normal(means, cov)
            x_prop = apply_constraints(x_prop)  # Apply T0 constraints
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


def _worker_run_one_chain(args):
    (chain_id, x0, T0, sigma, n_iter, burn_in, seed, time_points, data_points, measure_iter_time) = args

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

    # Return all samples for histogram-based center of mass calculation
    # This uses more memory but allows better estimation using distribution information
    return samples, iter_times


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
    
    # Set global flag for model selection
    USE_ORIGINAL_MODEL = args.use_original_model

    os.makedirs(args.outdir, exist_ok=True)

    # Use Slurm CPU allocation if present (single-node scaling)
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    # Load data
    data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)
    time_points = data[:, 0]
    data_points = data[:, 1]

    # Create noisy initial conditions for each chain
    # Similar to provided code: use small fixed noise scale (0.001) for all parameters
    n_params = X0.shape[0]
    rng = np.random.default_rng(args.seed)
    # Use small fixed noise scale like provided code: params0noisy = params_from_tuning + np.random.normal(0, 0.001, ...)
    # This keeps initial conditions very close to X0
    noise_scale = 0.001  # Fixed small noise scale (not proportional to parameter values)
    x0_list = np.abs(X0 + rng.normal(0, noise_scale, size=(args.n_chains, n_params)))

    # Prepare worker inputs
    worker_inputs = [
        (cid, x0_list[cid], args.T0, args.sigma, args.n_iter, args.burn_in, args.seed, time_points, data_points, args.measure_iter_time)
        for cid in range(args.n_chains)
    ]

    t0 = time.perf_counter()

    if n_workers == 1:
        results = list(map(_worker_run_one_chain, worker_inputs))
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_run_one_chain, worker_inputs)

    t1 = time.perf_counter()
    wall = t1 - t0
    
    # Extract iteration timing if measured and collect all samples
    iter_times_all = []
    all_samples_list = []
    
    if args.measure_iter_time:
        for r in results:
            samples, iter_times = r
            all_samples_list.append(samples)
            if iter_times is not None:
                iter_times_all.append(iter_times)
    else:
        for r in results:
            samples = r
            all_samples_list.append(samples)
    
    # Concatenate all samples from all chains into one array
    # Shape: (n_chains * (n_iter - burn_in), n_params)
    all_samples = np.vstack(all_samples_list)
    
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

    # Calculate Center of Mass using histogram-based weighted average
    # This method considers the distribution of samples, giving more weight to frequent values
    n_params = all_samples.shape[1]
    center_of_mass = np.zeros(n_params)
    
    print(f"\nCollected {len(all_samples)} samples from {args.n_chains} chains")
    print(f"Total samples shape: {all_samples.shape}")
    print(f"Calculating histogram-based center of mass for {n_params} parameters...")
    
    for ix in range(n_params):
        counts, bin_edges = np.histogram(all_samples[:, ix], bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Calculate weighted average: sum(bin_center * count) / sum(count)
        # This gives more weight to bins with more samples
        if np.sum(counts) > 0:
            center_of_mass[ix] = np.sum(bin_centers * counts) / np.sum(counts)
        else:
            # Fallback to simple mean if histogram fails
            center_of_mass[ix] = np.mean(all_samples[:, ix])
    
    print(f"Center of Mass shape: {center_of_mass.shape}")

    # Final MSE of the center-of-mass
    final_pred = model(time_points, center_of_mass)
    final_mse = float(np.mean((data_points - final_pred) ** 2))
    
    # Calculate chain statistics: compute MSE for each chain's mean
    chain_mses = []
    samples_per_chain = len(all_samples) // args.n_chains
    for chain_idx in range(args.n_chains):
        start_idx = chain_idx * samples_per_chain
        end_idx = start_idx + samples_per_chain
        chain_mean = np.mean(all_samples[start_idx:end_idx], axis=0)
        chain_mse = float(np.mean((data_points - model(time_points, chain_mean)) ** 2))
        chain_mses.append(chain_mse)
    chain_mses = np.array(chain_mses)

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
    
    # Add iteration timing if measured
    if args.measure_iter_time and iter_times_all:
        all_iter_times = np.concatenate(iter_times_all)
        out["avg_iter_time_sec"] = float(np.mean(all_iter_times))
        out["std_iter_time_sec"] = float(np.std(all_iter_times))
        out["min_iter_time_sec"] = float(np.min(all_iter_times))
        out["max_iter_time_sec"] = float(np.max(all_iter_times))

    # Write one JSON per run so you can plot wall-time vs cores later
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