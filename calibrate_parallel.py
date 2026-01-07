#!/usr/bin/env python3
import argparse
import json
import os
import time
import numpy as np
import multiprocessing as mp

# Keep same initial x0 as notebook
X0 = np.array([0.3, 5, 0.3, 5, 0.3, 5, 0.3, 5, 0.3, 5,
               0.3, 5, 0.3, 5, 0.3, 5, 0.3, 5, 0.3, 5], dtype=float)

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


def sa_optimize(x0, T0, sigma, f, n_iter=15000, burn_in=10000, seed=0):
    rng = np.random.default_rng(seed)

    x = x0.copy()
    n_params = x.shape[0]

    means = np.zeros(n_params)
    cov = np.diag(np.full(n_params, sigma))

    out = np.zeros((n_iter - burn_in, n_params), dtype=float)

    T = T0
    keep_i = 0

    for it in range(1, n_iter + 1):
        x_old = x
        x_prop = x_old + rng.multivariate_normal(means, cov)

        dE = f(x_prop) - f(x_old)
        # For optimization we divide by T (your notebook style)
        if np.exp(-np.clip(dE / max(T, 1e-12), -100, 100)) >= rng.random():
            x = x_prop

        # linear cooling
        T = T0 * (1 - it / n_iter)

        if it > burn_in:
            out[keep_i] = x
            keep_i += 1

    return out


def _worker_run_one_chain(args):
    (chain_id, x0, T0, sigma, n_iter, burn_in, seed, time_points, data_points) = args

    def mse(x):
        return float(np.mean((data_points - model(time_points, x)) ** 2))

    samples = sa_optimize(
        x0=x0,
        T0=T0,
        sigma=sigma,
        f=mse,
        n_iter=n_iter,
        burn_in=burn_in,
        seed=seed + chain_id
    )

    # return only what we need (saves memory)
    # Using the average of samples as a simple chain summary
    chain_mean = np.mean(samples, axis=0)
    chain_final_mse = mse(chain_mean)
    return chain_mean, chain_final_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T0", type=float, required=True)
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--n_chains", type=int, default=10)
    ap.add_argument("--n_iter", type=int, default=15000)
    ap.add_argument("--burn_in", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--outdir", type=str, default="results_calibration")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Use Slurm CPU allocation if present (single-node scaling)
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    # Load data
    data = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)
    time_points = data[:, 0]
    data_points = data[:, 1]

    # Create noisy initial conditions for each chain
    n_params = X0.shape[0]
    rng = np.random.default_rng(args.seed)
    noise_scale = 0.05 * np.abs(X0) + 1e-6
    x0_list = np.abs(X0 + rng.normal(0, noise_scale, size=(args.n_chains, n_params)))

    # Prepare worker inputs
    worker_inputs = [
        (cid, x0_list[cid], args.T0, args.sigma, args.n_iter, args.burn_in, args.seed, time_points, data_points)
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

    chain_means = np.vstack([r[0] for r in results])
    chain_mses = np.array([r[1] for r in results], dtype=float)

    # Center-of-mass style final estimate: average over chain means
    center_of_mass = np.mean(chain_means, axis=0)

    # Final MSE of the center-of-mass
    final_pred = model(time_points, center_of_mass)
    final_mse = float(np.mean((data_points - final_pred) ** 2))

    out = {
        "T0": float(args.T0),
        "sigma": float(args.sigma),
        "n_chains": int(args.n_chains),
        "n_iter": int(args.n_iter),
        "burn_in": int(args.burn_in),
        "n_workers": int(n_workers),
        "wall_time_sec": float(wall),
        "final_mse": float(final_mse),
        "chain_mse_mean": float(np.mean(chain_mses)),
        "chain_mse_std": float(np.std(chain_mses)),
        "center_of_mass": center_of_mass.tolist(),
    }

    # Write one JSON per run so you can plot wall-time vs cores later
    out_path = os.path.join(args.outdir, f"calib_workers{n_workers}_chains{args.n_chains}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved calibration result -> {out_path}")
    print(f"Workers={n_workers} wall_time={wall:.2f}s final_mse={final_mse:.6g}")


if __name__ == "__main__":
    main()