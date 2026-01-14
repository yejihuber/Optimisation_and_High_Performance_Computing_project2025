#!/usr/bin/env bash
#SBATCH --job-name=perf-test
#SBATCH --time=0-02:00:00
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --constraint=rhel8
#SBATCH --chdir=/cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025

module purge
module load DefaultModules
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

# Performance test parameters (matching calib_workers32_chains10.json conditions)
T0=0.5
SIGMA=1e-6
N_CHAINS=10
N_ITER=250000
BURN_IN=200000

# Create output directories
mkdir -p results_performance/results_performance_before
mkdir -p results_performance/results_performance_after

echo "=== Running Performance Test: BEFORE (original model, no numba) ==="
python3 calibrate_parallel.py \
  --T0 "${T0}" \
  --sigma "${SIGMA}" \
  --n_chains "${N_CHAINS}" \
  --n_iter "${N_ITER}" \
  --burn_in "${BURN_IN}" \
  --measure_iter_time \
  --use_original_model \
  --outdir results_performance/results_performance_before

echo ""
echo "=== Running Performance Test: AFTER (numba optimized) ==="
python3 calibrate_parallel.py \
  --T0 "${T0}" \
  --sigma "${SIGMA}" \
  --n_chains "${N_CHAINS}" \
  --n_iter "${N_ITER}" \
  --burn_in "${BURN_IN}" \
  --measure_iter_time \
  --outdir results_performance/results_performance_after

echo ""
echo "=== Performance Test Complete ==="
