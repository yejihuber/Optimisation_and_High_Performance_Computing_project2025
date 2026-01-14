#!/bin/bash
#SBATCH --job-name=ohpc-calib-test-best
#SBATCH --time=0-02:00:00
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8GB
#SBATCH --constraint=rhel8
#SBATCH --chdir=/cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025

cd /cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025 || exit 1

module purge
module load DefaultModules
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

# Parameters (test best sample selection improvement)
T0_OPT=0.5
SIGMA_OPT=1e-6
N_WORKERS=32
N_CHAINS=10
N_ITER=250000
BURN_IN=200000

# Export the number of cores for the Python script
export SLURM_CPUS_PER_TASK=$N_WORKERS

echo "=========================================="
echo "Running calibration with best sample selection"
echo "=========================================="
echo "Parameters:"
echo "  n_chains=${N_CHAINS}"
echo "  n_iter=${N_ITER}"
echo "  burn_in=${BURN_IN}"
echo "  n_workers=${N_WORKERS}"
echo "  T0=${T0_OPT}, sigma=${SIGMA_OPT}"
echo ""
echo "Improvements:"
echo "  - Using minimum MSE sample from each chain (instead of average)"
echo "  - Selecting best solution across all chains"
echo "  - Original noise scale: 5% for all parameters"
echo ""

python3 calibrate_parallel.py \
  --T0 "${T0_OPT}" \
  --sigma "${SIGMA_OPT}" \
  --n_chains "${N_CHAINS}" \
  --n_iter "${N_ITER}" \
  --burn_in "${BURN_IN}" \
  --seed 777 \
  --measure_iter_time \
  --suffix "test_best_sample" \
  --outdir results_calibration

echo ""
echo "Calibration completed"
