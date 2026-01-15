#!/bin/bash
#SBATCH --job-name=ohpc-calib-default
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

# Parameters (기본값 사용)
T0_OPT=${T0_OPT:-0.5}
SIGMA_OPT=${SIGMA_OPT:-1e-6}
N_WORKERS=32

# Export the number of cores for the Python script
export SLURM_CPUS_PER_TASK=$N_WORKERS

echo "=========================================="
echo "Running calibration with default values"
echo "=========================================="
echo "Parameters:"
echo "  n_chains=10 (default)"
echo "  n_iter=250000 (default)"
echo "  burn_in=200000 (default)"
echo "  n_workers=${N_WORKERS}"
echo "  T0=${T0_OPT}, sigma=${SIGMA_OPT}"
echo ""

python3 calibrate_parallel.py \
  --T0 "${T0_OPT}" \
  --sigma "${SIGMA_OPT}" \
  --measure_iter_time \
  --suffix "noise_test" \
  --outdir results_calibration

echo ""
echo "Calibration completed"
