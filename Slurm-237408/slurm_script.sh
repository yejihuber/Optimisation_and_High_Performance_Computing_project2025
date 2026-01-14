#!/bin/bash
#SBATCH --job-name=ohpc-calib-n32-200k
#SBATCH --time=0-04:00:00
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

# Parameters (matching calib_workers32_chains10.json conditions)
T0_OPT=${T0_OPT:-0.5}
SIGMA_OPT=${SIGMA_OPT:-1e-6}
N_CORES=32

# Export the number of cores for the Python script
export SLURM_CPUS_PER_TASK=$N_CORES

echo "=========================================="
echo "Running calibration with $N_CORES cores"
echo "=========================================="
echo "Parameters:"
echo "  n_iter=250000, burn_in=200000"
echo "  T0=${T0_OPT}, sigma=${SIGMA_OPT}"
echo ""

python3 calibrate_parallel.py \
  --T0 "${T0_OPT}" \
  --sigma "${SIGMA_OPT}" \
  --n_chains 10 \
  --n_iter 250000 \
  --burn_in 200000 \
  --measure_iter_time \
  --outdir results_calibration

echo ""
echo "Calibration completed for $N_CORES cores"
