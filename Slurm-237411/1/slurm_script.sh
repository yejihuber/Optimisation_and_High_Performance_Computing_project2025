#!/bin/bash
#SBATCH --array=0-2
#SBATCH --job-name=ohpc-calib-chains-test
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

# Map array task ID to number of chains (32의 배수)
CHAINS_ARRAY=(32 64 96)
N_CHAINS=${CHAINS_ARRAY[$SLURM_ARRAY_TASK_ID]}
N_WORKERS=32

# Parameters
T0_OPT=${T0_OPT:-0.5}
SIGMA_OPT=${SIGMA_OPT:-1e-6}

# Export the number of cores for the Python script
export SLURM_CPUS_PER_TASK=$N_WORKERS

echo "=========================================="
echo "Running calibration"
echo "=========================================="
echo "Parameters:"
echo "  n_chains=${N_CHAINS} (${N_WORKERS}의 ${N_CHAINS}/${N_WORKERS}배)"
echo "  n_workers=${N_WORKERS} (각 코어당 ${N_CHAINS}/${N_WORKERS}개 체인 처리)"
echo "  n_iter=500000, burn_in=400000"
echo "  T0=${T0_OPT}, sigma=${SIGMA_OPT}"
echo ""

python3 calibrate_parallel.py \
  --T0 "${T0_OPT}" \
  --sigma "${SIGMA_OPT}" \
  --n_chains "${N_CHAINS}" \
  --n_iter 500000 \
  --burn_in 400000 \
  --measure_iter_time \
  --outdir results_calibration

echo ""
echo "Calibration completed: n_chains=${N_CHAINS}, n_workers=${N_WORKERS}"
