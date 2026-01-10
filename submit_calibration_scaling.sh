#!/bin/bash
#SBATCH --array=0-5
#SBATCH --job-name=ohpc-calib-scaling
#SBATCH --time=0-02:00:00
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8GB
#SBATCH --constraint=rhel8
#SBATCH --chdir=/cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025

# Array task mapping:
# 0 -> 1 core
# 1 -> 2 cores
# 2 -> 4 cores
# 3 -> 8 cores
# 4 -> 16 cores
# 5 -> 32 cores

cd /cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025 || exit 1

module purge
module load DefaultModules
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

# Map array task ID to number of cores
CORES_ARRAY=(1 2 4 8 16 32)
N_CORES=${CORES_ARRAY[$SLURM_ARRAY_TASK_ID]}

# IMPORTANT: set these to the best values found from tuning
# Values from OHPC_Team9.ipynb:
# T0_opt = 10.0
# sigma_opt = 9.999999999999999e-06 (≈ 1e-5)
T0_OPT=${T0_OPT:-10.0}
SIGMA_OPT=${SIGMA_OPT:-1e-5}

# Export the number of cores for the Python script
export SLURM_CPUS_PER_TASK=$N_CORES

echo "Running calibration with $N_CORES cores"

python3 calibrate_parallel.py \
  --T0 "${T0_OPT}" \
  --sigma "${SIGMA_OPT}" \
  --n_chains 10 \
  --n_iter 250000 \
  --burn_in 50000 \
  --outdir results_calibration
