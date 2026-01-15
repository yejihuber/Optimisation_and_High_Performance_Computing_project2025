#!/usr/bin/env bash
#SBATCH --job-name=noise-test
#SBATCH --time=0-01:00:00
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=rhel8
#SBATCH --exclusive

module purge
module load DefaultModules

# Option A (matches your job-array template style):
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

# If your course uses micromamba envs instead, comment out the python module above and use:
# source ../../../init_micromamba.sh
# micromamba activate mp

# IMPORTANT: set these to the best values found from tuning (or pass them via sbatch export)
T0_OPT=${T0_OPT:-0.5}
SIGMA_OPT=${SIGMA_OPT:-1e-6}

srun python3 calibrate_parallel.py \
  --T0 "${T0_OPT}" \
  --sigma "${SIGMA_OPT}" \
  --n_chains 32 \
  --n_iter 250000 \
  --burn_in 200000 \
  --suffix "noise_0.01" \
  --outdir results_calibration_noise_test