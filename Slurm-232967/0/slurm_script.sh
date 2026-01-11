#!/usr/bin/env bash
#SBATCH --array=0-1
#SBATCH --job-name=ohpc-tune
#SBATCH --time=0-01:00:00
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8GB
#SBATCH --constraint=rhel8

module purge
module load DefaultModules
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

# Calculate indices for this array task
# Array task 0: indices 0-31 (32 combinations)
# Array task 1: indices 32-63 (32 combinations)
START_IDX=$((SLURM_ARRAY_TASK_ID * 32))
END_IDX=$((START_IDX + 32))

# Run 32 tuning points in parallel using 32 cores
srun python3 tune_sa.py \
  --start_idx "${START_IDX}" \
  --end_idx "${END_IDX}" \
  --outdir results_tuning \
  --n_iter 250000 \
  --thinning 10