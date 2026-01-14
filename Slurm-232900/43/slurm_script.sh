#!/usr/bin/env bash
#SBATCH --array=0-63
#SBATCH --job-name=ohpc-tune
#SBATCH --time=0-00:10:00
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --constraint=rhel8

module purge
module load DefaultModules
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

# Run one tuning point
srun python3 tune_sa.py \
  --idx "${SLURM_ARRAY_TASK_ID}" \
  --outdir results_tuning \
  --n_iter 6000 \
  --thinning 50