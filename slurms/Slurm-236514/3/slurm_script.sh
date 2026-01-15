#!/bin/bash
#SBATCH --job-name=mc_hist_multiple_iter
#SBATCH --output=Slurm-%j.out
#SBATCH --error=Slurm-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --constraint=rhel8
#SBATCH --array=0-4
#SBATCH --chdir=/cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025

# Array of n_iter values: 2.5e5, 5e5, 1e6, 1e7, 1e8
N_ITER_VALUES=(250000 500000 1000000 10000000 100000000)

# Get n_iter for this array job
N_ITER=${N_ITER_VALUES[$SLURM_ARRAY_TASK_ID]}

# Load modules
module purge
module load DefaultModules
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

# Run plot_mc_histogram.py
python3 plot_mc_histogram.py \
    --T0 10.0 \
    --sigma 1e-5 \
    --n_iter ${N_ITER} \
    --burn_in_values 0 \
    --outdir results_burnin_analysis

echo "Completed n_iter=${N_ITER}"
