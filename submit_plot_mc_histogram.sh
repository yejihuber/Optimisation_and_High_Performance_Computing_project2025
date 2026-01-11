#!/bin/bash
#SBATCH --job-name=mc-histogram
#SBATCH --time=0-04:00:00
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --constraint=rhel8
#SBATCH --chdir=/cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025

cd /cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025 || exit 1

module purge
module load DefaultModules
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

echo "Starting Monte Carlo histogram analysis..."
echo "T0=10.0, sigma=1e-5, n_iter=250000"
echo "Burn-in values: 0, 5000, 10000, 20000, 30000"
echo ""

python3 plot_mc_histogram.py \
  --T0 10.0 \
  --sigma 1e-5 \
  --n_iter 250000 \
  --burn_in_values 0 5000 10000 20000 30000 \
  --outdir results_burnin_analysis

echo ""
echo "Job completed!"
