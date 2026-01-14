#!/bin/bash
#SBATCH --job-name=mc-hist-compare
#SBATCH --time=0-06:00:00
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

# Parameters for comparison
T0=0.5
SIGMA=1e-6

echo "=========================================="
echo "MC Histogram Comparison"
echo "=========================================="
echo "T0=${T0}, sigma=${SIGMA}"
echo ""

# Create output directory
mkdir -p results_mc_comparison

echo "----------------------------------------"
echo "Condition 1: n_iter=250000, burn_in=200000"
echo "----------------------------------------"
python3 plot_mc_histogram.py \
  --T0 "${T0}" \
  --sigma "${SIGMA}" \
  --n_iter 250000 \
  --burn_in_values 0 100000 150000 200000 \
  --outdir results_mc_comparison

# Rename output for condition 1
if [ -f "results_mc_comparison/mc_histogram_iter2.5e5.png" ]; then
    mv results_mc_comparison/mc_histogram_iter2.5e5.png \
       results_mc_comparison/mc_histogram_condition1_niter250k_burnin200k.png
    echo "Saved: results_mc_comparison/mc_histogram_condition1_niter250k_burnin200k.png"
fi

echo ""
echo "----------------------------------------"
echo "Condition 2: n_iter=500000, burn_in=400000"
echo "----------------------------------------"
python3 plot_mc_histogram.py \
  --T0 "${T0}" \
  --sigma "${SIGMA}" \
  --n_iter 500000 \
  --burn_in_values 0 200000 300000 400000 \
  --outdir results_mc_comparison

# Rename output for condition 2
if [ -f "results_mc_comparison/mc_histogram_iter5e5.png" ]; then
    mv results_mc_comparison/mc_histogram_iter5e5.png \
       results_mc_comparison/mc_histogram_condition2_niter500k_burnin400k.png
    echo "Saved: results_mc_comparison/mc_histogram_condition2_niter500k_burnin400k.png"
fi

echo ""
echo "=========================================="
echo "MC Histogram Comparison Complete"
echo "=========================================="
echo "Results saved in: results_mc_comparison/"
echo "  - Condition 1: n_iter=250k, burn_in=200k"
echo "  - Condition 2: n_iter=500k, burn_in=400k"
