#!/bin/bash
#SBATCH --job-name=calib-mc-complete
#SBATCH --time=0-08:00:00
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

echo "=========================================="
echo "Complete Comparison: Calibration + MC Histogram"
echo "=========================================="
echo ""

# Step 1: Run MC histogram comparison first (takes longer, can run in parallel)
echo "Step 1: Running MC histogram comparison..."
echo "----------------------------------------"
sbatch --dependency=afterok:$SLURM_JOB_ID submit_mc_histogram_comparison.sh

# Step 2: Submit calibration jobs
echo ""
echo "Step 2: Submitting calibration jobs..."
echo "----------------------------------------"
CALIB_JOB_ID=$(sbatch --parsable submit_calibration_with_mc_comparison.sh)
echo "Calibration jobs submitted: $CALIB_JOB_ID"

echo ""
echo "=========================================="
echo "All jobs submitted"
echo "=========================================="
echo "  - MC histogram comparison: submitted"
echo "  - Calibration scaling study: $CALIB_JOB_ID"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
