#!/bin/bash
#SBATCH --job-name=ohpc-calib-burnin1e5
#SBATCH --time=0-04:00:00
#SBATCH --partition=earth-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8GB
#SBATCH --constraint=rhel8
#SBATCH --chdir=/cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025

cd /cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025 || exit 1

module purge
module load DefaultModules
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

# Set number of cores
N_CORES=16

# IMPORTANT: set these to the best values found from tuning
T0_OPT=${T0_OPT:-10.0}
SIGMA_OPT=${SIGMA_OPT:-1e-5}

# Export the number of cores for the Python script
export SLURM_CPUS_PER_TASK=$N_CORES

echo "Running calibration with $N_CORES cores"
echo "n_iter=250000, burn_in=100000 (1e5)"

python3 calibrate_parallel.py \
  --T0 "${T0_OPT}" \
  --sigma "${SIGMA_OPT}" \
  --n_chains 10 \
  --n_iter 250000 \
  --burn_in 100000 \
  --outdir results_calibration_different_burnin

# Rename output file to desired name
if [ -f "results_calibration_different_burnin/calib_workers16_chains10.json" ]; then
    mv "results_calibration_different_burnin/calib_workers16_chains10.json" \
       "results_calibration_different_burnin/calib_workers16_burnin1e5.json"
    echo "Renamed output file to calib_workers16_burnin1e5.json"
fi
