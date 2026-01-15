#!/bin/bash
#SBATCH --job-name=ohpc-calib-scaling
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

# Fixed parameters
T0_OPT=0.5
SIGMA_OPT=1e-6
N_CHAINS=32
N_ITER=250000
BURN_IN=200000

# Scaling study: vary n_workers
N_WORKERS_LIST=(1 2 4 8 16 32)

echo "=========================================="
echo "Scaling Study: Calibration with varying n_workers"
echo "=========================================="
echo "Fixed parameters:"
echo "  T0=${T0_OPT}"
echo "  sigma=${SIGMA_OPT}"
echo "  n_chains=${N_CHAINS}"
echo "  n_iter=${N_ITER}"
echo "  burn_in=${BURN_IN}"
echo ""
echo "Varying n_workers: ${N_WORKERS_LIST[@]}"
echo ""

for N_WORKERS in "${N_WORKERS_LIST[@]}"; do
    echo "=========================================="
    echo "Running with n_workers=${N_WORKERS}"
    echo "=========================================="
    
    # Export the number of cores for the Python script
    export SLURM_CPUS_PER_TASK=${N_WORKERS}
    
    python3 calibrate_parallel.py \
      --T0 "${T0_OPT}" \
      --sigma "${SIGMA_OPT}" \
      --n_chains "${N_CHAINS}" \
      --n_iter "${N_ITER}" \
      --burn_in "${BURN_IN}" \
      --seed 777 \
      --measure_iter_time \
      --suffix "scaling_n${N_WORKERS}" \
      --outdir results_calibration
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed n_workers=${N_WORKERS}"
    else
        echo "✗ Failed n_workers=${N_WORKERS}"
    fi
    echo ""
done

echo "=========================================="
echo "Scaling study completed!"
echo "=========================================="
