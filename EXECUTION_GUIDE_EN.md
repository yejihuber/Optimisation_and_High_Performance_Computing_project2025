# Execution Guide (English)

## Complete Execution Workflow

**Note**: Hyperparameter Tuning has already been completed, so we start from Step 1 (Calibration).

---

### Step 1: Calibration Scaling Study

The optimal hyperparameter values (T0=10.0, sigma=1e-5) are already set in the script.

```bash
# Run on cluster
cd /cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025

# Submit calibration scaling job
# This submits 6 array jobs (1, 2, 4, 8, 16, 32 cores)
sbatch submit_calibration_scaling.sh

# Check job status
squeue -u huberyej
```

**Output**: `results_calibration/calib_workers1_chains10.json` ~ `calib_workers32_chains10.json` files

**Runtime**: Approximately 10-20 minutes per job (for n_iter=250000)

**Configured Values**:
- T0 = 10.0
- sigma = 1e-5
- n_chains = 10
- n_iter = 250000
- burn_in = 250000

---

### Step 2: Generate Scaling Plot

After all calibration jobs complete:

```bash
# Generate scaling plot
python3 plot_scaling.py \
  --outdir results_calibration \
  --output scaling_plot.png

# Verify result
ls -lh scaling_plot.png
```

**Output**: `scaling_plot.png` file (wall time vs number of cores graph)

**Plot Contents**:
- Actual measurements (wall time vs cores)
- Ideal scaling curve (for comparison)
- Speedup and Efficiency statistics

---

### Step 3: Performance Measurement (Before/After Comparison) - Required

**Requirement**: "Performance increase: Timing of single iteration before and after your adjustments"

This step is mandatory. You must compare performance before and after numba optimization to measure the improvement.

#### Before (Original version, without numba):

```bash
python3 calibrate_parallel.py \
  --T0 10.0 \
  --sigma 1e-5 \
  --n_chains 10 \
  --n_iter 10000 \
  --burn_in 5000 \
  --measure_iter_time \
  --outdir results_performance_before
```

#### After (Numba optimized version):

```bash
# Run in environment with numba installed
python3 calibrate_parallel.py \
  --T0 10.0 \
  --sigma 1e-5 \
  --n_chains 10 \
  --n_iter 10000 \
  --burn_in 5000 \
  --measure_iter_time \
  --outdir results_performance_after
```

#### Compare Results:

```bash
# Check Before results
cat results_performance_before/calib_workers1_chains10.json | \
  grep -E "(avg_iter_time|wall_time_per_iter|has_numba)"

# Check After results
cat results_performance_after/calib_workers1_chains10.json | \
  grep -E "(avg_iter_time|wall_time_per_iter|has_numba)"
```

---

## Job Monitoring

### Check Job Status

```bash
# Check currently running jobs
squeue -u huberyej

# Check job history
sacct -u huberyej --format=JobID,JobName,State,ExitCode,Elapsed

# Detailed information for specific job
scontrol show job <JOB_ID>
```

### Verify Result Files

```bash
# Check calibration results (6 files)
ls -lh results_calibration/*.json | wc -l  # Should be 6

# List all files
ls -lh results_calibration/

# Check JSON file content (example)
cat results_calibration/calib_workers1_chains10.json
```

---

## Quick Start (Complete Process)

```bash
# Run all at once on cluster

# 1. Navigate to directory
cd /cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025

# 2. Run calibration scaling study
sbatch submit_calibration_scaling.sh

# 3. Wait for jobs to complete (check with squeue)

# 4. Generate plot
python3 plot_scaling.py

# 5. Performance measurement (Before/After comparison) - Required
# Before measurement
python3 calibrate_parallel.py --T0 10.0 --sigma 1e-5 --n_chains 10 --n_iter 10000 --burn_in 5000 --measure_iter_time --outdir results_performance_before

# After measurement (after installing numba)
python3 calibrate_parallel.py --T0 10.0 --sigma 1e-5 --n_chains 10 --n_iter 10000 --burn_in 5000 --measure_iter_time --outdir results_performance_after
```

---

## Important Notes

1. **Path Check**: Verify that the `--chdir` path in `submit_calibration_scaling.sh` is correct
2. **Optimal Values**: T0=10.0 and sigma=1e-5 are already set in the script
3. **Numba**: For performance measurement, numba must be installed to see optimization effects
4. **Runtime**: Each calibration job may take time since n_iter=250000 (approximately 10-20 minutes)
5. **Memory**: Each job requests 8GB of memory

---

## Troubleshooting

### If Jobs Fail

```bash
# Check error log
cat Slurm-<JOB_ID>/slurm-<JOB_ID>.err

# Check output log
cat Slurm-<JOB_ID>/slurm-<JOB_ID>.out

# Or find most recent log
ls -lt Slurm-*/slurm-*.err | head -1
```

### If Result Files Are Missing

```bash
# Check if job completed
sacct -j <JOB_ID>

# Check result directory
ls -la results_calibration/

# Check if jobs are still running
squeue -u huberyej
```

### Verify All Jobs Completed

```bash
# Check all tasks of array job
sacct -j <ARRAY_JOB_ID> --format=JobID,JobName,State,ExitCode

# Example: If Job ID is 123456
sacct -j 123456 --format=JobID,JobName,State,ExitCode
```

---

## Expected File Structure

After execution, the following files will be created:

```
Optimisation_and_High_Performance_Computing_project2025/
├── results_calibration/
│   ├── calib_workers1_chains10.json    (1 core)
│   ├── calib_workers2_chains10.json    (2 cores)
│   ├── calib_workers4_chains10.json    (4 cores)
│   ├── calib_workers8_chains10.json    (8 cores)
│   ├── calib_workers16_chains10.json   (16 cores)
│   └── calib_workers32_chains10.json   (32 cores)
└── scaling_plot.png                     (wall time vs cores graph)
```

Each JSON file contains:
- `wall_time_sec`: Total wall time
- `wall_time_per_iter_sec`: wall_time / n_iter
- `n_workers`: Number of cores used
- `center_of_mass`: Final model parameter estimates
- `final_mse`: Final MSE value
