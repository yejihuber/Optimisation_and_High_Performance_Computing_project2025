## How to run (correct workflow)

**Important:**  
**Important:**  
The Jupyter notebook loads and analyzes output files that were generated on the HPC cluster.  
For efficiency, all required cluster jobs (hyperparameter tuning and calibration) were **already executed in advance**, and the corresponding output files are included in this submission.  

As a result, the notebook `OHPC_Team9.ipynb` can be run **directly** to reproduce all figures and results shown in the report **without rerunning the cluster jobs**.  

Steps 2 and 3 are nevertheless described below for **reproducibility purposes**, in case the experiments need to be rerun on the cluster.
**afterwards**.

---

### 0) Prerequisites
You need:
- access to the OHPC cluster (Slurm),
- the Python environment described in `README_ENVIRONMENT.md`,
- the dataset `data_Team9.csv`.

---

### 1) Set up the environment on the cluster
Follow the instructions in **`README_ENVIRONMENT.md`**.

Typical session:
```bash
cd team9_OHPC_submission
source activate_env.sh
```

---

### 2) Run hyperparameter tuning on the cluster
The notebook expects tuning outputs (JSON files and MSE curve images), so this step **must be run first**.

Submit the tuning sweep:
```bash
cd team9_OHPC_submission
sbatch slurm/submit_tune_sa.sh
```

This produces (in `results/tuning/`):
- `tuning_00.json ... tuning_63.json`
- `mse_curve_00.png ... mse_curve_63.png`

These files are later loaded by the notebook for hyperparameter analysis.

---

### 3) Run calibration on the cluster
After tuning is complete, run the calibration jobs that generate:
- parallel scaling results (runtime vs number of cores),
- the final best calibration used for model evaluation.

Submit calibration:
```bash
cd team9_OHPC_submission
sbatch slurm/submit_calibration_parallel.sh
```

This produces (in `results/`):
- `results/calibration_scaling/*.json`
- `results/calibration_best/*.json`

---

### 4) Run the Jupyter notebook
Once all cluster outputs exist, open and run the notebook:

```bash
jupyter lab
# or
jupyter notebook
```

Open:
- `OHPC_Team9.ipynb`

The notebook will:
- load the saved tuning and calibration results,
- generate tables and figures,
- show final model fit, residuals, Ts–Td correlation, and discussion.

---

### 5) Sanity check (optional)
Before running the notebook, verify that the required result files exist:
```bash
find results -maxdepth 3 -type f | sort
```

You should see tuning JSON/PNG files and calibration JSON files.