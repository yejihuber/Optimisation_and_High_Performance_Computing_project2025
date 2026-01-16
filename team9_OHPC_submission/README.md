# OHPC Team 9 – Solar Cycle Calibration Project

This repository contains the full submission for the **Optimization and High Performance Computing (OHPC)** project.
The project focuses on calibrating a multi-cycle solar activity model using **Simulated Annealing (SA)** and evaluating
parallel performance on an HPC system.

---

## Important Note (Recommended Workflow)

The Jupyter notebook **OHPC_Team9.ipynb** analyzes and visualizes output files that were generated on the HPC cluster.

For efficiency and reproducibility, **all required cluster jobs (hyperparameter tuning and calibration) have already been executed in advance**, and the corresponding output files are included in this submission.

As a result, the notebook can be run **directly** to reproduce all figures, tables, and results presented in the report,
**without rerunning any cluster jobs**.

The cluster execution steps are nevertheless documented below for completeness and reproducibility, in case the experiments
should be rerun.

---

## 0) Prerequisites

You need:
- the provided dataset `data_Team9.csv`,
- a Python environment as described in `README_ENVIRONMENT.md`,
- access to the OHPC cluster **only if you want to rerun the jobs**.

---

## 1) Environment Setup

Follow the instructions in `README_ENVIRONMENT.md` to activate the Python environment and start Jupyter.

Typical workflow:
- change into the project directory,
- activate the environment,
- start Jupyter Notebook.

Typical session:
```bash
cd Optimisation_and_High_Performance_Computing_project2025
source activate_env.sh
```
---

## 2) (Optional) Hyperparameter Tuning on the Cluster

This step is **optional** and only required if the tuning results should be regenerated.

Hyperparameter tuning consists of multiple independent Simulated Annealing runs.
Each run stores its results as JSON files (and optional MSE convergence plots).

To submit the tuning jobs on the cluster:
- change into the project directory,
- submit the Slurm tuning script.

The tuning outputs are saved to disk and later loaded by the notebook for analysis.

Submit the tuning sweep:
```bash
cd team9_OHPC_submission
sbatch slurm/submit_tune_sa.sh
```

This generates output files in:
`results_tuning`:
- `tuning_00.json ... tuning_63.json`
- `mse_curve_00.png ... mse_curve_63.png`

These files are later loaded by the notebook for hyperparameter analysis.

---

### 3) (Optional) Calibration on the Cluster

This step is optional and only required if the calibration results should be regenerated.

After tuning is complete, run the calibration jobs that generate:
- parallel scaling results (runtime vs number of cores),
- the final best calibration used for model evaluation.

Submit calibration jobs:
```bash
cd team9_OHPC_submission
sbatch slurm/submit_calibration_parallel.sh
```
Calibration results are written to directories such as:
- `results/calibration_scaling/*.json`
- `results/calibration_best/*.json`

All required results are already included in this repository.

---

### 4) Run the Jupyter notebook

The notebook is located in the project root directory:
- `OHPC_Team9.ipynb`

Once the result files exist (included by default), open and run the notebook:

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

To verify that the required result files are present, you can inspect the results directories:
```bash
ls results_calibration
ls results_tuning
ls results_performance
```

You should see tuning JSON/PNG files and calibration JSON files.

---

### 6) Reproducibility Statement

All numerical results, figures, and conclusions presented in the report can be reproduced by running
OHPC_Team9.ipynb using the provided environment and included output files.

Slurm job scripts and configuration files are included to ensure full reproducibility of the HPC workflow.

---


