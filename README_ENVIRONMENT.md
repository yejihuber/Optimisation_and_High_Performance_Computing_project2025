# Environment Setup for OHPC_Team9 Notebook

This directory contains the setup for running the `OHPC_Team9.ipynb` notebook.

## Quick Start

### Option 1: Use the activation script (Recommended)

```bash
source activate_env.sh
jupyter notebook
# or
jupyter lab
```

### Option 2: Manual activation

```bash
module purge
module load DefaultModules
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

# Add local bin to PATH
export PATH="$HOME/.local/bin:$PATH"

# Start Jupyter
jupyter notebook
# or
jupyter lab
```

## Using the Notebook

1. Open `OHPC_Team9.ipynb` in Jupyter
2. Select the kernel: **"Python (OHPC Team9)"** from the kernel menu
3. Run cells as needed

## Installed Packages

- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.3.0
- jupyter >= 1.0.0
- notebook >= 6.0.0
- ipykernel >= 6.0.0

All packages are installed in your user directory: `~/.local/lib/python3.9/site-packages`

## Files

- `requirements.txt` - List of required Python packages
- `setup_environment.sh` - Initial setup script (run once)
- `activate_env.sh` - Quick activation script (use each session)
- `README_ENVIRONMENT.md` - This file

## Notes

- The environment uses Python 3.9.12 from the USS/2022 module system
- Packages are installed with `--user` flag, so they're in your home directory
- The Jupyter kernel is registered as "Python (OHPC Team9)"
- Make sure to load the modules before starting Jupyter

