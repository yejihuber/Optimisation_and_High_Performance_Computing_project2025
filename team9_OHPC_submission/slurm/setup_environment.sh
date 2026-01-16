#!/usr/bin/env bash
# Setup script for OHPC notebook environment

set -e

echo "Setting up environment for OHPC_Team9 notebook..."

# Load required modules
module purge
module load DefaultModules
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

# Get Python path
PYTHON_EXE=$(which python3)
echo "Using Python: $PYTHON_EXE"

# Install packages to user directory
echo "Installing required packages..."
python3 -m pip install --user --upgrade pip
python3 -m pip install --user -r requirements.txt

# Install Jupyter kernel for this environment
echo "Setting up Jupyter kernel..."
python3 -m pip install --user ipykernel
python3 -m ipykernel install --user --name ohpc-team9 --display-name "Python (OHPC Team9)"

echo ""
echo "=" * 60
echo "Environment setup complete!"
echo "=" * 60
echo ""
echo "To use this environment:"
echo "1. Load modules:"
echo "   module purge"
echo "   module load DefaultModules"
echo "   module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34"
echo ""
echo "2. Start Jupyter:"
echo "   jupyter notebook"
echo "   or"
echo "   jupyter lab"
echo ""
echo "3. Select kernel 'Python (OHPC Team9)' in your notebook"
echo ""

