#!/usr/bin/env bash
# Quick activation script for the notebook environment

module purge
module load DefaultModules
module load USS/2022 gcc/9.4.0-pe5.34 python/3.9.12-pe5.34

# Add user's local bin to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=========================================="
echo "Environment activated!"
echo "=========================================="
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""
echo "To start Jupyter:"
echo "  jupyter notebook"
echo "  or"
echo "  jupyter lab"
echo ""
echo "Remember to select kernel 'Python (OHPC Team9)' in your notebook!"
echo "=========================================="

