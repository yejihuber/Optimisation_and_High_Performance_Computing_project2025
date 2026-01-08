#!/bin/bash
# Simple test script to debug the tuning script

cd /cfs/earth/scratch/huberyej/Optimisation_and_High_Performance_Computing_project2025 || exit 1

echo "=== Environment Test ==="
echo "Current directory: $(pwd)"
echo "Python version:"
python3 --version

echo ""
echo "=== File Check ==="
ls -la tune_sa.py data_Team9.csv 2>&1

echo ""
echo "=== Python Module Test ==="
python3 -c "import numpy; print('numpy OK')" || echo "numpy FAILED"
python3 -c "import json; print('json OK')" || echo "json FAILED"
python3 -c "import multiprocessing; print('multiprocessing OK')" || echo "multiprocessing FAILED"

echo ""
echo "=== Test Script Syntax ==="
python3 -m py_compile tune_sa.py && echo "Syntax OK" || echo "Syntax ERROR"

echo ""
echo "=== Test Run (single index) ==="
python3 tune_sa.py --idx 0 --n_iter 100 --thinning 10 --outdir test_results 2>&1 | head -20
