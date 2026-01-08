#!/usr/bin/env bash
# Find SLURM log files for job 232898/232899

JOB_ID=232898

echo "=========================================="
echo "Searching for SLURM log files..."
echo "=========================================="

# Search in current directory
echo "1. Current directory:"
ls -la slurm-${JOB_ID}* 2>/dev/null || echo "   Not found in current directory"

# Search in Slurm subdirectories
echo ""
echo "2. Slurm subdirectories:"
find . -type d -name "Slurm-*" 2>/dev/null | head -5

# Search for any files with job ID
echo ""
echo "3. All files containing job ID:"
find . -name "*${JOB_ID}*" -o -name "*${JOB_ID}*" 2>/dev/null | head -10

# Check standard SLURM output locations
echo ""
echo "4. Standard SLURM output files:"
ls -la slurm-${JOB_ID}_*.out slurm-${JOB_ID}_*.err 2>/dev/null || echo "   Not found"

# Check if files are in subdirectories
echo ""
echo "5. Checking Slurm-* subdirectories:"
for dir in Slurm-*/; do
    if [ -d "$dir" ]; then
        echo "   Checking $dir"
        ls -la "$dir"slurm-* 2>/dev/null | head -3
    fi
done
