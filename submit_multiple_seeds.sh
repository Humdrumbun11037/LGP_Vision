#!/bin/bash
# Script to submit multiple seeds for the same experiment
# Usage: ./submit_multiple_seeds.sh <config_file> <seed1> <seed2> ... <seedN>
# Example: ./submit_multiple_seeds.sh config.yaml 42 123 456 789 999

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_file> <seed1> <seed2> ... <seedN>"
    echo "Example: $0 config.yaml 42 123 456"
    exit 1
fi

CONFIG_FILE=$1
shift  # Remove config file from arguments
SEEDS=("$@")  # Remaining arguments are seeds

echo "=========================================="
echo "Submitting jobs for multiple seeds"
echo "Config file: $CONFIG_FILE"
echo "Seeds: ${SEEDS[@]}"
echo "Total jobs: ${#SEEDS[@]}"
echo "=========================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "Submitting job with seed: $seed"
    sbatch submit_evolution.sh "$seed" "$CONFIG_FILE"
    sleep 1  # Small delay to avoid overwhelming the scheduler
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Check status with: squeue -u $USER"
echo "=========================================="

