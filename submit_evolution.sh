#!/bin/bash
#SBATCH --account=def-skelly
#SBATCH --job-name=flappy_evolution
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=128M
#SBATCH --time=08:00:00
#SBATCH --output=evolution_%j.out
#SBATCH --error=evolution_%j.err

# Email notifications
# CHANGE THIS to your email address
#SBATCH --mail-user=hillroy@mcmaster.ca
#SBATCH --mail-type=ALL

# Optional: Get seed from command line argument (if you want to override config.yaml)
seed=${1:-42}

# Load Python module
module load python/3.10

# Create virtual environment in node-local storage (faster I/O)
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Upgrade pip
pip install --no-index --upgrade pip

# Install packages available in wheelhouse (excluding flappy-bird-env)
# Create a temporary requirements file without flappy-bird-env
grep -v "flappy-bird-env" requirements.txt > $SLURM_TMPDIR/requirements_wheelhouse.txt
pip install --no-index -r $SLURM_TMPDIR/requirements_wheelhouse.txt

# Ensure submodule is initialized (in case repo was cloned without --recursive)
if [ ! -d "flappy-bird-env" ] || [ -z "$(ls -A flappy-bird-env 2>/dev/null)" ]; then
    echo "Initializing flappy-bird-env submodule..."
    git submodule update --init --recursive
fi

# Install flappy-bird-env from local submodule (editable install)
# This uses the flappy-bird-env submodule in the project directory
pip install --no-index -e ./flappy-bird-env

# Set random seed in environment (optional, if you want to override config.yaml)
export PYTHONHASHSEED=$seed

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory per CPU: 128M (Total: ~4GB)"
echo "Time limit: 48:00:00"
echo "Random seed: $seed"
echo "=========================================="

# Run the evolution script
# The script will use all 32 CPUs automatically (n_jobs: null in config.yaml)
python run_flappy_bird.py config.yaml

# Optional: If you want to pass seed as argument, modify run_flappy_bird.py to accept it
# python run_flappy_bird.py config.yaml --seed $seed

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="

