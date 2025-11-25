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
#SBATCH --mail-user=hillroyx@mcmaster.ca
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

# Get the directory where the script is located (should be project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"
echo "Current directory: $(pwd)"
cd "$SCRIPT_DIR" || exit 1
echo "Changed to: $(pwd)"

# Install packages available in wheelhouse (excluding flappy-bird-env)
# Create a temporary requirements file without flappy-bird-env
grep -v "flappy-bird-env" requirements.txt > $SLURM_TMPDIR/requirements_wheelhouse.txt
pip install --no-index -r $SLURM_TMPDIR/requirements_wheelhouse.txt

# Ensure submodule is initialized (in case repo was cloned without --recursive)
echo "Checking for flappy-bird-env submodule..."
if [ ! -d "flappy-bird-env" ] || [ -z "$(ls -A flappy-bird-env 2>/dev/null)" ]; then
    echo "WARNING: flappy-bird-env submodule not found. Initializing..."
    git submodule update --init --recursive
    if [ ! -d "flappy-bird-env" ]; then
        echo "ERROR: Failed to initialize submodule. Trying alternative installation..."
        # Fallback: try installing from PyPI if available
        pip install flappy-bird-env || {
            echo "ERROR: Could not install flappy-bird-env. Exiting."
            exit 1
        }
    fi
fi

# Verify submodule exists and has content
if [ -d "flappy-bird-env" ]; then
    echo "flappy-bird-env directory found:"
    ls -la flappy-bird-env/ | head -5
    echo "Installing flappy-bird-env from local submodule..."
    # Try without --no-index first (in case it needs to install dependencies)
    pip install -e ./flappy-bird-env || {
        echo "WARNING: Editable install failed, trying regular install..."
        pip install --no-index -e ./flappy-bird-env || {
            echo "WARNING: pip install failed, but submodule exists."
            echo "Code will attempt direct import from submodule path (fallback mode)."
        }
    }
    echo "Verifying import (will work via pip install OR direct submodule import)..."
    python -c "import flappy_bird_env; print('✓ flappy_bird_env imported successfully')" || {
        echo "ERROR: flappy_bird_env import failed!"
        echo "Checking if submodule structure is correct..."
        ls -la flappy-bird-env/flappy_bird_env/ 2>/dev/null || echo "Submodule structure may be incorrect"
        pip list | grep -E "flappy|gymnasium" || true
        exit 1
    }
else
    echo "ERROR: flappy-bird-env directory does not exist!"
    exit 1
fi

# Set random seed in environment (optional, if you want to override config.yaml)
export PYTHONHASHSEED=$seed

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory per CPU: 128M (Total: ~4GB)"
echo "Time limit: 4:00:00"
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

