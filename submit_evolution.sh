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
#SBATCH --mail-user=hillroyx@mcmaster.ca
#SBATCH --mail-type=ALL

# Change to the directory where sbatch was run from
cd "$SLURM_SUBMIT_DIR" || {
    echo "ERROR: Failed to change to submission directory: $SLURM_SUBMIT_DIR"
    exit 1
}

seed=${1:-42}

# Load Python module
module load python/3.10

# Create virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Upgrade pip
pip install --no-index --upgrade pip

# Install packages (excluding flappy-bird-env)
grep -v "flappy-bird-env" requirements.txt > $SLURM_TMPDIR/requirements_wheelhouse.txt
pip install --no-index -r $SLURM_TMPDIR/requirements_wheelhouse.txt

# Verify submodule exists (should be initialized before submitting)
if [ ! -d "flappy-bird-env/flappy_bird_env" ]; then
    echo "ERROR: flappy-bird-env submodule not found!"
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi

# Verify import works (using direct import from submodule)
python -c "import flappy_bird_env; print('✓ flappy_bird_env imported successfully')" || {
    echo "ERROR: flappy_bird_env import failed!"
    exit 1
}

# Set random seed
export PYTHONHASHSEED=$seed

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo "Random seed: $seed"
echo "=========================================="

# Run evolution
python run_flappy_bird.py config.yaml

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
