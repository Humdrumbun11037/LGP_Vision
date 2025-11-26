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

# Load required modules BEFORE creating virtualenv
# See: https://docs.alliancecan.ca/wiki/OpenCV
module load gcc/12.3
module load python/3.10
module load opencv/4.8.1

# Create virtual environment with system site-packages (for opencv access)
virtualenv --no-download --system-site-packages $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Upgrade pip
pip install --no-index --upgrade pip

# Install packages (excluding opencv-python since it's provided by module)
# Also exclude flappy-bird-env since it's in the repo
grep -v "flappy-bird-env\|opencv" requirements.txt > $SLURM_TMPDIR/requirements_wheelhouse.txt
pip install --no-index -r $SLURM_TMPDIR/requirements_wheelhouse.txt

# Verify cv2 is available from module
python -c "import cv2; print(f'✓ cv2 version: {cv2.__version__}')" || {
    echo "ERROR: cv2 not available!"
    exit 1
}

# Verify flappy_bird_env package exists (it's now part of the repo)
if [ ! -d "flappy_bird_env" ]; then
    echo "ERROR: flappy_bird_env package not found!"
    echo "The flappy_bird_env directory should be in the project root"
    exit 1
fi

# Verify import works
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
