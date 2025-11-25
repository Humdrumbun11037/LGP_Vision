#!/bin/bash
# Helper script to set up the environment on the cluster
# Run this once after cloning/pulling the repo

echo "Setting up flappy-bird-env on cluster..."

# Initialize submodule if not already done
if [ ! -d "flappy-bird-env" ] || [ -z "$(ls -A flappy-bird-env 2>/dev/null)" ]; then
    echo "Initializing flappy-bird-env submodule..."
    git submodule update --init --recursive
fi

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: Not in a virtual environment. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install flappy-bird-env
echo "Installing flappy-bird-env..."
pip install -e ./flappy-bird-env

# Verify installation
echo "Verifying installation..."
python3 -c "import flappy_bird_env; print('✓ flappy_bird_env imported successfully')" || {
    echo "ERROR: Installation failed!"
    exit 1
}

echo "✓ Setup complete!"


