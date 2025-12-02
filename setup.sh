#!/bin/bash

# 1. Configuration
# Set Hugging Face cache to scratch to avoid filling up home directory
export HF_HOME="/work/scratch/$USER"
VENV_NAME="venv"

echo "========================================"
echo "      Environment Setup Script"
echo "========================================"
echo ">> HF_HOME configured to: $HF_HOME"

# 2. Check/Create Virtual Environment
if [ ! -d "$VENV_NAME" ]; then
    echo ">> Virtual environment '$VENV_NAME' not found. Creating it..."
    
    # Check if python3 exists
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 could not be found. Please load the python module (e.g., 'module load python')."
        exit 1
    fi
    
    python3 -m venv "$VENV_NAME"
    echo ">> Virtual environment created successfully."
else
    echo ">> Virtual environment '$VENV_NAME' already exists. Skipping creation."
fi

# 3. Activate Virtual Environment
echo ">> Activating '$VENV_NAME'..."
source "$VENV_NAME/bin/activate"

# 4. Install Dependencies
if [ -f "requirements.txt" ]; then
    echo ">> Found requirements.txt. Installing dependencies..."
    
    # Upgrade pip first to avoid errors with newer wheels
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    echo ">> Dependency installation complete."
else
    echo ">> Warning: 'requirements.txt' not found in current directory."
    echo ">> Skipping dependency installation."
fi

echo "========================================"
echo ">> Setup Finished!"
echo "========================================"