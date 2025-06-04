#!/usr/bin/env bash

echo "############################################################"
echo "#           ScaleX Automatic Installation Script           #"
echo "############################################################"
echo ""
echo "This script will guide you through setting up the ScaleX environment."
echo "Please ensure Anaconda or Miniconda is installed."
echo ""

# --- Check if requirements.txt exists ---
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found in the current directory."
    echo "Please run this script from the root of the ScaleX project folder."
    exit 1
fi

# --- Conda Environment Name and Python Version ---
ENV_NAME="ScaleX"
PYTHON_VERSION="3.12"

# --- Attempt to initialize Conda for this script session if not already done ---
# This helps if `conda activate` doesn't work directly in scripts.
_CONDA_PROFILE_DIR_USER="$HOME/anaconda3"
_CONDA_PROFILE_DIR_MINI_USER="$HOME/miniconda3"
_CONDA_PROFILE_PATH=""

if [ -f "$_CONDA_PROFILE_DIR_USER/etc/profile.d/conda.sh" ]; then
    _CONDA_PROFILE_PATH="$_CONDA_PROFILE_DIR_USER/etc/profile.d/conda.sh"
elif [ -f "$_CONDA_PROFILE_DIR_MINI_USER/etc/profile.d/conda.sh" ]; then
    _CONDA_PROFILE_PATH="$_CONDA_PROFILE_DIR_MINI_USER/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then # Common system-wide install path
    _CONDA_PROFILE_PATH="/opt/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    _CONDA_PROFILE_PATH="/opt/miniconda3/etc/profile.d/conda.sh"
fi

if [ -n "$_CONDA_PROFILE_PATH" ]; then
    # shellcheck source=/dev/null
    source "$_CONDA_PROFILE_PATH"
else
    echo "WARNING: Could not find conda.sh to source. 'conda activate' might fail."
    echo "If you encounter issues, try running 'conda init bash' (or your shell) in your terminal,"
    echo "then close and reopen your terminal before running this script again."
    echo "Alternatively, ensure your Conda installation's 'bin' directory is in your PATH."
fi
echo ""


# --- Create Conda Environment ---
echo "Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create the Conda environment."
    echo "It might already exist, or there was an issue with your Conda installation."
    echo "If it already exists, you might want to remove it first with: conda env remove -n $ENV_NAME"
    exit 1
fi
echo "Environment $ENV_NAME created successfully."
echo ""

# --- Activate Environment and Install PyTorch ---
echo "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate the Conda environment $ENV_NAME."
    echo "Make sure Conda is initialized for your shell (e.g., run 'conda init bash')."
    exit 1
fi
echo "Environment $ENV_NAME activated."
echo ""

echo "--- PyTorch Installation ---"
echo "Choose PyTorch installation type:"
echo "1. CPU only (Recommended if you don't have a compatible NVIDIA GPU or are on macOS without an Apple Silicon GPU for MPS)"
echo "2. GPU (NVIDIA CUDA or Apple MPS - You will need to provide the correct command)"
echo ""
read -r -p "Enter your choice (1 or 2): " PYTORCH_CHOICE

if [ "$PYTORCH_CHOICE" = "1" ]; then
    echo "Installing PyTorch for CPU..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install PyTorch (CPU). Please check your internet connection or Conda setup."
        exit 1
    fi
elif [ "$PYTORCH_CHOICE" = "2" ]; then
    echo "To install PyTorch for GPU (NVIDIA CUDA or Apple MPS):"
    echo "1. Visit the PyTorch official website: https://pytorch.org/get-started/locally/"
    echo "2. Select your OS, Package (Conda), Language (Python), and Compute Platform (CUDA version or MPS)."
    echo "3. Copy the generated 'conda install ...' command."
    echo ""
    echo "Example for CUDA 11.8: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"
    echo "Example for CUDA 12.1: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y"
    echo "Example for Apple MPS: conda install pytorch torchvision torchaudio -c pytorch -y (PyTorch usually auto-detects MPS)"
    echo ""
    read -r -p "Paste the PyTorch GPU/MPS install command here and press Enter: " PYTORCH_GPU_COMMAND
    if [ -n "$PYTORCH_GPU_COMMAND" ]; then
        echo "Installing PyTorch for GPU/MPS with your command:"
        echo "$PYTORCH_GPU_COMMAND"
        eval "$PYTORCH_GPU_COMMAND" # Use eval to execute the pasted command
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install PyTorch (GPU/MPS) using your command."
            echo "Please ensure the command was correct and your system setup (CUDA/Drivers/MPS) is complete."
            exit 1
        fi
    else
        echo "No command entered. Skipping GPU/MPS PyTorch installation. You will need to install it manually."
    fi
else
    echo "Invalid choice. Skipping PyTorch installation. You will need to install it manually."
fi
echo ""

# --- Install other requirements ---
echo "Installing other dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install packages from requirements.txt."
    echo "Please check the error messages above."
    exit 1
fi
echo "Dependencies installed successfully."
echo ""

echo "############################################################"
echo "#              Installation Complete!                      #"
echo "############################################################"
echo ""
echo "To use ScaleX:"
echo "1. Open a new terminal."
echo "2. Activate the environment: conda activate $ENV_NAME"
echo "3. Navigate to the ScaleX directory."
echo "4. Run: python inference_scalex.py --help"
echo ""
exit 0