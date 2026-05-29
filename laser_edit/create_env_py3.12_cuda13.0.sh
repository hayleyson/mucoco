#!/bin/bash
# Installation script for loc-edit environment with Python 3.11 and CUDA-enabled PyTorch

set -e  # Exit on error

echo "=========================================="
echo "Creating loc-edit environment with Python 3.12 and CUDA 13.0"
echo "=========================================="

# Create conda environment
conda create -n loc-edit-pro6000 python=3.12 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate loc-edit-pro6000

echo ""
echo "=========================================="
echo "Installing PyTorch with CUDA 13.0 support"
echo "=========================================="
# Install PyTorch with CUDA 13.0 support first
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo "Upgrading pip..."
pip3 install --upgrade pip

echo ""
echo "=========================================="
echo "Installing remaining packages"
echo "=========================================="
# Install all other packages (excluding PyTorch packages)
pip install -r requirements_py3.12_cuda13.0.txt

echo ""
echo "=========================================="
echo "Verifying PyTorch CUDA installation"
echo "=========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo "To activate the environment, run:"
echo "  conda activate loc-edit-pro6000"

