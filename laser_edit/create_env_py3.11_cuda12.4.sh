#!/bin/bash
# Installation script for loc-edit environment with Python 3.11 and CUDA-enabled PyTorch

set -e  # Exit on error

echo "=========================================="
echo "Creating loc-edit environment with Python 3.11 and CUDA 12.4"
echo "=========================================="

# Create conda environment
conda create -n loc-edit python=3.11 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate loc-edit

echo ""
echo "=========================================="
echo "Installing PyTorch with CUDA 12.4 support"
echo "=========================================="
# Install PyTorch with CUDA 12.4 support first
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "=========================================="
echo "Installing remaining packages"
echo "=========================================="
# Install all other packages (excluding PyTorch packages)
pip install -r requirements_py3.11_cuda12.4.txt

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
echo "  conda activate loc-edit"

