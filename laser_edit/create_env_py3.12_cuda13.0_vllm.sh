#!/bin/bash
# Installation script for loc-edit environment with Python 3.12, CUDA 13.0 PyTorch, and vLLM.
#
# Install order matters: install torch + vLLM together from the CUDA 13.0 index so
# vLLM does not replace a separately installed torch build with an incompatible one.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="loc-edit-pro6000-vllm"
REQ_FILE="${SCRIPT_DIR}/requirements_py3.12_cuda13.0_vllm.txt"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

echo "=========================================="
echo "Creating ${ENV_NAME} with Python 3.12, CUDA 13.0, and vLLM"
echo "=========================================="

# Create conda environment
conda create -n "${ENV_NAME}" python=3.12 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Upgrading pip..."
pip3 install --upgrade pip

echo ""
echo "=========================================="
echo "Installing PyTorch + vLLM with CUDA 13.0 support"
echo "=========================================="
# Install torch stack and vLLM in one step against the cu130 index.
# vLLM pins exact torch/torchvision/torchaudio versions; torchaudio is required by vLLM.
pip3 install torch torchvision torchaudio vllm --extra-index-url "${TORCH_INDEX}"

echo ""
echo "=========================================="
echo "Installing remaining packages"
echo "=========================================="
# Remaining app/eval packages (torch / vLLM already installed above)
pip3 install -r "${REQ_FILE}"

echo ""
echo "=========================================="
echo "Verifying PyTorch / vLLM installation"
echo "=========================================="
python -c "
import torch
import vllm
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
print(f'vLLM version: {vllm.__version__}')
"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
