# VENV_NAME=$1
# conda create -n $VENV_NAME python=3.11
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate $VENV_NAME
conda install pytorch==1.11 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
# conda deactivate $VENV_NAME