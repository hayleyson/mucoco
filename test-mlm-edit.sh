#!/bin/bash
#SBATCH --job-name=mlm-test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=50gb
#SBATCH --partition=P2
#SBATCH --output='slurm_output/%j.out'

start=$(date +%s)
echo "[start date] $(date)"
echo "$(pwd), $(hostname)"
echo 

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python /home/s3/hyeryung/mucoco/ell-e/locate-edit-by-LM.py
# srun bash /home/s3/hyeryung/mucoco/examples/prompt/evaluate_only.sh

end=$(date +%s)
echo "[end date] $(date)"
echo "Elapsed Time: $((($end-$start)/60)) minutes $((($end-$start)%60)) seconds"