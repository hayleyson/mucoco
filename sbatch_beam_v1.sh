#!/bin/bash
#SBATCH --job-name=mucola-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=20gb
#SBATCH --partition=P2
#SBATCH --output='slurm_output/%j.out'

start=$(date +%s)
echo "[start date] $(date)"
echo "$(pwd), $(hostname)"
echo 

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

srun python /home/s3/hyeryung/mucoco/new_module/mlm_reranking_beam_search_variant1.py

end=$(date +%s)

echo 
echo "[end date] $(date)"
echo "Elapsed Time: $((($end-$start)/60)) minutes $((($end-$start)%60)) seconds"