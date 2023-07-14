#!/bin/bash
#SBATCH --job-name=mucola
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=80gb
#SBATCH --cpus-per-task=8
#SBATCH --output='./slurm_output/%j.out'

start=$(date +%s)
echo "[start date] $(date)"
echo "$(pwd), $(hostname)"
echo 

source /home/${USER}/.bashrc
source ~/anaconda/etc/profile.d/conda.sh
conda activate hson-mucola

# srun bash examples/training_constraint_models/train_toxicity_classifier.sh  || echo "!!! error occurred"
# srun python -m notebooks.resume_run  || echo "!!! error occurred"
srun bash examples/prompt/toxicity-all/mucola-disc.sh

end=$(date +%s)

echo 
echo "[end date] $(date)"
echo "Elapsed Time: $((($end-$start)/60)) minutes $((($end-$start)%60)) seconds"