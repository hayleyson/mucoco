#!/bin/bash
#SBATCH --job-name=mucola-gpt2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=40gb
#SBATCH --partition=cas_v100_2
#SBATCH --output='/scratch/x2600a03/mucoco/slurm_output/%j.out'
#SBATCH --comment python

start=$(date +%s)
echo "[start date] $(date)"
echo "$(pwd), $(hostname)"
echo 

# source /home/${USER}/.bashrc
# source ~/anaconda/etc/profile.d/conda.sh
# conda activate hson-mucola
# conda activate mucoco
module purge
module load python/3.7.1
source activate mucola

# srun bash examples/training_constraint_models/train_toxicity_classifier.sh  || echo "!!! error occurred"
# srun python -m notebooks.resume_run  || echo "!!! error occurred"
# srun bash examples/prompt/toxicity-all/mucola-disc.sh
# srun bash examples/prompt/toxicity-all/locate-edit-disc.sh
srun bash examples/prompt/toxicity-all/locate-edit-disc-for-testset-gpt2.sh
# srun bash examples/prompt/toxicity-all/mucola-disc-save-init-gen.sh

end=$(date +%s)

echo 
echo "[end date] $(date)"
echo "Elapsed Time: $((($end-$start)/60)) minutes $((($end-$start)%60)) seconds"