#!/bin/bash
#SBATCH --job-name=mucola-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mem=20gb
#SBATCH --partition=P2
#SBATCH --output='slurm_output/%j.out'

start=$(date +%s)
echo "[start date] $(date)"
echo "$(pwd), $(hostname)"
echo 

source ~/.bashrc
source ~/conda3/etc/profile.d/conda.sh
# conda activate hson-mucola
conda activate mucola

# srun bash examples/training_constraint_models/train_toxicity_classifier.sh  || echo "!!! error occurred"
# srun python -m notebooks.resume_run  || echo "!!! error occurred"
# srun bash examples/prompt/toxicity-all/mucola-disc.sh
# srun bash examples/prompt/toxicity-all/locate-edit-disc.sh
srun bash examples/prompt/toxicity-all/locate-edit-disc-for-testset_wo_project_re.sh
# srun bash examples/prompt/toxicity-all/mucola-disc-save-init-gen.sh
# srun bash examples/prompt/evaluate_only.sh

end=$(date +%s)

echo 
echo "[end date] $(date)"
echo "Elapsed Time: $((($end-$start)/60)) minutes $((($end-$start)%60)) seconds"