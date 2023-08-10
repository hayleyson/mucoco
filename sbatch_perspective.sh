#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --mem=10gb
#SBATCH --partition=skl
#SBATCH --output='/scratch/x2600a03/mucoco/slurm_output/%j.out'
#SBATCH --comment python

pwd; hostname; date
start=$(date +%s)

# source /home/${USER}/.bashrc
# source ~/anaconda/etc/profile.d/conda.sh
# conda activate mucola
module purge
module load python/3.7.1
source activate mucola

# srun python perspective_api_call.py --dpath 'data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/toxicity_gte0.5.jsonl' --nsamples 10000 --outpath perspective_result_1_v3.csv
# srun python perspective_api_call.py --dpath 'data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/toxicity_eq0_subsample.jsonl' --nsamples 10000 --outpath perspective_result_0_v3.csv\
srun python perspective_api_call.py --dpath '/scratch/x2600a03/mucoco/outputs/toxicity/save-init-gen-all-uniform/outputs.txt.init.cleaned' --text-col 'generation' --outpath /scratch/x2600a03/mucoco/outputs/toxicity/save-init-gen-all-uniform/outputs_cleaned_perspective_2.csv\
|| echo 'error occurred at'; date;

end=$(date +%s)
# elapsed time with second resolution
echo "Elapsed Time: $((($end-$start)/60)) minutes $((($end-$start)%60)) seconds"