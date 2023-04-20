#!/bin/bash
#SBATCH --job-name=mucola_toxicity
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --mem=400gb
#SBATCH --cpus-per-task=8
#SBATCH --output='/home/hyeryungson/mucoco/slurm_output/%j.out'

pwd; hostname; date
start=$(date +%s)

source /home/${USER}/.bashrc
# source ~/anaconda/etc/profile.d/conda.sh
# conda activate mucola

# srun python perspective_api_call.py --dpath 'data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/toxicity_gte0.5.jsonl' --nsamples 10000 --outpath perspective_result_1_v3.csv
# srun python perspective_api_call.py --dpath 'data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/toxicity_eq0_subsample.jsonl' --nsamples 10000 --outpath perspective_result_0_v3.csv\
srun python perspective_api_call.py --dpath 'data/toxicity/jigsaw-unintended-bias-in-toxicity-classification/test_0.jsonl' --nsamples 2000 --outpath perspective_result_test_0.csv\
|| echo 'error occurred at'; date;

end=$(date +%s)
# elapsed time with second resolution
echo "Elapsed Time: $((($end-$start)/60)) minutes $((($end-$start)%60)) seconds"