#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=mucola_clean
#SBATCH --output='laser_edit/_slurm_outs/mucola_clean_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/home/hyeryung/data/
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

# srun python decode_new_clean.py --argument_file_path examples/prompt/toxicity-all/arguments_below_nontoxic_threshold_468.txt
# srun python decode_new_clean.py --argument_file_path examples/prompt/sentiment-all/arguments_below_positive_threshold_778.txt
# srun python decode_new_clean.py --argument_file_path examples/prompt/sentiment-all/arguments_below_negative_threshold_827.txt
# srun python decode_new_clean.py --argument_file_path examples/prompt/toxicity-all/arguments_below_nontoxic_threshold_468_epsilon-5.txt
# srun python decode_new_clean.py --argument_file_path examples/prompt/sentiment-all/arguments_below_positive_threshold_778_epsilon-2.txt
# srun python decode_new_clean.py --argument_file_path examples/prompt/sentiment-all/arguments_below_negative_threshold_827_epsilon-2.txt
# srun python decode_new_clean.py --argument_file_path examples/prompt/formality-all/arguments_formal2informal.txt
# srun python decode_new_clean.py --argument_file_path examples/prompt/formality-all/arguments_informal2formal.txt
# srun python decode_new_clean.py --argument_file_path examples/prompt/formality-all/arguments_formal2informal_epsilon-2.txt
# srun python decode_new_clean.py --argument_file_path examples/prompt/formality-all/arguments_informal2formal-epsilon-2.txt
srun -n 1 -c 1python decode_new_clean.py --argument_file_path examples/prompt/toxicity-all/arguments_gpt35_gpt2_clsf.txt
# srun python decode_new_clean.py --argument_file_path examples/prompt/sentiment-all/arguments_below_positive_threshold_778_random_init.txt
# srun python decode_new_clean.py --argument_file_path examples/prompt/sentiment-all/arguments_below_positive_threshold_778_epsilon-2_random_init.txt



