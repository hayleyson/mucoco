#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:0
#SBATCH --job-name=gen_gpt4_toxic
#SBATCH --output='new_module/_slurm_outs/gen_gpt_toxic_%j.out'

source /home/${USER}/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

DATA_DIR=/home/hyeryung/data
export PYTHONPATH=.
export HF_HOME=$DATA_DIR/hf_cache
export HF_DATASETS_CACHE=$DATA_DIR/hf_cache
export TRANSFORMERS_CACHE=$DATA_DIR/hf_cache
export LOGGING_LEVEL=INFO

# srun -n 1 -c 1 python /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/gpt_api_generate.py \
# --model "gpt-3.5-turbo-0125" \
# --openai_api_key $OPENAI_API_KEY \
# --file_save_path /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_1k_prompts_0shot_150.jsonl \
# --input_file_path /home/hyeryung/data/mucoco/data/control-prompts/nontoxic_prompts-1k.jsonl \
# --prompt_type nontoxic_0shot \
# --max_tokens 150 \
# --num_test_prompts -1 \
# --num_return_sequences 10

srun -n 1 -c 1 python /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/gpt_api_generate_no_batch.py \
--model "gpt-3.5-turbo-0125" \
--openai_api_key $OPENAI_API_KEY \
--file_save_path /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nontoxic/gpt-3.5-turbo-0125_realtoxicityprompts_0shot_150_for_time_no_batch.jsonl \
--input_file_path /home/hyeryung/data/mucoco/new_module/data/toxicity-avoidance/dev_set.jsonl \
--prompt_type nontoxic_0shot \
--max_tokens 150 \
--num_test_prompts -1 \
--num_return_sequences 10

# srun python /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/gpt_api_generate.py \
# --model "gpt-3.5-turbo-0125" \
# --openai_api_key $OPENAI_API_KEY \
# --file_save_path /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/sentiment/gpt-3.5-turbo-0125_pplm_prompts_noprompt_150.jsonl \
# --input_file_path /home/hyeryung/data/mucoco/new_module/data/sentiment/outputs.txt.init.jsonl \
# --prompt_type nan \
# --max_tokens 150 \
# --num_test_prompts -1 \
# --num_return_sequences 20

# srun python /home/hyeryung/data/mucoco/new_module/llm_experiments/gpt_api_generate.py \
# --openai_api_key $OPENAI_API_KEY \
# --file_save_path /home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_prompting_gens_formal_0shot.jsonl \
# --input_file_path /home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal \
# --prompt_type formal_0shot \
# --max_tokens 60 \
# --num_test_prompts -1 \
# --num_return_sequences 1

# srun python /home/hyeryung/data/mucoco/new_module/llm_experiments/gpt_api_generate.py \
# --openai_api_key $OPENAI_API_KEY \
# --file_save_path /home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_prompting_gens_formal_3shot.jsonl \
# --input_file_path /home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/informal \
# --prompt_type formal_3shot \
# --max_tokens 60 \
# --num_test_prompts -1 \
# --num_return_sequences 1

# srun python /home/hyeryung/data/mucoco/new_module/llm_experiments/gpt_api_generate.py \
# --openai_api_key $OPENAI_API_KEY \
# --file_save_path /home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_prompting_gens_informal_0shot_ungrammar.jsonl \
# --input_file_path /home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal \
# --prompt_type informal_0shot_ungrammar \
# --max_tokens 60 \
# --num_test_prompts -1 \
# --num_return_sequences 1

# srun python /home/hyeryung/data/mucoco/new_module/llm_experiments/gpt_api_generate.py \
# --openai_api_key $OPENAI_API_KEY \
# --file_save_path /home/hyeryung/data/mucoco/new_module/llm_experiments/baselm_gens/gpt4o_prompting_gens_informal_3shot_ungrammar.jsonl \
# --input_file_path /home/hyeryung/data/mucoco/data/formality/GYAFC_Corpus/Entertainment_Music/test/formal \
# --prompt_type informal_3shot_ungrammar \
# --max_tokens 60 \
# --num_test_prompts -1 \
# --num_return_sequences 1

# srun -n 1 -c 1 python /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/gpt_api_generate_no_batch.py \
# --model "gpt-3.5-turbo-0125" \
# --openai_api_key $OPENAI_API_KEY \
# --file_save_path /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nli/gpt-3.5-turbo-0125_anli-r2-test_prompt_4_150_for_time_no_batch.jsonl \
# --input_file_path new_module/data/logical-consistency/anli-r2-test_prompt_4_prompts.jsonl \
# --prompt_type nli_0shot \
# --max_tokens 150 \
# --num_test_prompts 1 \
# --num_return_sequences 10

# srun -n 1 -c 1 python /home/hyeryung/data/mucoco/new_module/llm_experiments/generate_with_llm/gpt_api_generate.py \
# --model "gpt-3.5-turbo-0125" \
# --openai_api_key $OPENAI_API_KEY \
# --file_save_path new_module/llm_experiments/generate_with_llm/baselm_gens/gpt-3.5-turbo-0125/nli_ifeval/gpt-3.5-turbo-0125_nli_ifeval_150.jsonl \
# --input_file_path new_module/data/logical-consistency/anli-r2-test_prompt_4_prompts.jsonl \
# --prompt_type nli_ifeval \
# --max_tokens 150 \
# --num_test_prompts 720 \
# --num_return_sequences 10

