#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=P2
#SBATCH --gres=gpu:1
#SBATCH --output='new_module/_slurm_outs/s_gbi_em_gn_%j.out'

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

export PYTHONPATH=.
export LOGGING_LEVEL="INFO"
export HF_HOME=/shared/s3/lab07/hyeryung/hf_cache
export HF_DATASETS_CACHE=/shared/s3/lab07/hyeryung/hf_cache
export TRANSFORMERS_CACHE=/shared/s3/lab07/hyeryung/hf_cache

srun python new_module/decode_new_for_testset_iter.py \
 --AR-temperature=1.0\
 --AR-top-k=0\
 --AR-top-p=0.96\
 --adam-betas='(0.9, 0.999)'\
 --adam-eps=1e-08\
 --additional-data='none'\
 --allow-diff-vocab\
 --always-mucoco='false'\
 --baselm-gen-online\
 --batch-size=1\
 --beam-size=1\
 --betas='0.8:0.2'\
 --bos\
 --coeff-pattern='constant'\
 --coeff-steps=200\
 --custom-epsilons='none'\
 --dampness=0.1\
 --data='new_module/data/sentiment/outputs.txt.init.jsonl'\
 --datastyle='jsonl'\
 --debug-gradients='false'\
 --decay-steps=1\
 --decode-temperature=0.1\
 --dynamic-lambda-update\
 --dynamic-lr-update\
 --early-stop-steps=4\
 --embedgd-begin-temperature=0.5\
 --embedgd-do-sample='false'\
 --embedgd-final-temperature=0.05\
 --embedgd-grad-distance='l2'\
 --embedgd-gumbel-noise-max=0.0\
 --embedgd-lr-pattern='constant'\
 --embedgd-momentum=0.0\
 --embedgd-noise-variance=1.0\
 --embedgd-temperature-reduction-steps=50\
 --embedgd-top-k=0\
 --embedgd-top-p=1.0\
 --end-idx=-1\
 --eos\
 --epsilon=0.2\
 --epsilon_cooldown_steps='1'\
 --epsilon_decay_functions='linear'\
 --epsilon_warmup_steps='0'\
 --epsilons='-3'\
 --evaluation_metrics='fluency'\
 --expgd-gumbel-noise-max=0.0\
 --expgd-momentum=0.0\
 --expgd_mw=1\
 --expgd-top-k=0\
 --expgd-top-p=1.0\
 --fp16_source='pytorch'\
 --gold-loss-epsilons='none'\
 --init='target'\
 --jsonl-primary-key='prompt'\
 --jsonl-secondary-key='text'\
 --jsonl-tokenized='false'\
 --keyword_tau=2.0\
 --keyword_topk=1\
 --keywords='none'\
 --kweight=5.0\
 --label-id='1:1'\
 --lambda-lr=1.0\
 --lambda-update=50\
 --length-diff='0'\
 --linear-scale='false'\
 --locate-edit\
 --log-interval=25\
 --loss='gpt2:classification_no_prefix'\
 --loss-type='dotplusplus'\
 --lossabbr='pyx:sentiment'\
 --lr=0.25\
 --lr-decay=1.0\
 --lr-update-size=0.01\
 --match_with='reference'\
 --max-allowed-length=200\
 --max-grad-norm=0.0\
 --max-length=20\
 --max-lr=0.45\
 --max-output-length=20\
 --max-prefix-length=50\
 --metric='l2'\
 --min_epsilons='-3'\
 --model='gpt2-large:/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint'\
 --model_dtype='fp32'\
 --model_types='AutoModelForCausalLM:RobertaCustomForSequenceClassification'\
 --num_edit_token_per_step=3\
 --num-examples=100\
 --num_locate_steps=1\
 --num_log_steps=5\
 --num_project_steps=1\
 --num_samples=20\
 --only-mucoco='false'\
 --optim='embedgd_le'\
 --optim-steps=20\
 --output_dir_prefix='outputs/sentiment/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training'\
 --output-style='jsonl'\
 --prefix-length=0\
 --random-example='true'\
 --repetition-penalty=0.0\
 --restarts=0\
 --same-embeds\
 --sampling-strategy='greedy'\
 --sampling-strategy-k='none'\
 --scale_loss='none'\
 --seed=42\
 --selection_criterion='allsat'\
 --semantic_methods='bertscore'\
 --sgd-momentum=0.0\
 --start-decay-steps=1\
 --start-idx=0\
 --suffix-length=0\
 --target-type='embeds'\
 --tokenizer='gpt2-large:/shared/s3/lab07/hyeryung/loc_edit/roberta-base-yelp-sentiment-classifier-with-gpt2-large-embeds-energy-training/step_44900_best_checkpoint'\
 --topic-target='none'\
 --topic-word-lists='none'\
 --use_context='false'\
 --warmup-steps=1\
 --weight-decay=0.0\
 --task_type='prompted_generation'\
 --locate_unit 'word'\
 --wandb_project 'sentiment_gbi'\
 --wandb_entity 'hayleyson'\
 --source_style 'negative'\
 --target_style 'positive'\
 --locate_method='grad_norm'