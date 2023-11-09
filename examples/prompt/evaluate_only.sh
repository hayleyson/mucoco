option="nontoxic"
OUTFILE=/home/s3/hyeryung/mucoco/outputs/toxicity/mlm-reranking/mlm-token-nps4-k3-weighted_sum-0.5-0.5/outputs_epsilon-3-test.txt
EVALFILE=/home/s3/hyeryung/mucoco/outputs/toxicity/mlm-reranking/mlm-token-nps4-k3-weighted_sum-0.5-0.5/results_epsilon-3-test.txt
EXTRAS=data/control-prompts/nontoxic_prompts-10k.jsonl
bash examples/prompt/evaluate.sh $option $OUTFILE $EVALFILE $EXTRAS $DATAFILE 