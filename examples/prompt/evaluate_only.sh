option="nontoxic"
OUTFILE=/home/s3/hyeryung/mucoco/outputs/toxicity/locate-unit/gpt2-word-netps-1-nls-1-nps1-modelenergy/outputs_epsilon-3.txt
EVALFILE=/home/s3/hyeryung/mucoco/outputs/toxicity/locate-unit/gpt2-word-netps-1-nls-1-nps1-modelenergy/results_epsilon-3.txt
EXTRAS=data/control-prompts/nontoxic_prompts-10k.jsonl
bash examples/prompt/evaluate.sh $option $OUTFILE $EVALFILE $EXTRAS $DATAFILE 