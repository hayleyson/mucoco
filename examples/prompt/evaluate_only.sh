option="nontoxic"
OUTFILE=/home/s3/hyeryung/mucoco/outputs/toxicity/locate-edit-gpt2-loc--1toks--1steps-project-1steps-mrr_allsat-modeltype-energy/outputs_epsilon-5.txt
EVALFILE=results_epsilon-5.txt
EXTRAS=data/control-prompts/nontoxic_prompts-10k.jsonl
bash examples/prompt/evaluate.sh $option $OUTFILE $EVALFILE $EXTRAS $DATAFILE 