option="nontoxic"
OUTFILE=/home/hyeryung_son/mucoco/outputs/toxicity/locate-edit-gpt2-loc--1toks--1steps-project-1steps-mrr_allsat-modeltype-mucola/outputs_epsilon-3.txt
EVALFILE=results_epsilon-3.txt
EXTRAS=data/control-prompts/nontoxic_prompts-10k.jsonl
bash examples/prompt/evaluate.sh $option $OUTFILE $EVALFILE $EXTRAS $DATAFILE 