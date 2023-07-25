option="nontoxic"
OUTFILE=outputs/toxicity/locate-edit2/outputs.txt
EVALFILE=results1.txt
EXTRAS=data/control-prompts/nontoxic_prompts-10k.jsonl
bash examples/prompt/evaluate.sh $option $OUTFILE $EVALFILE $EXTRAS $DATAFILE 