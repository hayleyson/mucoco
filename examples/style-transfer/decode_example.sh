####
# How to use this file:
# Run it as bash decode.sh <path to folder with data to decode> <path to folder with output to be saved> <which option for constraint (see below)> <debug mode/run and evaluate/only evaluate>
# path to data folder should point to informal file and it's paraphrase generated by Krishna et al 2020 (https://github.com/martiansideofthemoon/style-transfer-paraphrase), denoted by informal.paraphrase, and optionally formal.ref0 (this is useful for debugging/evaluation)

# activate the virtual env
# if [ -z "${5}" ]
# then
#     source activate 2022
# else
#     source activate $5
# fi 

DATA_DIR=$1
outdir=$2
option=$3
if [[ -z "$option" ]]
then 
    option="all"
fi

mkdir -p $outdir
OUTDIR=$outdir

PRIMARYMODEL=path/to/primary/model
STSMODEL=path/to/usim/model
CLASSIFIERMODEL=path/to/formality/classifier

# PRIMARYMODEL=../../related-work/style-transfer-paraphrase/style_paraphrase/saved_models/gpt2-entertainment_music-formality_0_eos/best-checkpoint/
# STSMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-nli-stsb-mean-tokens/0_Transformer
# CLASSIFIERMODEL=/projects/tir5/users/sachink/embed-style-transfer/models/gpt2-formality-classifier/checkpoint_best/

if [[ "$option" == "plain" ]]
then 
    echo "plain"
    model=$PRIMARYMODEL
    tokenizer=$PRIMARYMODEL
    model_types=AutoModelForCausalLM
    betas=1.0
    loss=gpt2conditional
    lossabbr="logpyx"
    epsilons=none
    min_epsilons=none
    epsilon_warmup_steps=none
    epsilon_cooldown_steps=none
    epsilon_decay_functions=none
elif [[ "$option" == "usim" ]]
then
    echo "usim"
    model=$PRIMARYMODEL:$STSMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL
    model_types=AutoModelForCausalLM:AutoModel
    betas=0.8:0.2
    loss=gpt2conditional:usim
    lossabbr="logpyx:usim"
    epsilons=0.4
    min_epsilons=0.25
    epsilon_warmup_steps=20
    epsilon_cooldown_steps=40
    epsilon_decay_functions=step
elif [[ "$option" == "usim:wmd" ]]
then
    echo "usim:wmd"
    model=$PRIMARYMODEL:$STSMODEL:$STSMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL:$STSMODEL
    model_types=AutoModelForCausalLM:AutoModel:AutoModel
    betas=0.8:0.1:0.1
    loss=gpt2conditional:usim:wmd
    lossabbr="logp(y|x):usim:wmd"
    epsilons=1.0:1.0
    min_epsilons=0.25:0.4
    epsilon_warmup_steps=20:20
    epsilon_cooldown_steps=40:40
    epsilon_decay_functions=step:step
elif [[ "$option" == "wmd" ]]
then
    echo 2
    model=$PRIMARYMODEL:$STSMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL
    model_types=AutoModelForCausalLM:AutoModel
    betas=1.0:0.0
    loss=gpt2conditional:wmd
    lossabbr="logp(y|x):wmd"
    epsilons=0.8
    min_epsilons=0.4
    epsilon_warmup_steps=20
    epsilon_cooldown_steps=0:0
    epsilon_decay_functions=step:step
elif [[ "$option" == "classification" ]]
then
    echo "else"
    model=$PRIMARYMODEL:$CLASSIFIERMODEL
    tokenizer=$PRIMARYMODEL:$CLASSIFIERMODEL
    model_types=AutoModelForCausalLM:AutoModelForSequenceClassification
    betas=0.8:0.2
    loss=gpt2conditional:classification
    label_id=1
    lossabbr="pyx:binary"
    epsilons=10.0
    min_epsilons=0.69
    epsilon_warmup_steps=40
    epsilon_cooldown_steps=80
    epsilon_decay_functions=step
elif [[ "$option" == "usim_classify" ]]
then
    echo "usim,wmd,classify"
    model=$PRIMARYMODEL:$STSMODEL:$STSMODEL:$CLASSIFIERMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL:$STSMODEL:$CLASSIFIERMODEL
    model_types=AutoModelForCausalLM:AutoModel:AutoModel:AutoModelForSequenceClassification
    betas=0.7:0.1:0.1:0.1
    loss=gpt2conditional:usim:wmd:classification
    lossabbr="pyx:sts:wmd:binary"
    epsilons=1.0:1.0:10.0
    min_epsilons=0.25:0.4:0.69
    epsilon_warmup_steps=40:40:40
    epsilon_cooldown_steps=80:80:80
    epsilon_decay_functions=step:step:step
else
    echo "usim,classify"
    model=$PRIMARYMODEL:$STSMODEL:$CLASSIFIERMODEL
    tokenizer=$PRIMARYMODEL:$STSMODEL:$CLASSIFIERMODEL
    model_types=AutoModelForCausalLM:AutoModel:AutoModelForSequenceClassification
    betas=0.8:0.1:0.1
    loss=gpt2conditional:usim:classification
    lossabbr="pyx:sts:binary"
    epsilons=1.0:10.0
    min_epsilons=0.25:0.69
    epsilon_warmup_steps=40:40
    epsilon_cooldown_steps=80:80
    epsilon_decay_functions=step:step
fi


debug=$4
if [[ -z "$debug" ]]
then 
    debug="only_evaluate"
fi 

selection_criterion="primary_allsat"
always_mucoco="false"
lr=50

if [[ "$debug" == "debug" ]]
then
    python -W ignore -u decode.py\
        --data $DATA_DIR/informal:$DATA_DIR/formal.ref0\
        --additional-data $DATA_DIR/informal.paraphrase\
        --model $model\
        --tokenizer $tokenizer\
        --model_types $model_types\
        --betas $betas\
        --loss $loss\
        --lossabbr "$lossabbr"\
        --prefix-length 0\
        --model_dtype fp32\
        --target-type probs\
        --always-mucoco $always_mucoco\
        --max-grad-norm 0.5\
        --optim-steps 200\
        --log-interval 10\
        --bos\
        --eos\
        --st\
        --selection_criterion $selection_criterion\
        --length_diff 0\
        --optim expgd\
        --expgd_mw 2\
        --lr $lr\
        --length-normalize\
        --dampness 1.0\
        --epsilons $epsilons\
        --min_epsilons $min_epsilons\
        --epsilon_warmup_steps $epsilon_warmup_steps\
        --epsilon_cooldown_steps $epsilon_cooldown_steps\
        --epsilon_decay_functions $epsilon_decay_functions\
        --lambda-lr 2.5\
        --num-examples 10\
        --debug
elif [[ "$debug" == "run_and_evaluate" ]]
then
    python -W ignore -u decode.py\
        --data $DATA_DIR/informal:$DATA_DIR/formal.ref0\
        --additional-data $DATA_DIR/informal.paraphrase\
        --model $model\
        --tokenizer $tokenizer\
        --model_types $model_types\
        --betas $betas\
        --loss $loss\
        --lossabbr "$lossabbr"\
        --prefix-length 0\
        --model_dtype fp32\
        --target-type probs\
        --always-mucoco $always_mucoco\
        --max-grad-norm 0.05\
        --optim-steps 100\
        --log-interval 50\
        --bos\
        --eos\
        --st\
        --selection_criterion $selection_criterion\
        --length_diff 0\
        --optim expgd\
        --expgd_mw 2\
        --lr $lr\
        --length-normalize\
        --dampness 1.0\
        --epsilons $epsilons\
        --min_epsilons $min_epsilons\
        --epsilon_warmup_steps $epsilon_warmup_steps\
        --epsilon_cooldown_steps $epsilon_cooldown_steps\
        --epsilon_decay_functions $epsilon_decay_functions\
        --lambda-lr 2.5\
        --num-examples 0\
        --outfile $OUTDIR/prediction.txt
    
    bash ./examples/style-transfer/evaluate.sh $DATA_DIR $OUTDIR/prediction.txt
else
    bash ./examples/style-transfer/evaluate.sh $DATA_DIR $OUTDIR/prediction.txt
fi
