# LENGTH=$1
# OUTPUTDIRSUFFIX=$2
LENGTH=20
# datevar=$(date +%d-%m-%Y)
MODELNAME=gpt2-large
data_source="gpt2"

NUM_LOG_STEPS=1
NUM_LOCATE_STEPS=-1 # if -1 then locate only once.
NUM_EDIT_TOKEN_PER_STEP=6
NUM_PROJECT_STEPS=2
NUM_LOG_STEPS=$NUM_PROJECT_STEPS

# OUTPUTDIR=outputs/toxicity/locate-edit-${data_source}-loc-${NUM_EDIT_TOKEN_PER_STEP}toks-${NUM_LOCATE_STEPS}steps-project-${NUM_PROJECT_STEPS}steps
# mkdir -p $OUTPUTDIR

BEGINNOISE=5.0
ENDNOISE=0.05
NOISESTEPS=150

POSITIVELABEL=1
NEGATIVELABEL=0

# EPSILONS=(-5 -2.2 -1.4)
EPSILONS=(-3) # the constraint is  log(negative_prob) - log (positive_prob) < epsilon, where positive_prob is the desired label

for EPSILON in "${EPSILONS[@]}"
do
    echo $data_source
    echo $EPSILON
    echo $NUM_PROJECT_STEPS
    NUM_LOG_STEPS=$NUM_PROJECT_STEPS
    OUTPUTDIR=outputs/toxicity/locate-edit-${data_source}-loc-${NUM_EDIT_TOKEN_PER_STEP}toks-${NUM_LOCATE_STEPS}steps-project-${NUM_PROJECT_STEPS}steps-mrr_allsat
    mkdir -p $OUTPUTDIR
    bash examples/prompt/constrained_sampling_locate_edit_for_testset.sh \
    nontoxic \
    $OUTPUTDIR \
    $MODELNAME \
    run_and_evaluate \
    0.25 \
    dotplusplus \
    l2 \
    0.0 \
    constant \
    legal \
    0 \
    1 \
    0.0 \
    false \
    1 \
    0.5 \
    0.05 \
    50 \
    target \
    100 \
    1.0 \
    constant \
    2 \
    50 \
    $EPSILON \
    false \
    false \
    0.45 \
    0.01 \
    0 \
    true \
    false \
    true \
    $data_source \
    $NUM_LOCATE_STEPS \
    $NUM_EDIT_TOKEN_PER_STEP\
    $NUM_PROJECT_STEPS\
    $NUM_LOG_STEPS
    #note: most arguments passed to constrained_sampling_mucola.sh are not used anymore
done


# sbatch sbatch.sh examples/prompt/constrained_sampling_new.sh nontoxic nontoxic/$datevar-nontoxic-$EPSILON $MODELNAME run_and_evaluate 0.25 dotplusplus l2 0.0 constant legal 0 1 0.0 false 1 0.5 0.05 50 target 100 1.0 constant 2 50 $EPSILON false false 0.45 0.01 0 true
