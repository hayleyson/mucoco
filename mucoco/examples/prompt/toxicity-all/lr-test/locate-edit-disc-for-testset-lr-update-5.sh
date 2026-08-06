# LENGTH=$1
# OUTPUTDIRSUFFIX=$2
LENGTH=20
# datevar=$(date +%d-%m-%Y)
MODELNAME=gpt2-large
# data_source="gpt2"
# model_type="energy"
# selection_criterion="attribute" # previously: primary_allsat, mrr_allsat, weighted_sum / # modified on 23/09/04: allsat, attribute, weighted_sum
# locate_unit="word"

# OUTPUTDIR=outputs/toxicity/locate-edit-${data_source}-loc-${NUM_EDIT_TOKEN_PER_STEP}toks-${NUM_LOCATE_STEPS}steps-project-${NUM_PROJECT_STEPS}steps
# mkdir -p $OUTPUTDIR

BEGINNOISE=5.0
ENDNOISE=0.05
NOISESTEPS=150

POSITIVELABEL=1
NEGATIVELABEL=0

# EPSILONS=(-3) # the constraint is  log(negative_prob) - log (positive_prob) < epsilon, where positive_prob is the desired label
model_types=("mucola")

NUM_EDIT_TOKEN_PER_STEP=3
locate_unit="word"
selection_criterion="allsat"
data_source="gpt2"
EPSILON=-3

# NUM_LOG_STEPS=1
# NUM_LOCATE_STEPS=1 # if -1 then locate only once.
# NUM_EDIT_TOKEN_PER_STEP=1 # if -1 then locate all tokens (same as mucola)

# 1 를 나눠서 수행
NUM_PROJECT_STEPSS=(16 18)
STEP_SIZES=(0.01)


for model_type in "${model_types[@]}"
do
    for NUM_PROJECT_STEPS in "${NUM_PROJECT_STEPSS[@]}"
    do 
        for STEP_SIZE in "${STEP_SIZES[@]}"
        do 
            NUM_LOG_STEPS=$NUM_PROJECT_STEPS
            NUM_LOCATE_STEPS=$NUM_PROJECT_STEPS
            
            echo $NUM_LOCATE_STEPS
            echo $NUM_LOG_STEPS
            echo $NUM_PROJECT_STEPS
            echo $model_type
            echo $STEP_SIZE

            OUTPUTDIR=outputs/toxicity/projection-debug/${data_source}-${locate_unit}-netps${NUM_EDIT_TOKEN_PER_STEP}-nls${NUM_LOCATE_STEPS}-nps${NUM_PROJECT_STEPS}-model${model_type}-lr${STEP_SIZE}-2
            mkdir -p $OUTPUTDIR
            bash examples/prompt/constrained_sampling_locate_edit_for_testset.sh \
            nontoxic \
            $OUTPUTDIR \
            $MODELNAME \
            run_and_evaluate \
            $STEP_SIZE \
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
            true \
            true \
            $STEP_SIZE \
            0.0 \
            0 \
            true \
            false \
            true \
            $data_source \
            $NUM_LOCATE_STEPS \
            $NUM_EDIT_TOKEN_PER_STEP\
            $NUM_PROJECT_STEPS\
            $NUM_LOG_STEPS\
            $model_type\
            $selection_criterion\
            $locate_unit
            #note: most arguments passed to constrained_sampling_mucola.sh are not used anymore
        done
    done
done

# sbatch sbatch.sh examples/prompt/constrained_sampling_new.sh nontoxic nontoxic/$datevar-nontoxic-$EPSILON $MODELNAME run_and_evaluate 0.25 dotplusplus l2 0.0 constant legal 0 1 0.0 false 1 0.5 0.05 50 target 100 1.0 constant 2 50 $EPSILON false false 0.45 0.01 0 true
