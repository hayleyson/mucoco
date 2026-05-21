import json, torch, os
from argparse import Namespace

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import mucoco.losses as lossbuilder
from mucoco.utils import TargetEmbeddings, RobertaCustomForSequenceClassification
from new_module.utils.utils import read_outputs
from new_module.utils.robertacustom import define_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## load configuration
config = json.load(open('examples/prompt/toxicity-all/arguments_gpt35_gpt2.txt', 'r'))
args = Namespace(**config)

## load data to test
test_data = read_outputs(
    'new_module/data/logical-consistency/anli-r2-test_prompt_4.jsonl',
)

## load models (gpt2-large, nli classifier with embed share)

embed_luts=[]
embed_scales=[]

causal_lm_config = AutoConfig.from_pretrained('gpt2-large')
causal_lm = lossbuilder.ModelWrapper(AutoModelForCausalLM.from_pretrained('gpt2-large', config=causal_lm_config, cache_dir=os.getenv('HF_HOME')))
causal_lm.eval()
causal_lm.to(device)
causal_lm_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
causal_lm_input_embeds = causal_lm.get_input_embeddings()
causal_lm_input_embeds.requires_grad=False
embed_luts.append(causal_lm_input_embeds)
embed_scales.append(1.0)
primary_embed_dim = causal_lm_input_embeds.embedding_dim

model_config = json.load(open('/home/hyeryung/data/loc_edit/models/nli/roberta_large_snli_mnli_anli_train_dev_with_finegrained_binary_labels_binary_cross_entropy_n_a_gpt2_embeds/sef7vf7z/config.json', 'r'))
if model_config['energynet']['output_form'] == 'real_num':
    num_classes = 1
elif model_config['energynet']['output_form'] == '2dim_vec':
    num_classes = 2
elif model_config['energynet']['output_form'] == '3dim_vec':
    num_classes = 3

nli_classifier, nli_classifier_tokenizer = define_model(num_classes=num_classes, 
                                                        mod_path=model_config['model_path'],
                                                        load_weights=True,
                                                        device=model_config['device'], 
                                                        output_hidden_states=True,
                                                        encoder_model=model_config['energynet']['base_model'],
                                                        embedding_model=model_config['energynet'].get('embedding_model', 'gpt2-large'),
                                                        task="nli")
nli_classifier.eval()

nli_classifier_input_embeds = nli_classifier.get_input_embeddings()[0]
nli_classifier_input_embeds.requires_grad=False
embed_luts.append(nli_classifier_input_embeds)
embed_scales.append(1.0)

lossfns = []
losses = ['gpt2_no_prefix', 'classification_nli']


## define losses

loss1 = lossbuilder.build_loss(
                'gpt2_no_prefix',
                causal_lm,
                causal_lm_tokenizer,
                args,
            )

loss2 = lossbuilder.build_loss(
                'classification_nli',
                nli_classifier,
                nli_classifier_tokenizer,
                args,
            )

lossfns = [loss1, loss2]

final_bias = None
if args.final_bias:
    final_bias = lossfns[0].model.final_logits_bias

## define sample premise & hypothesis

sample_premise = test_data['prompt'].tolist()[:10]
sample_hypothesis = test_data['text'].tolist()[:10]


## run compute_loss and compute_gold_loss for gpt2_no_prefix
## run compute_loss and compute_gold_loss for classification_nli

compute_loss1_list = []
compute_gold_loss1_list = []
compute_loss2_list = []
compute_gold_loss2_list = []

step = 0
label_ids = [-1, 1]
keywords = ["the", "the"]
new_kweight = args.kweight

# Optional decode kwargs (mirrors decode_new_clean); unused by these loss paths when empty/None.
additional_batch = None
context_batch = None
original_preds = None

for i, (premise, hypothesis) in enumerate(zip(sample_premise, sample_hypothesis)):
    source_batch = causal_lm_tokenizer.encode(premise, return_tensors='pt').to(device)
    predicted_batch = causal_lm_tokenizer.encode(hypothesis, return_tensors='pt').to(device)
    prefix_ids = torch.empty((source_batch.size(0), 0)).long().to(device)
    
    init_value = embed_luts[0](predicted_batch)
    sent_length = init_value.size(1)
    
    
    
    outputs = TargetEmbeddings(
                                    embed_dim=primary_embed_dim,
                                    embed_lut=embed_luts[0],
                                    sent_length=sent_length,
                                    batch_size=1,
                                    device=device,
                                    st=args.st,
                                    init_value=init_value,
                                    random_init=args.init == "random",
                                    sampling_strategy=args.sampling_strategy,
                                    sampling_strategy_k=args.sampling_strategy_k,
                                    embed_scales=embed_scales,
                                    metric=args.metric,
                                    same_embed=args.same_embeds,
                                    final_bias=final_bias,
                                    eos_token_id=causal_lm_tokenizer.eos_token_id
                                )
    
    pred_embeds, pred_tokens, pred_probs = outputs.forward_multiple(embed_luts, new_predictions=predicted_batch)
    
    compute_gold_loss1_list.append(loss1.compute_gold_loss((source_batch, predicted_batch)))
    compute_loss1_list.append(loss1.compute_loss(batch = (source_batch, prefix_ids), 
                                                 preds = (pred_tokens, pred_embeds[0][0], pred_probs),
                                                 additional_batch=additional_batch, 
                                                    context_batch=context_batch,
                                                    use_context=args.use_context,
                                                    embed_scale=embed_scales[0], 
                                                    label_id=label_ids[0],
                                                    keyword=keywords[0],
                                                    original_preds=original_preds,
                                                    kweight=new_kweight,
                                                    step=step))
    compute_gold_loss2_list.append(loss2.compute_gold_loss((source_batch, predicted_batch)))
    compute_loss2_list.append(loss2.compute_loss(batch = (source_batch, prefix_ids), 
                                                preds = (pred_tokens, pred_embeds[0][1], pred_probs),
                                                additional_batch=additional_batch, 
                                                    context_batch=context_batch,
                                                    use_context=args.use_context,
                                                    embed_scale=embed_scales[1], 
                                                    label_id=label_ids[1],
                                                    keyword=keywords[1],
                                                    original_preds=original_preds,
                                                    kweight=new_kweight,
                                                    step=step))
    print(f"==== {i}th Example ====")                 
    print(f"Premise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"Compute Loss 1: {compute_loss1_list[-1][0]}")
    print(f"Compute Gold Loss 1: {compute_gold_loss1_list[-1][0]}")
    print(f"Compute Loss 2: {compute_loss2_list[-1][0]}")
    print(f"Compute Gold Loss 2: {compute_gold_loss2_list[-1][0]}")
    


