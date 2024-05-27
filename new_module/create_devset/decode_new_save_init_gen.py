import logging
import math
import os
import sys
import re
import torch
import numpy as np
import pandas as pd
import transformers
import gc
import time
import json
import os
## 23/7/17 - Hayley
import joblib
from tqdm import tqdm
##

from transformers import AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer, util

from mucoco.utils import TargetProbability, TargetEmbeddings, TargetSimplex, Lambda, Optimizer, OptimizerLE, get_epsilon
import mucoco.losses as lossbuilder
import mucoco.options as options
import mucoco.utils as utils
import torch.nn.functional as F

# To control logging level for various modules used in the application:
# from here: https://github.com/huggingface/transformers/issues/3050
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

def main(args):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.ERROR,
        stream=sys.stdout,
    )
    logger = logging.getLogger("mucoco")
    # logger.setLevel(logging.ERROR)
    logger.setLevel(logging.DEBUG)
    logger.info(args)

    ## 23/7/17 - Hayley
    outf_init = open(args.outfile + ".init", "w")
    ## 

    # Fix seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu
    logger.info(
        "loading model(s) from {} and tokenizer(s) from {}".format(
            args.model, args.tokenizer
        )
    )

    name2tokenizer = {}
    name2model = {}
    name2config = {}
    loss2modelname = {}
    loss2tokenizer = {}
    embed_luts = []
    embed_scales = []

    betas = []
    model_paths = args.model.split(":")
    tokenizer_paths = args.tokenizer.split(":")
    print('tokenizer_paths', tokenizer_paths)
    cur_lr = args.lr
    args.jsonl_tokenized = args.jsonl_tokenized == "true"

    if args.model_types is not None:
        model_types = args.model_types.split(":")
    else:
        model_types = [AutoModel for _ in model_paths]

    losses = args.loss.split(":")
    if args.lossabbr is not None:
        lossabbr = args.lossabbr.split(":")
    else:
        lossabbr = [x for x in losses]

    if args.label_id is None or args.label_id == "none":
        label_ids = [1 for _ in losses]
    else:
        label_ids = [int(i) for i in args.label_id.split(":")]
    
    if args.keywords is None or args.keywords == "none":
        keywords = ["the" for _ in losses]
    elif args.keywords in ["_roc_", "_commongen_", "_commongenunique_"]:
        keywords = ["" for _ in losses] # will be different for each input
    else:
        keywords = args.keywords.split(":")
        if len(keywords) == 1:
            keywords = [f"_topic_:{args.keywords[0]}" for _ in losses] #when keyword isn't used but topic is passed
    
    if "allsat" in args.selection_criterion: 
        # with this flag, the output which minimized the primary objective while satisfying all objectives is selected. In case all constraints are not satisfied (e.g when constraints are competing or optimization fails), this will predict the default output (Using an autoregressive decoding setup: beam search in this case)
        betas = [1.0] + [0.0 for _ in range(len(losses)-1)]
    elif (args.selection_criterion == "weighted_sum" and args.betas is not None) or args.selection_criterion == "last":
        # this setup will select the best outputs according to the weights betas for each of the losses (even though they are not satisfied)
        betas = [float(beta) for beta in args.betas.split(":")]
    else:
        raise ValueError("correct selection_criterion or betas needs to be specified")

    assert len(betas) == len(losses) and len(losses) == len(model_paths) and len(model_paths) == len(model_types) and len(betas) == len(lossabbr)
    assert np.abs(sum(betas) - 1.0) < 1e-6, f"sum of betas is {sum(betas)} != 1.0"

    prev_vocab_size = None
    vocab_size = None
    primary_vocab_size = None

    ##################################################################################################################################################################
    #Load the models and tokenizers
    ##################################################################################################################################################################
    for i, model_path in enumerate(model_paths):
        if model_path not in name2model: #making sure we are not loading the model twice in case some constraints use the same model. 
            name2tokenizer[model_path] = AutoTokenizer.from_pretrained(tokenizer_paths[i], cache_dir=args.cache_dir,  use_fast=True)
            name2config[model_path] = AutoConfig.from_pretrained(model_path, cache_dir=args.cache_dir)
            # print(model_path)
            # print(args.cache_dir)
            # print(os.getcwd())
            # print(name2config[model_path])
            if model_types[i] == "sentence-transformer":
                name2model[model_path] = lossbuilder.ModelWrapper(SentenceTransformer(model_path))
            elif "Custom" in model_types[i]:
                name2model[model_path] = lossbuilder.ModelWrapper(getattr(utils, model_types[i]).from_pretrained(model_path, config=name2config[model_path], cache_dir=args.cache_dir))
            else:
                name2model[model_path] = lossbuilder.ModelWrapper(getattr(transformers, model_types[i]).from_pretrained(model_path, config=name2config[model_path], cache_dir=args.cache_dir))
            
            # if not args.show_warnings:
            #     # print(logging.root.manager.loggerDict)
            #     # input()
            #     set_global_logging_level(logging.ERROR, [name2model[model_path].__module__])
            #     # logging.getLogger(name2model[model_path].__class__.__name__).setLevel(logging.ERROR) 
            
            name2model[model_path].eval()
            embed_lut_ = name2model[model_path].get_input_embeddings()
            if isinstance(embed_lut_, torch.nn.Sequential):
                new_vocab_size = embed_lut_[0].num_embeddings
            else:
                new_vocab_size = embed_lut_.num_embeddings
            if prev_vocab_size is None:
                vocab_size=new_vocab_size
            if new_vocab_size != prev_vocab_size and prev_vocab_size is not None:
                if not args.allow_diff_vocab:
                    raise ValueError(f"all models should have the same vocabulary {new_vocab_size} != {vocab_size}")
                else:
                    logger.warning("all models don't have the same vocabulary and we are still proceeding")
            prev_vocab_size = vocab_size
        
        if args.target_tokenize_different: # for seq2seq models where target tokenizer is different than the source tokenizer
            embed_luts.append(name2model[model_path].get_decoder().get_input_embeddings())
        else:
            input_embeds = name2model[model_path].get_input_embeddings()
            if isinstance(input_embeds, torch.nn.Sequential):
                input_embeds = input_embeds[0]
            embed_luts.append(input_embeds)
        
        if args.target_type == "embeds":
            embed_luts[-1].requires_grad=False
        
        if i == 0:
            primary_vocab_size = vocab_size
            primary_embed_dim = embed_luts[-1].embedding_dim
        
        if getattr(name2model[model_path], "get_decoder", None) is None: #this is for MarianMT models which have a weird embedding_scale parameter
            embed_scales.append(1.0)
        else:
            embed_scales.append(getattr(name2model[model_path].get_decoder(), "embed_scale", 1.0))
    
    if use_cuda:
        for name, model in name2model.items():
            model.cuda()
        logger.info("model(s) moved to GPU")
    ##################################################################################################################################################################
    
    ##################################################################################################################################################################
    #first loss is the primary loss, others are constraints
    ##################################################################################################################################################################
    lossfns = []
    for i, loss in enumerate(losses):
        lossfns.append(lossbuilder.build_loss(loss, name2model[model_paths[i]], name2tokenizer[model_paths[i]], args))
        loss2modelname[loss] = model_paths[i]
        loss2tokenizer[loss] = name2tokenizer[model_paths[i]]
    primary_tokenizer = loss2tokenizer[losses[0]]
    
    logger.info("tokenizer(s), model(s) and loss function(s) loaded")

    if args.model_dtype == "fp16": #while this is supported it doesn't work that well yet. Not recommended
        for name, model in name2model.items():
            model.half()
        logger.info("changed everything to fp16")

    #constraint thresholds. In the paper, we recommend to start with a high threshold value which is usually satisfied by default or easily satisfied and then decrease it gradually, otherwise weird adversarial solutions come up. This code supports different kinds of schedules for decreasing this threshold (usually just step or linear suffices). If no schedule is specified, it just remains the same as the original. 
    if args.epsilons is not None and args.epsilons != "none":
        epsilons = [float(eps) for eps in args.epsilons.split(":")]
        if args.min_epsilons is not None:
            min_epsilons = [float(eps) for eps in args.min_epsilons.split(":")]
            epsilon_warmup_steps = [int(steps) for steps in args.epsilon_warmup_steps.split(":")]
            epsilon_cooldown_steps = [int(steps) for steps in args.epsilon_cooldown_steps.split(":")]
            epsilon_decay_functions = [f for f in args.epsilon_decay_functions.split(":")]
        else:
            min_epsilons = [float(eps) for eps in args.epsilons.split(":")]
            epsilon_warmup_steps = [1 for eps in min_epsilons]
            epsilon_cooldown_steps = [2 for eps in min_epsilons]
            epsilon_decay_functions = ["none" for eps in min_epsilons]
        min_epsilons = [eps + getattr(lossfns[i+1], "epsilon_additive", 0)  for i, eps in enumerate(min_epsilons)]
    else:
        epsilons = []
        min_epsilons = []
        decay_function = []
        epsilon_warmup_steps = []
        epsilon_cooldown_steps = []
    ##################################################################################################################################################################
    
    ##################################################################################################################################################################
    # assert args.data is not None or args.additional_data is not None, "no data path has been provided"
    source_dataset = None
    target_dataset = None
    additional_dataset = None
    # print("yass queen", args.use_context)
    args.use_context = args.use_context == "true"
    # print(args.use_context)
    if args.data is not None:
        data_paths = args.data.split(":")
        if len(data_paths) == 1:
            source_data = data_paths[0]
            target_data = data_paths[0] #not used
            context_data = data_paths[0] # not used
            # args.use_context = False
        elif len(data_paths) == 2:
            source_data = data_paths[0]
            target_data = data_paths[1] # useful for debugging
            context_data = data_paths[1] #not used here
            # args.use_context = False
        else:
            source_data = data_paths[0]
            target_data = data_paths[1] # useful for debugging
            context_data = data_paths[2] # tsv file
    
        additional_data = args.additional_data
        if additional_data is None or additional_data == "none":
            additional_data = source_data # additional data was used in STRAP (Krishna et al 2020) when x is paraphrased to z, then the model is used to generate y in the target style. If there's no additional_data, it defaults to the source text
    elif args.additional_data is not None and additional_data != "none":
        source_data = args.additional_data
        target_data = args.additional_data
        additional_data = args.additional_data
    else:
        source_dataset = sys.stdin
        start_idx = 0
        end_idx = 1000000 # a very high number
    
    # load data
    if source_dataset is None:
        logger.info("Loading the dataset ...")
        if args.datastyle == "text":
            source_dataset = [l.strip() for l in open(source_data)]
            target_dataset = [l.strip() for l in open(target_data)]
            context_dataset = []
            import csv
            with open(context_data) as csvfile: #there can be multiple contexts, for example for paraphrasing, so we allow for a list of contexts for every input
                reader = csv.reader(csvfile, delimiter="\t")
                for row in reader:
                    context_dataset.append(row)
            additional_dataset = [l.strip() for l in open(additional_data)]
        elif args.datastyle == "jsonl": #for some prompts datasets
            source_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(source_data)]
            target_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(target_data)]
            additional_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(additional_data)]
            if args.jsonl_secondary_key is not None and args.jsonl_secondary_key != "none":
                source_dataset = [x[args.jsonl_secondary_key] for x in source_dataset]
                target_dataset = [x[args.jsonl_secondary_key] for x in target_dataset]
                additional_dataset = [x[args.jsonl_secondary_key] for x in additional_dataset]

            context_dataset = [None] * len(source_dataset)
            if args.use_context:
                context_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(context_data)]
                if args.jsonl_secondary_key is not None and args.jsonl_secondary_key != "none":
                    context_dataset = [x[args.jsonl_secondary_key] for x in context_dataset]
        elif args.datastyle == "single-jsonl": #one jsonl file has all the information
            source_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(source_data)]
            target_dataset = [json.loads(l)[args.jsonl_secondary_key] for l in open(target_data)]
            additional_dataset = [json.loads(l)[args.jsonl_secondary_key] for l in open(additional_data)]
            
            context_dataset = [None] * len(source_dataset)
            if args.use_context:
                context_dataset = [[json.loads(l)[args.jsonl_secondary_key]] for l in open(context_data)] #meaningful
        start_idx = args.start_idx
        end_idx = (len(source_dataset) + args.end_idx) % len(source_dataset) # also works with negative end_idx

        logger.info("Data loaded")
    ##################################################################################################################################################################
    
    ##################################################################################################################################################################
    # Doing prediction + constrained decoding
    ##################################################################################################################################################################
    source_batch, target_batch, additional_batch, for_predicted_source_batch, predicted_batch, context_batch = [], [], [], [], [], []
    batch_size = args.batch_size # higher than 1 batch size does not work at the moment. It won't fit in a single GPU anyway 
    
    device = "cuda" if use_cuda else "cpu"
    c = 0

    losslists = [[] for _ in range(len(losses))]
    predictedlosslists = [[] for _ in range(len(losses))]
    source_primarylosslist = [] 
    # allparetosets = []
    all_stepcounts = []
    avg_time = 0

    #data loading is very simple and probably can be sped up

    if args.gold_loss_epsilons is not None and args.gold_loss_epsilons != "none":
        args.gold_loss_epsilons = args.gold_loss_epsilons.lower().split(":")
        assert len(args.gold_loss_epsilons) == len(losses)-1
    else:
        args.gold_loss_epsilons = ["false" for _ in range(len(losses)-1)]

    # for source_text, target_text, additional_text in zip(source_dataset, target_dataset, additional_dataset):
    example_p = 1
    args.random_example = args.random_example == "true"
    
    ## 23/7/17 - Hayley
    args.num_examples = 1000 #len(source_dataset)
    ##
    
    if args.num_examples > 0 and target_dataset is not None:
        example_p = args.num_examples*1.0/len(source_dataset)
    print(example_p, args.random_example)
    print(start_idx, end_idx)
    ##################################################################################################################################################################
    ##################################################################################################################################################################
    ##################################################################################################################################################################
    # start looping over prompts
    ##################################################################################################################################################################
    ##################################################################################################################################################################
    ##################################################################################################################################################################
    
    ## 23/7/17 Hayley
    init_gen_ids = dict()
    ##
    start_time = time.time()
    for text_id, source_text in enumerate(source_dataset):
        
        # # ITERATING OVER PROMPTS. DO FOLLOWING FOR EACH OF PROMPT.
        # if text_id < start_idx or text_id > end_idx:
        #     continue

        # if args.num_examples > 0 and c > 0 and c == args.num_examples: #stop after processing num_examples if it is set 
        #     print(f"done {c}")
        #     break
        
        # do_this_example = np.random.rand() <= example_p
        # if not do_this_example:
        #     print(text_id, "NOT do_this_example")
        #     continue
        
        # print(text_id, "doing it! do_this_example")

        c += 1

        new_kweight = args.kweight
        if target_dataset is not None:
            target_text = target_dataset[text_id]
            additional_text = additional_dataset[text_id]
            context_texts = context_dataset[text_id]
            # print(context_texts)
            # input()
        else:
            args.jsonl_tokenized = False
            items = source_text.split("::")
            source_text = items[0].rstrip()
            target_text = items[1].rstrip()
            if target_text == "-":
                args.init = "zeros"
            elif target_text == "--":
                args.init = "target"
            else:
                args.init = "targettarget"
                print("aaaaaaaaaaaaaaaa")
            additional_text = items[2]

            if len(items) > 3:
                args.max_output_length = int(items[3])
                args.max_length = int(items[3])
            if len(items) > 4:
                args.use_context = True
                context_texts = [items[4]].rstrip()                               
            else:
                args.use_context = False
                context_texts = []

            if len(items) > 5:
                new_kweight = float(items[5])
            # if len(items) > 4:
            #     target_text = items[4].rstrip()
            
            print(args.use_context, context_texts)
                
        if args.keywords == "_roc_":
            keywords = ["none"] + additional_text.split(", ")
            # input(keywords)
        if args.keywords == "_rocunique_":
            keywords = ["none"] + additional_text.split(", ") + + ['none']
            # input(keywords)
        elif args.keywords == "_commongen_":
            print(additional_text)
            keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#")
        elif args.keywords == "_commongenunique_":
            keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#") + ['none']
            # input(keywords)

        
        early_skip="n"
        if args.debug:
            early_skip = input(f"skip this example? {source_text} [yes(y)/maybe(m)/no(n)]")
            if early_skip == "y":
                continue
        ##################################################################################################################################################################
        # encode source and context text
        ##################################################################################################################################################################
        if not args.jsonl_tokenized:
            if source_text == "":
                source_text = primary_tokenizer.bos_token
            source_indices = primary_tokenizer.encode(source_text, return_tensors="pt").to(device)
            source_indices_write = source_indices[0].tolist()
            # if source_indices
            additional_indices = primary_tokenizer.encode(additional_text, return_tensors="pt", add_special_tokens=False).to(device)
            
            eos_token_id = primary_tokenizer.eos_token_id
            bos_token_id = primary_tokenizer.bos_token_id
            context_indices = None
            if args.target_tokenize_different:
                with primary_tokenizer.as_target_tokenizer():
                    eos_token_id=primary_tokenizer.eos_token_id
                    bos_token_id = primary_tokenizer.bos_token_id
                    if args.use_context:
                        context_indices = primary_tokenizer.encode(context_texts[0], return_tensors="pt", add_special_tokens=False).to(device).unsqueeze(1)
            elif args.use_context:
                context_indices = primary_tokenizer.encode(context_texts[0], return_tensors="pt", add_special_tokens=False).to(device).unsqueeze(1)

            if not args.target_tokenize_different and "Seq2SeqLM" in model_paths[0]:
                logger.warning("you are using a seq2seq model for your primary loss but not tokenizing the target sentences with a different target tokenizer.")

            #for_predicted_source_indices are used to compute the primary loss wrt source as target. Useful for debugging style transfer models. 
            if args.target_tokenize_different:
                with primary_tokenizer.as_target_tokenizer():
                    for_predicted_source_indices = primary_tokenizer.encode(source_text, return_tensors="pt").to(device)
                    target_indices = primary_tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False).to(device)
            else:
                for_predicted_source_indices = source_indices
                target_indices = primary_tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        else:
            source_indices_write = source_text # to write to file
            source_indices = source_text
            target_indices = target_text
            additional_indices = additional_text
            context_indices = context_texts
            if len(source_indices) == 0:
                source_indices.append(primary_tokenizer.bos_token_id)

            source_indices = torch.LongTensor([source_indices]).to(device)
            additional_indices = torch.LongTensor([additional_indices]).to(device)
                        
            #unused
            context_indices = None
            if args.use_context:
                context_indices = torch.LongTensor([context_indices]).to(device).to(device).unsqueeze(1)
            #end unused

            #for_predicted_source_indices are used to compute the primary loss wrt source as target. Useful for debugging style transfer models. 
            for_predicted_source_indices = source_indices
            target_indices = torch.LongTensor([target_indices]).to(device)

            bos_token_id = primary_tokenizer.bos_token_id
            eos_token_id = primary_tokenizer.eos_token_id
            if args.target_tokenize_different:
                with primary_tokenizer.as_target_tokenizer():
                    bos_token_id = primary_tokenizer.bos_token_id
                    eos_token_id = primary_tokenizer.eos_token_id
            
            source_text = primary_tokenizer.decode(source_indices[0].tolist())

        source_batch.append(source_indices)
        target_batch.append(target_indices)
        for_predicted_source_batch.append(for_predicted_source_indices)
        additional_batch.append(additional_indices)
        context_batch.append(context_indices)

        
        if len(source_batch) == batch_size: #this is just one for now, greater than 1 batch size will not work

            source_batch = torch.cat(source_batch, dim=0).to(device)
            target_batch = torch.cat(target_batch, dim=0).to(device)
            additional_batch = torch.cat(additional_batch, dim=0).to(device)
            for_predicted_source_batch = torch.cat(for_predicted_source_batch, dim=0).to(device)  
            
            # print("what", args.use_context)
            if args.use_context:
                context_batch = torch.cat(context_batch, dim=0).to(device)
                print(context_batch)
                
            ##################################################################################################################################################################
            # generating AR samples
            ##################################################################################################################################################################
            predicted_batches = [] #each sample x restart becomes a tensor
            for batchidx in range(source_batch.size(0)): #batch size is 1
                
                with torch.no_grad():
                    starttime = time.time()
                    AR_predicted_all, seq_length =\
                        lossfns[0].generate(
                            input_ids=source_batch[batchidx].unsqueeze(0),
                            additional_ids=additional_batch[batchidx].unsqueeze(0),
                            num_return_sequences=(args.restarts + 1)*args.num_samples) # 25 for nontoxic
                    #some bug about length

                    # AR_predicted_indices_all = []
                    AR_prediction_all = []
                    # clean output for predicted token ids
                    for sample_idx in range(len(AR_predicted_all)):
                        logger.debug(f"AR prediction: {AR_predicted_all[sample_idx]}")
                        logger.debug(f"AR prediction: {AR_predicted_all[sample_idx].squeeze().tolist()}")
                        AR_predicted_indices =\
                            clean_output(AR_predicted_all[sample_idx].squeeze().tolist(), # remove eos token, etc. # edit 24/01/13: squeeze() added to remove extra dimension. before clean_output was not working since the inputs had another dimension.
                                eos_token_id=eos_token_id,
                                return_tensors=True, allow_first_eos=losses[0] == "bart",
                                skip_special_tokens=[bos_token_id, eos_token_id])[0]
                        # AR_predicted_indices_all.append(AR_predicted_indices)
                        logger.debug(f"AR_predicted_indices: {AR_predicted_indices}")
                        
                        if args.target_tokenize_different:
                            with primary_tokenizer.as_target_tokenizer():
                                AR_prediction = primary_tokenizer.decode(AR_predicted_indices.tolist())
                        else:
                            AR_prediction = primary_tokenizer.decode(AR_predicted_indices.tolist())
                            
                        logger.debug(f"AR prediction: {AR_prediction}")
                        ## added 24/01/13
                        AR_prediction = remove_unfinished_sent(AR_prediction)
                        AR_predicted_indices = primary_tokenizer.encode(AR_prediction, return_tensors="pt").to(device)
                        logger.debug(f"After removing unfinished sentences.\n AR prediction: {AR_prediction}")
                            
                        AR_prediction_all.append(AR_prediction)
                        # print(AR_prediction)
                        
                        # predicted_batch.append(AR_predicted_indices)
                        predicted_batches.append(AR_predicted_indices.to(device))
                        
                        if sample_idx == 0:
                            output = {
                                "prompt":{
                                    "text":source_text,
                                    "tokens":source_indices_write}, 
                                "generations":[{
                                    "text": AR_prediction,
                                    "tokens": AR_predicted_indices.squeeze().tolist()
                                    }]
                            }
                        else:
                            output['generations'].append(
                                {
                                    "text": AR_prediction,
                                    "tokens": AR_predicted_indices.squeeze().tolist()
                                }
                            )
                            
                        if sample_idx + 1 == args.num_samples:
                            json.dump(output, outf_init)
                            outf_init.write("\n")
                            outf_init.flush()
                    if args.time:
                        print(time.time()-starttime)
                        
            ### 23/7/17 Hayley
                
            del source_batch
            del target_batch
            del additional_batch
            del for_predicted_source_batch
            # del predicted_batch
            source_batch = []
            target_batch = []
            for_predicted_source_batch = []
            additional_batch = []
            # predicted_batch = []
            context_batch = []

    end_time = time.time() 
    outf_init.close()
    print('decoding time: ', start_time-end_time)


def sentence_completion(prompt, tokens, lossfn):
    lossfn.args.max_output_length = lossfn.args.max_output_length + 10
    print(tokens)
    new_tokens = lossfn.generate(torch.cat([prompt, torch.LongTensor([tokens]).to(lossfn.device)], dim=1))
    print(new_tokens)
    lossfn.args.max_output_length = lossfn.args.max_output_length - 10
    return tokens + new_tokens[0].tolist()
    # return tokens

def clean_output(tokens, eos_token_id, return_tensors=False, allow_first_eos=False, skip_special_tokens=[], prompt=None, sentence_complete=False, lossfn=None):
    # print(tokens)
    if sentence_complete:
        tokens = sentence_completion(prompt, tokens, lossfn)
    new_tokens = []
    if isinstance(tokens, int):
        tokens = [tokens]
    for i, tok in enumerate(tokens):
        if tok == eos_token_id and (not allow_first_eos or i > 0):
            break
        
        if (tok not in skip_special_tokens):
            new_tokens.append(tok)
        
    if return_tensors:
        return torch.LongTensor([new_tokens])
    return new_tokens

def remove_unfinished_sent(x):
    x = re.sub(r'(?<=\.\s)[^\.\!\?\;\:]*$','',x).strip()
    return x
    
def cli_main():
    parser = options.get_parser()
    args = parser.parse_args()
    main(args)
