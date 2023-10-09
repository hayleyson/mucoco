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
## 23/7/18 - Hayley
import joblib
import time 
from tqdm import tqdm
import pdb
##

from transformers import AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer, util

from mucoco.utils import TargetProbability, TargetEmbeddings, TargetSimplex, Lambda, Optimizer, OptimizerLE, get_epsilon, locate
import mucoco.losses as lossbuilder
import mucoco.options as options
import mucoco.utils as utils
import torch.nn.functional as F

# torch.autograd.set_detect_anomaly(True)

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
    logger.setLevel(logging.ERROR)
    logger.info(args)

    if args.outfile is not None:
        if args.resume_index == 0:
            outf = open(args.outfile, "w")
            outallsatf = open(args.outfile + ".allsat", "w")
            outf2 = open(args.outfile + ".intermediate", "w")
            
            outfparams = open(args.outfile + ".params", "w")
            outfparams.write("%s\n" %args)
            outfparams.flush()
            outfparams.close()
        else:
            outf = open(args.outfile, "a")
            outallsatf = open(args.outfile + ".allsat", "a")
            outf2 = open(args.outfile + ".intermediate", "a")
    
    if args.num_locate_steps == -1:
        args.num_locate_steps = args.optim_steps + 1

    # Fix seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    ## 23/7/18 - Hayley 
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
    
    ## 23/09/04 - always set betas with the arguments provided.
    # if "allsat" in args.selection_criterion: 
    #     # with this flag, the output which minimized the primary objective while satisfying all objectives is selected. In case all constraints are not satisfied (e.g when constraints are competing or optimization fails), this will predict the default output (Using an autoregressive decoding setup: beam search in this case)
    #     betas = [1.0] + [0.0 for _ in range(len(losses)-1)]
    # elif (args.selection_criterion == "weighted_sum" and args.betas is not None) or args.selection_criterion == "last":
    #     # this setup will select the best outputs according to the weights betas for each of the losses (even though they are not satisfied)
    #     betas = [float(beta) for beta in args.betas.split(":")]
    # else:
    #     raise ValueError("correct selection_criterion or betas needs to be specified")
    betas = [float(beta) for beta in args.betas.split(":")]

    assert len(betas) == len(losses) and len(losses) == len(model_paths) and len(model_paths) == len(model_types) and len(betas) == len(lossabbr)
    assert np.abs(sum(betas) - 1.0) < 1e-6, f"sum of betas is {sum(betas)} != 1.0"

    prev_vocab_size = None
    vocab_size = None
    primary_vocab_size = None

    ##################################################################################################################################################################
    #😃 Load the models and tokenizers
    ##################################################################################################################################################################
    for i, model_path in enumerate(model_paths):
        if model_path not in name2model: #making sure we are not loading the model twice in case some constraints use the same model. 
            name2tokenizer[model_path] = AutoTokenizer.from_pretrained(tokenizer_paths[i], cache_dir=args.cache_dir,  use_fast=True)
            name2config[model_path] = AutoConfig.from_pretrained(model_path, cache_dir=args.cache_dir)

            if model_types[i] == "sentence-transformer":
                name2model[model_path] = lossbuilder.ModelWrapper(SentenceTransformer(model_path))
            elif "Custom" in model_types[i]:
                name2model[model_path] = lossbuilder.ModelWrapper(getattr(utils, model_types[i]).from_pretrained(model_path, config=name2config[model_path], cache_dir=args.cache_dir))
            else:
                name2model[model_path] = lossbuilder.ModelWrapper(getattr(transformers, model_types[i]).from_pretrained(model_path, config=name2config[model_path], cache_dir=args.cache_dir))
            
            if not args.show_warnings:
                set_global_logging_level(logging.ERROR, [name2model[model_path].__module__])
            
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
    args.use_context = args.use_context == "true"
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
            if args.task_type == "prompted_generation":
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
                ##@
                if args.dev_mode == "true":
                    generation_dataset = [json.loads(l)["generations"] for l in open(source_data)]
            elif args.task_type == "revision":
                source_dataset = ["" for l in open(source_data)]
                target_dataset = ["" for l in open(target_data)]
                additional_dataset = ["" for l in open(additional_data)]
                context_dataset = ["" for l in open(context_data)]
                ##@
                generation_dataset = [json.loads(l)["source"] for l in open(source_data)]
        elif args.datastyle == "single-jsonl": #one jsonl file has all the information
            source_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(source_data)]
            target_dataset = [json.loads(l)[args.jsonl_secondary_key] for l in open(target_data)]
            additional_dataset = [json.loads(l)[args.jsonl_secondary_key] for l in open(additional_data)]
            
            context_dataset = [None] * len(source_dataset)
            if args.use_context:
                context_dataset = [[json.loads(l)[args.jsonl_secondary_key]] for l in open(context_data)] #meaningful
        start_idx = args.start_idx
        end_idx = (len(source_dataset) + args.end_idx) % len(source_dataset) # also works with negative end_idx

        ## 23/07/18 - Hayley - load already generated data.
        ## 23/09/04 modified the code to accomodate change in file format 
        ## (a pkl file containing only input_ids => a jsonl file containing prompt, generation, input_ids, locate_labels)
        ## (=> don't use args.input_ids_path or args.texts_path)
        ## 23/09/05 - moved this inside the if-else block above. (marked as ##@)
        # init_gen_ids = joblib.load(args.input_ids_path)
        # if os.path.exists(args.texts_path):
        #     source_dataset = joblib.load(args.texts_path)
        # else:
        #     source_dataset = ["" for i in range(len(init_gen_ids))]
        ##

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
    all_stepcounts = []
    avg_time = 0

    #data loading is very simple and probably can be sped up

    if args.gold_loss_epsilons is not None and args.gold_loss_epsilons != "none":
        args.gold_loss_epsilons = args.gold_loss_epsilons.lower().split(":")
        assert len(args.gold_loss_epsilons) == len(losses)-1
    else:
        args.gold_loss_epsilons = ["false" for _ in range(len(losses)-1)]

    # for source_text, target_text, additional_text in zip(source_dataset, target_dataset, additional_dataset):
    example_p = 1.0
    args.random_example = args.random_example == "true"
    if args.num_examples > 0 and target_dataset is not None:
        example_p = args.num_examples*1.0/len(source_dataset)
    ##################################################################################################################################################################
    ##################################################################################################################################################################
    ##################################################################################################################################################################
    # start looping over prompts
    ##################################################################################################################################################################
    ##################################################################################################################################################################
    ##################################################################################################################################################################
    for text_id, source_text in enumerate(source_dataset):
        
        ## 23/7/18 - Hayley - commented it out.
        ## 23/9/6 - Added if-else block depending on args.dev_mode
        if args.dev_mode == "true":
            pass # do all the prompts
        else:
            # ITERATING OVER PROMPTS. DO FOLLOWING FOR EACH OF PROMPT.
            if text_id < start_idx or text_id > end_idx:
                continue

            if args.num_examples > 0 and c > 0 and c == args.num_examples: #stop after processing num_examples if it is set 
                print(f"done {c}")
                break
            
            do_this_example = np.random.rand() <= example_p
            if not do_this_example:
                continue
        
        if text_id < args.resume_index:
            continue
            
        print(text_id, "doing it! do_this_example")

        c += 1

        new_kweight = args.kweight
        if target_dataset is not None:
            target_text = target_dataset[text_id]
            additional_text = additional_dataset[text_id]
            context_texts = context_dataset[text_id]
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
            
            print(args.use_context, context_texts)
                
        if args.keywords == "_roc_":
            keywords = ["none"] + additional_text.split(", ")
        if args.keywords == "_rocunique_":
            keywords = ["none"] + additional_text.split(", ") + + ['none']
        elif args.keywords == "_commongen_":
            print(additional_text)
            keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#")
        elif args.keywords == "_commongenunique_":
            keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#") + ['none']

        
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
            
            if args.use_context:
                context_batch = torch.cat(context_batch, dim=0).to(device)
                print(context_batch)
            
            ## 23/07/18 - Hayley
            
            ##################################################################################################################################################################
            # generating AR samples
            ##################################################################################################################################################################
            if args.task_type == "revision":
                # assumption: args.num_samples == 1
                AR_prediction_all = [generation_dataset[text_id]["text"]]
                predicted_batches = primary_tokenizer.encode(AR_prediction_all[0], return_tensors="pt", add_special_tokens=False).to(device).unsqueeze(0)
            elif args.task_type == "prompted_generation":
                if args.dev_mode == "true":
                    predicted_batches = [x["tokens"] for x in generation_dataset[text_id]]
                    predicted_batches = [torch.tensor([x], dtype=torch.long, device=device) for x in predicted_batches]
                    
                    AR_prediction_all = [x["text"] for x in generation_dataset[text_id]]
                else:
                    predicted_batches = [] #each sample x restart becomes a tensor
                    for batchidx in range(source_batch.size(0)): #batch size is 1
                        with torch.no_grad():
                            starttime = time.time()
                            AR_predicted_all =\
                                lossfns[0].generate(
                                    input_ids=source_batch[batchidx].unsqueeze(0),
                                    additional_ids=additional_batch[batchidx].unsqueeze(0),
                                    num_return_sequences=(args.restarts + 1)*args.num_samples) # 25 for nontoxic
                            #some bug about length

                            # AR_predicted_indices_all = []
                            AR_prediction_all = []
                            # clean output for predicted token ids
                            for sample_idx in range(len(AR_predicted_all)):
                                AR_predicted_indices =\
                                    clean_output(AR_predicted_all[sample_idx].tolist(), # remove eos token, etc.
                                        eos_token_id=eos_token_id,
                                        return_tensors=True, allow_first_eos=losses[0] == "bart",
                                        skip_special_tokens=[bos_token_id, eos_token_id])
                                # AR_predicted_indices_all.append(AR_predicted_indices)

                                if args.target_tokenize_different:
                                    with primary_tokenizer.as_target_tokenizer():
                                        AR_prediction = primary_tokenizer.decode(AR_predicted_indices[0].tolist())
                                else:
                                    AR_prediction = primary_tokenizer.decode(AR_predicted_indices[0].tolist())
                                AR_prediction_all.append(AR_prediction)
                                print(AR_prediction)
                                
                                # predicted_batch.append(AR_predicted_indices)
                                predicted_batches.append(AR_predicted_indices.to(device))
                            if args.time:
                                print(time.time()-starttime)
                    
                    # change to read from testset file.
                    if type(init_gen_ids[text_id]) == list:
                        predicted_batch = torch.tensor([init_gen_ids[text_id]], dtype=torch.long, device=device)
                    else:
                        predicted_batch = init_gen_ids[text_id].unsqueeze(0).to(device)
                    
                    AR_prediction = primary_tokenizer.decode(predicted_batch[0])
                    AR_predicted_indices = predicted_batch
            
            
            ##################################################################################################################################################################
            # 25 initial outputs per prompt
            # intermediate_result={"prompt": source_text}
            ##################################################################################################################################################################
            
            broken_skip = False
            # print("args.restarts", args.restarts)
            ##################################################################################################################################################################
            # for each prompt loop over 25 samples
            ##################################################################################################################################################################
            for sample_idx in range(args.num_samples): # 25 for nontoxic
                print("sample_idx", sample_idx)
                for restart_idx in range(args.restarts + 1): # 0 for nontoxic. restart the optimization if the constraints are not satisfied
                    predicted_batch = predicted_batches[sample_idx * (args.restarts + 1) + restart_idx]
                    if use_cuda:
                        predicted_batch = predicted_batch.cuda()
                    AR_prediction = AR_prediction_all[sample_idx * (args.restarts + 1) + restart_idx]
                    intermediate_result={"text_id": text_id, "original_text": AR_prediction}
                    start_time = time.time() ## add timer
                    # print("predicted_batch", predicted_batch)
                    # print("AR_prediction", AR_prediction)
                    
                    ##TODO: in case of always_mucoco=false and num_restarts > 0, comb through the restarts and skip if constraints are satisfied

                    skip=False
                    predicted_allsat=False
                    lengthwise_best_prediction = [None] * batch_size

                    if args.debug:
                        print("AR output:", source_text, additional_text, predicted_batch)

                    # losses of the autoregressive output: we should perform atleast as well as this. If we don't, we predict this output
                    # Also, if the autoregressive output already satisfies the constraints, we skip mucoco unless, args.always_mucoco is true
                    predicted_labels = {}
                    total_weighted_loss = 0.0
                    predicted_allsat=True
                    predictedlosses = []

                    for lossid in range(len(losses)):
                        lossname = losses[lossid]

                        predicted_loss, predicted_lo =\
                            lossfns[lossid].compute_gold_loss(
                                # (source_batch, target_batch), # bug: if it's target_batch, we're inputting 2 copies of source_batch
                                (source_batch, predicted_batch), 
                                additional_batch=additional_batch, 
                                context_batch=context_batch,
                                use_context=args.use_context,
                                label_id=label_ids[lossid],
                                keyword=keywords[lossid],
                                kweight=new_kweight)

                        predictedlosses.append(predicted_loss.data.cpu())
                        predicted_loss = predicted_loss.sum().item()
                        intermediate_result.update({f"original_loss{lossid}": predicted_loss})
                        total_weighted_loss += betas[lossid] * predicted_loss

                        if lossid > 0:
                            predicted_allsat = predicted_allsat and (predicted_loss <= min_epsilons[lossid-1])
                        
                        if "label_prediction" in predicted_lo:
                            predicted_labels[lossid] = predicted_lo['label_prediction']
                        else:
                            predicted_labels[lossid] = "NA"
                        
                        if lossid > 0 and args.gold_loss_epsilons[lossid-1] == "true": #use the predicted loss as the threshold, mucoco has to beat it then
                            min_epsilons[lossid - 1] = predicted_loss + getattr(lossfns[lossid], "epsilon_additive", 0)
                            epsilons[lossid - 1] = predicted_loss + getattr(lossfns[lossid], "epsilon_additive", 0) ##TODO check 
                        
                    predictedlosslists.append(predictedlosses)
                    # print(f"[autoregressive] total_weighted_loss: {total_weighted_loss}, losses_for_backward[0]: {predictedlosses[0].data.cpu()}, losses_for_backward[1]: {predictedlosses[1].data.cpu()}, min_epsilons for 1st constraint: {min_epsilons[0]}, predicted_allsat: {predicted_allsat}")
                                        
                    
                    # if args.only_mucoco == "false":
                    #     lengthwise_best_prediction = [(AR_prediction, total_weighted_loss, predicted_allsat, predicted_batch[0].tolist(), -1)]
                    skip = predicted_allsat
                        
                    definite_skip = False
                    ask_skip = ""
                    if args.debug and early_skip=="m": 
                        print(f"new example: {source_text}\nautoregressive output: {AR_prediction}")
                        for lossid in range(len(losses)):
                            print(f"{lossabbr[lossid]} for desired label_id({label_ids[lossid]}): {predictedlosslists[-1][lossid]}; predicted label: {predicted_labels[lossid]}")
                        if predicted_allsat:
                            print(f"autoregressive output already satisfies the constraints")
                        ask_skip = input(f"skip this example? [y/n]")
                        definite_skip = ask_skip == "y"

                    # elif skip and predicted_allsat and (args.always_mucoco == "false"):
                    elif skip and predicted_allsat:
                        definite_skip = True

                    if args.debug:
                        # print('definite_skip', definite_skip, skip, predicted_allsat, args.always_mucoco)
                        print('definite_skip', definite_skip, skip, predicted_allsat)
                    
                    if not definite_skip:
                        
                        if (args.max_length is None or args.max_length == -1) and args.init not in ["source", "target"]: 
                            #since we don't know the about length, we search in a (-length_diff, length_diff) window and predict the best performing one.
                            predicted_length = predicted_batch.size(1)
                            length_range = [predicted_length + int(diff) for diff in args.length_diff.split(":")]
                            length_range = [x for x in length_range if x <= args.max_allowed_length and x >= 1]
                            if len(length_range) == 0:
                                length_range = [args.max_allowed_length]
                            length_range = sorted(list(set(length_range)))
                        elif args.init == "targettarget":
                            length_range = [target_batch.size(1)]
                        elif args.init == "target":
                            length_range = [predicted_batch.size(1)]
                        elif args.init == "source":
                            length_range = [source.size(1)]
                        else: 
                            #another way to use this approach is train models which also compute loss on <pad> token and then predict the entire sentence including pad, it has shown to work in some of our experiments
                            length_range = [args.max_length]           
                        
                        ## 23/09/04 - removed looping over length_range. just do one length.       
                        # for sent_length_ in length_range:
                        sent_length_ = length_range[0]
                        
                        # prefix_length is used to indicate if instead of predicting the entire sentence via optimization, we want to fix a prefix (of specified length) and predict the remaining suffix. We use part of the beam search prediction as the prefix. 
                        if args.prefix_length > 0:
                            sent_length = sent_length_ - args.prefix_length
                            target_prefix = predicted_batch[:, :args.prefix_length]
                        else:
                            sent_length = sent_length_
                            target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                        
                        if sent_length <= 0:
                            continue
                        if sent_length > args.max_allowed_length:
                            #max_allowed_length is just to make sure things don't go out of memory,
                            old_l = sent_length
                            sent_length = args.max_allowed_length
                            print(f"changed output length to {sent_length} from {old_l} to avoid GPU overflow. This is a temporary solution")
                        else:
                            # print("predicting a sentence length: ", sent_length)
                            pass
                            
                        if args.target_type == "simplex": # use V sized real vector for each token and apply softmax before output
                            outputs = TargetSimplex(
                                vocabsize=primary_vocab_size,
                                sent_length=sent_length,
                                batch_size=batch_size,
                                device=device,
                                temperature=args.decode_temperature,
                                st=args.st,
                                init_value=source_batch[:,1:-1] if args.init == "source" else None,
                                random_init=args.init == "random",
                                do_sample=args.expgd_do_sample,
                                top_p=args.expgd_top_p,
                                top_k=args.expgd_top_k,
                                embed_scales=embed_scales
                            )
                        elif args.target_type == "probs": # use V sized vector which sums to one for each token and apply softmax before output
                            init_value = None
                            break_after=False
                            if args.init == "source": #initialize the target with the source
                                init_value = source_batch
                                target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                sent_length = init_value.size(1)
                                break_after=True
                            elif args.init == "target": #initialize the target with the autoregressive output
                                init_value = target_batch
                                target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                sent_length = init_value.size(1)
                                break_after=True
                            
                            outputs = TargetProbability(
                                vocabsize=primary_vocab_size,
                                sent_length=sent_length,
                                batch_size=batch_size,
                                device=device,
                                st=args.st,
                                init_value=init_value,
                                random_init=args.init == "random",
                                do_sample=args.expgd_do_sample,
                                top_p=args.expgd_top_p,
                                top_k=args.expgd_top_k,
                                embed_scales=embed_scales,
                                max_steps=args.optim_steps
                            )
                        ##################################################################################################################################################################
                        # initialize embedding
                        ##################################################################################################################################################################
                        elif args.target_type == "embeds":
                            init_value = None
                            break_after=False
                            if args.init == "source": #initialize the target with the source
                                init_value = embed_luts[0](source_batch)
                                target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                sent_length = init_value.size(1)
                                break_after=True
                            elif args.init == "targettarget": #initialize the target with given target
                                init_value = embed_luts[0](target_batch)
                                target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                sent_length = init_value.size(1)
                                break_after=True 
                            elif args.init == "target": #initialize the target with the autoregressive output
                                ##################################################################################################################################################################
                                init_value = embed_luts[0](predicted_batch)
                                ##################################################################################################################################################################
                                target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                sent_length = init_value.size(1)
                                break_after=True 
                            elif args.init == "random_vocab":
                                random_indices = torch.multinomial(torch.ones(primary_vocab_size,)/primary_vocab_size, num_samples=batch_size*sent_length, replacement=True).view(batch_size, sent_length).to(device)
                                init_value = embed_luts[0](random_indices)
                            elif args.init == "embedgd-zeros":
                                if args.target_tokenize_different:
                                    with primary_tokenizer.as_target_tokenizer():
                                        indices = torch.empty((batch_size, sent_length)).long().fill_(primary_tokenizer.eos_token_id).to(device)
                                else:
                                    indices = torch.empty((batch_size, sent_length)).long().fill_(primary_tokenizer.eos_token_id).to(device)
                                init_value = embed_luts[0](indices)
                            elif args.init == "zeros":
                                indices = torch.zeros((batch_size, sent_length)).long().to(device)
                                init_value = embed_luts[0](indices)

                            
                            final_bias = None
                            if args.final_bias:
                                final_bias = lossfns[0].model.final_logits_bias
                            ##################################################################################################################################################################
                            outputs = TargetEmbeddings(
                                embed_dim=primary_embed_dim,
                                embed_lut=embed_luts[0],
                                sent_length=sent_length,
                                batch_size=batch_size,
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
                                eos_token_id=primary_tokenizer.eos_token_id
                            )
                            ##################################################################################################################################################################
                        else:
                            raise ValueError("Wrong target_type")

                        if len(losses) > 1:
                            lambda_ = Lambda(count=len(epsilons))
                            if use_cuda:
                                lambda_.cuda()
                                
                        ## 23/7/.. - Hayley - updated to allow locate & edit
                        args.optim= "embedgd_le" # change option
                        optimizer = OptimizerLE.from_opt(outputs, args)
                        optimizer.set_init_pred(predicted_batch)
                        ##
                        cur_lr = args.lr
                        if len(losses) > 1:
                            old_optim = args.optim
                            args.optim = "gradascent"
                            old_lr = args.lr
                            args.lr = args.lambda_lr
                            optimizer_lambda = Optimizer.from_opt(lambda_, args)
                            args.optim = old_optim
                            args.lr = old_lr

                        best_loss = [None] * batch_size
                        best_allsat = [False] * batch_size
                        best_repeat_count = [0] * batch_size
                        best_losses = [[None] * batch_size for _ in range(len(losses))]
                        best_step = -100
                        
                        best_pred_tokens = [None] * batch_size
                        best_prediction_set = [set() for _ in range(batch_size)]
                        best_pred_probs = [None] * batch_size
                        best_index = [-1 for i in range(batch_size)]
                        
                        scaler = None
                        if args.model_dtype == "fp16" and args.fp16_source == "pytorch":
                            scaler = torch.cuda.amp.GradScaler()
                    
                        for lossid, lossname in enumerate(losses):
                            losslists[lossid].append([])

                        broken = False
                        prev_loss = None
                        dynamic_lambda_update_prev_loss = None
                        same_loss_count = 0
                        dynamic_loss_update_same_loss_count = 0
                        starttime = time.time()
                        repeat_counts = [0] * batch_size
                        
                        ##################################################################################################################################################################
                        # gradient inference the embeddings
                        ##################################################################################################################################################################
                        
                        for step in range(args.optim_steps):
                            try:
                                with torch.cuda.amp.autocast():
                                    losses_for_backward = []
                                    logging_outputs = []

                                    pred_embeds, pred_tokens, pred_probs = outputs.forward_multiple(embed_luts, new_predictions=getattr(optimizer._optimizer, "new_predictions", None))  # forward                                    
                                    
                                    def get_sent(tokens, tokenizer):
                                        batch = []
                                        if args.target_tokenize_different:
                                            with tokenizer.as_target_tokenizer():
                                                for toks in tokens:
                                                    batch.append(tokenizer.decode(clean_output(toks.tolist(), -1, allow_first_eos=losses[0] == "bart")))
                                        else:
                                            for toks in tokens:
                                                batch.append(tokenizer.decode(clean_output(toks.tolist(), -1, allow_first_eos=losses[0] == "bart")))
                                        return batch

                                    target_sents = get_sent(torch.cat([target_prefix, pred_tokens], dim=1), primary_tokenizer)
                                    if step % args.num_log_steps == 0:
                                        intermediate_result.update({f"step_{step}_text": target_sents})
                                    # print(target_sents, end="\n")
                                    
                                    original_preds = None
                                    if len(pred_embeds) > 1:
                                        original_preds = pred_embeds[1]

                                    for lossid, lossname in enumerate(losses):
                                        lossvalue, logging_output =\
                                            lossfns[lossid].compute_loss(
                                                [source_batch, target_prefix], 
                                                [pred_tokens, pred_embeds[0][lossid], pred_probs], 
                                                additional_batch=additional_batch, 
                                                context_batch=context_batch,
                                                use_context=args.use_context,
                                                embed_scale=embed_scales[lossid], 
                                                label_id=label_ids[lossid],
                                                keyword=keywords[lossid],
                                                original_preds=original_preds,
                                                kweight=new_kweight,
                                                step=step
                                            )

                                        losslists[lossid][-1].append(lossvalue.sum().item())  #for logging
                                        # intermediate_result.update({f"step_{step}_loss{lossid}": lossvalue.sum().item()})
                                        losses_for_backward.append(lossvalue)  # for backward
                                        logging_outputs.append(logging_output)
                                    
                                    optimizer.zero_grad(set_to_none=True)
                                    outputs.zero_grad()
                                    if len(losses) > 1:
                                        optimizer_lambda.zero_grad(set_to_none=True)
                                        lambda_.zero_grad()

                                    for model in name2model.values():
                                        model.zero_grad(set_to_none=True)
                                    
                                    if args.linear_scale == "true": # no lagragian, plain old linear sum
                                        total_loss = 0
                                        cur_epsilons = [] # just for avoiding syntax errors, epsilons are useless in this setting
                                        for sid in range(len(losses_for_backward)):
                                            total_loss = total_loss + betas[sid] * losses_for_backward[sid]
                                            cur_epsilons.append(0.0)
                                        
                                        total_batchloss = total_loss.sum()
                                        optimizer.backward(total_batchloss, retain_graph=False, scaler=scaler)
                                    else:
                                        total_loss = 0.0
                                        total_loss = losses_for_backward[0]
                                        cur_epsilons = []

                                        constraint_values = []
                                        for sid in range(1, len(losses_for_backward)): #the secondary losses or constraints
                                            cur_epsilon = get_epsilon(step, epsilons[sid-1], min_epsilons[sid-1], epsilon_warmup_steps[sid-1], epsilon_cooldown_steps[sid-1], epsilon_decay_functions[sid-1])
                                            constraint_value = (cur_epsilon - losses_for_backward[sid]).detach()
                                            damp = args.dampness * constraint_value
                                            mask = lambda_.get_mask(sid-1, damp)

                                            closs_for_theta = lambda_.get_loss(sid - 1, damp * mask, (cur_epsilon - losses_for_backward[sid]))
                                            total_loss = total_loss - closs_for_theta
                                            
                                            cur_epsilons.append(cur_epsilon)                             
                                            constraint_values.append(constraint_value.item())
                                    
                                        total_batchloss = total_loss.sum()
                                        # print(f"[step{step}] cur_lr {cur_lr}") 
                                        # print(f"[step{step}] total_batchloss.item()", total_batchloss.item())
                                        optimizer.backward(total_batchloss, retain_graph=False, scaler=scaler) ### calculate gradient

                                        # print(f"[step{step}] pred_embeds[0][1].grad.norm(p=2, dim=-1)", pred_embeds[0][1].grad.norm(p=2, dim=-1))

                                    if args.debug and args.debug_gradients == "true":
                                        total_norm = 0
                                        gi=0
                                        for p in outputs.parameters():
                                            gi+=1
                                            param_norm = p.grad.data.norm(2, -1).sum(dim=0)
                                            print("for theta", param_norm)
                                        for p in lambda_.parameters():
                                            print("for lambda", p.grad)
                    
                                ## 23/7/21 - add locate code
                                ## 23/8/14 - moved inside for loop for gradient-based inference
                                if step % args.num_locate_steps == 0:
                                    batch = {"input_ids": pred_tokens}
                                    if args.num_edit_token_per_step == -1: 
                                        print("Editing all tokens")
                                        indices = list(range(len(pred_tokens[0])))
                                    else:
                                        indices = locate(name2model[model_paths[1]], name2tokenizer[model_paths[1]], batch, max_num_tokens=args.num_edit_token_per_step, unit=args.locate_unit, use_cuda=use_cuda)
                                    intermediate_result.update({f"step_{step}_indices": indices}) # save indices along with update results

                                    if indices == [[]]:
                                        print(f"Early stop at @{step} with a loss value of {weighted_loss} since no token is to be edited.")
                                        break
                                
                                ## 08/24/23 add gradient clipping
                                if args.num_project_steps != 1:
                                    torch.nn.utils.clip_grad_norm_(outputs.parameters(), 1)
                                
                                ## 08/17/23: to pass option for projection.
                                project_yn = True if step % args.num_project_steps == 0 else False
                                if logging_outputs[0].get('entropy', None) is not None:
                                    optimizer.step(indices, scaler=scaler, entropy=logging_outputs[0].get('entropy', None), project = project_yn) ### backpropagate
                                else:
                                    optimizer.step(indices, scaler=scaler, project = project_yn) ### backpropagate
                                
                                update_lr_condition = "none"
                                if args.linear_scale != "true" and  len(losses) > 1:
                                    sats = torch.Tensor(constraint_values).ge(0.).to(device)
                                    update_lambda_condition = (step % args.lambda_update == 0)
                                    lambda_mask = float(update_lambda_condition) * torch.ones_like(sats)
                                    
                                    lambda_mask += (1-sats.float()) * (lambda_.is_zero())
                                
                                total_batchlossitem = losses_for_backward[0].item()
                                if dynamic_lambda_update_prev_loss is not None and abs(total_batchlossitem - dynamic_lambda_update_prev_loss) <= 1e-6:
                                    repeat_counts[0] += 1
                                    if args.linear_scale != "true" and  len(losses) > 1 and args.dynamic_lambda_update:
                                        lambda_mask = (1 - sats.float())

                                    if args.dynamic_lr_update and best_allsat[0] is not None and best_allsat[0]:
                                        update_lr_condition = "increase"
                                else:
                                    repeat_counts[0] = 1
                                
                                dynamic_lambda_update_prev_loss = total_batchlossitem

                                if update_lr_condition == "increase":
                                    cur_lr = optimizer._optimizer.update_lr(min(cur_lr + args.lr_update_size, args.max_lr))

                                if args.linear_scale != "true" and len(losses) > 1:
                                    optimizer_lambda._optimizer.set_mask(lambda_mask.clamp(max=1.0, min=0.0))
                                    optimizer_lambda.step()
                                    lambda_.make_positive()
                        
                                    
                                gc.collect()

                                #[0] is batch index, batch size in our case in 1 always so it doesn't matter.
                                weighted_losses = []
                                weighted_loss = 0.0
                                for beta, lossval in zip(betas, losses_for_backward):
                                    weighted_loss = weighted_loss + beta * lossval[0].item()     
                                weighted_losses.append(weighted_loss)
                                
                                ## added 23/09/04
                                primary_loss = losses_for_backward[0][0].item()
                                attribute_index = 1 # need to modify this part for the case where there're more than one constraints
                                attribute_loss = losses_for_backward[attribute_index][0].item()
                                ## done addition
                                
                                constrained = []
                                allsat = True
                                for i in range(1, len(losses)):
                                    if losses_for_backward[i] <= min_epsilons[i - 1]:
                                        constrained.append("sat")
                                    else:
                                        constrained.append("vio")
                                        allsat=False
                                
                                if args.show_all_outputs and len(losses) > 1 and allsat:
                                    best_prediction_set[0].add(target_sents[0])
                                    
                                constrained = ",".join(constrained)
                                intermediate_result.update({f"step{step}_best_loss": best_loss[0],f"step{step}_loss0": losses_for_backward[0].item(), f"step{step}_loss1": losses_for_backward[1].item(), f"step{step}_allsat": allsat, f"step{step}_repeat_counts[0]": repeat_counts[0]})
                                # print(f"[step{step}] best_loss[0]: {best_loss[0]}, weighted_loss: {weighted_loss}, losses_for_backward[0]: {losses_for_backward[0].data.cpu()}, losses_for_backward[1]: {losses_for_backward[1].data.cpu()}, min_epsilons for 1st constraint: {min_epsilons[0]}, allsat: {allsat}, repeat_counts[0]: {repeat_counts[0]}")
                                
                            
                                # modify_condition =\
                                #     args.selection_criterion == "last" or\
                                #     (best_loss[0] is None and args.selection_criterion == "weighted_sum") or\
                                #     (best_loss[0] is not None and args.selection_criterion == "weighted_sum" and best_loss[0] > weighted_loss)
                                
                                # if not modify_condition and args.selection_criterion == "mrr_allsat":
                                #     modify_condition =\
                                #         (best_loss[0] is None and allsat and repeat_counts[0] == 2) or\
                                #         (best_loss[0] is not None and best_allsat[0] and allsat and repeat_counts[0] == 2)

                                # elif not modify_condition and args.selection_criterion == "primary_allsat":
                                #     modify_condition =\
                                #         (best_loss[0] is None and allsat) or\
                                #         (best_loss[0] is not None and not best_allsat[0] and allsat) or\
                                #         (best_allsat[0] and allsat and best_loss[0] > weighted_loss)
                                
                                # always update best_xxx if step = 1 (initialize best_xxx after 1st update step)
                                modify_condition = (step == 1)
                                
                                if not modify_condition:
                                    if (step == 0):
                                        modify_condition = False # if step == 0, same as AR prediction. don't update!
                                    elif args.selection_criterion == "allsat":
                                        modify_condition = (not best_allsat[0] and not allsat and weighted_loss < best_loss[0]) or \
                                                        (not best_allsat[0] and allsat) or \
                                                        (best_allsat[0] and allsat and primary_loss < best_losses[0][0])
                                    elif args.selection_criterion == "attribute":
                                        modify_condition = (attribute_loss < best_losses[attribute_index][0])
                                    elif args.selection_criterion == "weighted_sum":
                                        modify_condition = (weighted_loss <  best_loss[0])
                                
                                if modify_condition:
                                    if args.dynamic_lr_update:
                                        print("resetting the learning rate and noise std, a constraint has been satisfied")
                                        cur_lr = optimizer._optimizer.update_lr(args.lr)
                                        optimizer._optimizer.set_begin_std(0.01) #CHECK
                                    if args.selection_criterion != "last":
                                        print(f"modify condition @{step}", time.time()-starttime, end="\n")
                                    best_loss[0] = weighted_loss
                                    best_allsat[0] = allsat
                                    best_repeat_count[0] = repeat_counts[0]
                                    for i in range(len(losses)):
                                        best_losses[i][0] = losses_for_backward[i][0].item()
                                    
                                    best_pred_tokens[0] = pred_tokens[0]
                                    best_index[0] = step
                                    best_constrained = constrained
                                    best_step = step
                                
                                del losses_for_backward

                                ## turn off early_stopping for now (23/09/04)
                                # if args.early_stop_steps > 0: #[0] is batch index, batch size in our case in 1 always so it doesn't matter.

                                #     early_stop_condition =\
                                #         ("allsat" in args.selection_criterion and best_allsat[0]) or\
                                #         (args.selection_criterion == "weighted_sum") or\
                                #         (args.selection_criterion == "last")

                                #     if prev_loss is not None and abs(weighted_loss - prev_loss) <= 1e-6:
                                #         same_loss_count += 1
                                #     else:   
                                #         same_loss_count = 0

                                #     if early_stop_condition and same_loss_count >= args.early_stop_steps:
                                #         print(f"Early stop at @{step} with a loss value of {weighted_loss} and satisfied constraints")
                                #         break
                                #     elif same_loss_count >= args.early_stop_steps + 100:#2 * args.lambda_update:
                                #         print(f"Early stop at @{step} with a loss value of {weighted_loss} and unsatisfied constraints")
                                #         break
                                        
                                #     prev_loss = weighted_loss



                            except KeyboardInterrupt:
                                print("skipping remaining optimizing steps and showing the best option so far")
                                broken=True
                                break
                        ## gradient update done.


                        if args.time:
                            r = time.time()-starttime
                            print(r)
                            avg_time += r
                        
                        ## after doing all optim_steps for the sample
                        predictions = []
                        prediction_idss = []
                        broken_skip = False
                        skip_printing = False
                        
                        #[0] is batch index, batch size in our case in 1 always so it doesn't matter.
                        ## 23/9/4 removed looping over batches 
                        # for b, item in enumerate(best_pred_tokens):
                        item = best_pred_tokens[0]
                        if item is None and broken:
                            skip_printing = True
                            if broken:
                                broken_skip=input("Skip this input entirely? yes(y)/no(continue)/press ctrl+c to exit")
                                broken_skip = broken_skip.lower() == "y"
                                break
                        ## 23/9/4 commented out falling back to beam search for now
                        # if (args.only_mucoco == "false" and not best_allsat[0]) or (item is None): #item is none happens when optimization fails
                        #     prediction_ids = ", ".join([str(idx) for idx in AR_predicted_indices[0].tolist()])
                        #     prediction_indices = AR_predicted_indices[0].tolist()
                        #     prediction = AR_prediction

                        #     lossvalue = 0.0
                        #     for lossid in range(len(betas)):
                        #         lossvalue += betas[lossid] * predictedlosslists[-1][lossid][0] # VERIFICATION NEEDED
                        #     print(f"best prediction is from beam search, all constraints were not satisfied, allsat={lengthwise_best_prediction[0][2]}")
                        # else:
                        ## 23/9/4 added the item is not None condition b/c if not error occurs when the run early stops for a sample for whom no indices are located.
                        ## e.g. Early stop at @0 with a loss value of 72.02294921875 since no token is to be edited.
                        if item is not None: 
                            prediction_ids = ", ".join([str(x) for x in target_prefix[0].tolist()])
                            prediction_ids +=   f'[{", ".join([str(x) for x in item.tolist()])}]'
                            prediction_indices = target_prefix[0].tolist() + item.tolist()
                            
                            targets = clean_output(item.tolist(), primary_tokenizer.eos_token_id, allow_first_eos=losses[0] == "bart")#, prompt=source_batch[0].unsqueeze(0), sentence_complete=True, lossfn=lossfns[0])
                            if args.target_tokenize_different:
                                with primary_tokenizer.as_target_tokenizer():
                                    prediction = primary_tokenizer.decode(target_prefix[0].tolist() + targets)
                            else:
                                prediction = primary_tokenizer.decode(target_prefix[0].tolist() + targets)

                            print("best prediction at step",best_index[0])
                            lossvalue = best_loss[0]
                            losses_value = [x[0] for x in best_losses]

                            ## modify_condition에 해당 할때에만 AR_prediction으로 initialize되어 있던 lengthwise_best_prediction을 업데이트 함.
                            ## -> 항상 locate & edit 결과로 덮어 쓰게끔 단순화하기.
                            # modify_condition =\
                            #     lengthwise_best_prediction[0] is None or\
                            #     (args.selection_criterion == "weighted_sum" and lengthwise_best_prediction[0][1] > lossvalue)
                            
                            # if not modify_condition and args.selection_criterion == "primary_allsat":
                            #     modify_condition =\
                            #         (not lengthwise_best_prediction[0][2] and best_allsat[0]) or\
                            #         (lengthwise_best_prediction[0][2] and best_allsat[0] and lengthwise_best_prediction[0][1] > lossvalue)
                            
                            # elif not modify_condition and args.selection_criterion == "mrr_allsat":
                            #     modify_condition =\
                            #         (not lengthwise_best_prediction[0][2] and best_allsat[0] and best_repeat_count[0] >= 2) or\
                            #         (lengthwise_best_prediction[0][2] and lengthwise_best_prediction[0][4] >= 2 and lengthwise_best_prediction[0][1] > lossvalue)
                                
                            
                            # if modify_condition:
                            if args.debug:
                                print("modify condition satisfied", end="\n")
                            else:
                                outallsatf.write("modify_condition satisfied ")
                            lengthwise_best_prediction[0] = (prediction, lossvalue, best_allsat[0], prediction_indices, best_repeat_count[0], losses_value)
                            intermediate_result.update({"best_step": best_index[0], 
                                                "best_prediction": prediction
                                                })
                            
                            prediction_idss.append(prediction_ids)
                            predictions.append(prediction)

                            # if args.debug and not skip_printing:                    
                            #     for i, item in enumerate(best_pred_tokens):
                            #         print(f"predicting length: {sent_length}")
                            #         print("Given source:", source_text)
                            #         print("Given target: ", target_text)
                            #         print("Given additional: ", additional_text)
                            #         print(f"Prediction ids: {prediction_ids}")
                            #         print(f"Prediction: {prediction}")
                            #         print("All generations that satisfied the constraints: ", best_prediction_set[i])

                            #         out = []
                            #         for lossid in range(len(losses)):
                            #             out.append(f"{losses[lossid]}: {best_losses[lossid][i]}")
                            #         print("; ".join(out))
                                
                                
                            #     if broken:
                            #         broken_skip=input("Skip this input entirely? yes(y)/no(continue)/press ctrl+c to exit")
                            #         broken_skip = broken_skip.lower() == "y"

                            all_stepcounts += best_index

                            optimizer.zero_grad(set_to_none=True)
                            del outputs
                            del optimizer
                            if len(losses) > 1:
                                optimizer_lambda.zero_grad()
                                del optimizer_lambda
                                del lambda_
                            for modelname in loss2modelname.values():
                                name2model[modelname].zero_grad(set_to_none=True) 
                            torch.cuda.empty_cache()
                            
                            ## 23/09/04 no longer needed since the for b in range(len(batch_size)) type of loop is not removed.
                            # if args.debug and broken_skip: 
                            #     break
                            
                            # if break_after:
                            #     break
                        
                    ### RESTART HERE (check if convergence failed and restart if it failed & restart_index < restarts)
                    b=0
                    if lengthwise_best_prediction[b] is None or not lengthwise_best_prediction[b][2]: #constraints are not satisfied
                        if restart_idx < args.restarts: #atleast one more restart is left
                            continue #skip printing and loop over
                        elif lengthwise_best_prediction[b] is None:
                            lengthwise_best_prediction = [("", -1, False, [], -1, -1)] #just blank which didn't satisfy the constraints

                    if args.debug:
                        if not skip_printing:
                            for b in range(batch_size):
                                print("sample #"+str(sample_idx), f"repeat count: {lengthwise_best_prediction[b][4]}" , "best prediction for all lengths: ", lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                    else:   
                        if args.output_style == "text":
                            for b in range(batch_size):
                                outf.write(lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                                outf.flush()
                                outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n")
                                outallsatf.flush()
                        else:
                            if sample_idx == 0:
                                output = {
                                    "prompt":{
                                        "text":source_text,
                                        "tokens":source_indices_write}, 
                                    "generations":[{
                                        "text": lengthwise_best_prediction[b][0],
                                        "tokens": lengthwise_best_prediction[b][3],
                                        "allsat": lengthwise_best_prediction[b][2],
                                        "losses": lengthwise_best_prediction[b][5],
                                        "weighted_loss":lengthwise_best_prediction[b][1],
                                        "repeat_count": lengthwise_best_prediction[b][4],
                                        "mucoco": True
                                        }]
                                }
                            else:
                                output['generations'].append(
                                    {
                                        "text": lengthwise_best_prediction[b][0],
                                        "tokens": lengthwise_best_prediction[b][3],
                                        "allsat": lengthwise_best_prediction[b][2],
                                        "losses": lengthwise_best_prediction[b][5],
                                        "weighted_loss":lengthwise_best_prediction[b][1],
                                        "repeat_count": lengthwise_best_prediction[b][4],
                                        "mucoco": True
                                    }
                                )
                            
                            if sample_idx + 1 == args.num_samples:
                                json.dump(output, outf)
                                outf.write("\n")
                                outf.flush()

                                outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n")
                                outallsatf.flush()
                                #VERIFY
                            intermediate_result.update({"time": time.time() - start_time})
                            output2 = intermediate_result
                            json.dump(output2, outf2)
                            outf2.write("\n")
                            outf2.flush()
                    print(f"required output achieved or number of restarts ran out at attempt #{restart_idx+1}")
                    break # don't restart if already reached here

                else: # skipping mucoco and writing beam search output 
                    lengthwise_best_prediction[0] = [(AR_prediction, total_weighted_loss, predicted_allsat, predicted_batch[0].tolist(), -1, predictedlosses)]
                    if ask_skip != "y":
                        if args.debug:
                            print("Skipping this example. the beam search output already satisfies all the constraints or there's no constraints")
                            for b in range(batch_size):
                                print("best prediction for all lengths: ", lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                        else:
                            print("Skipping this example. the beam search output already satisfies all the constraints or there's no constraints")
                            if args.output_style == "text":
                                for b in range(batch_size):
                                    outf.write(lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                                    outf.flush()
                                    outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n")
                                    outallsatf.flush()
                            else:
                                for b in range(batch_size):
                                    if sample_idx == 0:
                                        output = {
                                            "prompt":{
                                                "text":source_text,
                                                "tokens":source_indices_write}, 
                                            "generations":[{
                                                "text": lengthwise_best_prediction[b][0],
                                                "tokens": lengthwise_best_prediction[b][3],
                                                "allsat": lengthwise_best_prediction[b][2],
                                                "losses": lengthwise_best_prediction[b][5],
                                                "weighted_loss":lengthwise_best_prediction[b][1],
                                                "mucoco": False
                                                }]
                                        }
                                        # print(output)
                                    else:
                                        output['generations'].append(
                                            {
                                                "text": lengthwise_best_prediction[b][0],
                                                "tokens": lengthwise_best_prediction[b][3],
                                                "allsat": lengthwise_best_prediction[b][2],
                                                "losses": lengthwise_best_prediction[b][5],
                                                "weighted_loss":lengthwise_best_prediction[b][1],
                                                "mucoco": False
                                            }
                                        )
                                
                                if sample_idx + 1 == args.num_samples:
                                    json.dump(output, outf)
                                    outf.write("\n")
                                    outf.flush()
                                    #VERIFY
                    intermediate_result.update({"time": time.time() - start_time})
                    output2 = intermediate_result
                    json.dump(output2, outf2)
                    outf2.write("\n")
                    outf2.flush()
                    break # don't restart
            
                if args.debug and broken_skip:
                    break

            if args.debug and broken_skip: 
                break

            del source_batch
            del target_batch
            del additional_batch
            del for_predicted_source_batch
            del predicted_batch
            source_batch = []
            target_batch = []
            for_predicted_source_batch = []
            additional_batch = []
            predicted_batch = []
            context_batch = []

    if args.outfile is not None:
        outf.close()
        outallsatf.close()
        outf2.close()
    print("average numbers of steps to converge =", np.mean(all_stepcounts))
    print("average time = ", avg_time/c)

def prune(sentence):
    pass 

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
    for i, tok in enumerate(tokens):
        if tok == eos_token_id and (not allow_first_eos or i > 0):
            break
        
        if (tok not in skip_special_tokens):
            new_tokens.append(tok)
        
    if return_tensors:
        return torch.LongTensor([new_tokens])
    return new_tokens
    
def cli_main():
    parser = options.get_parser()
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    
    args = joblib.load('/home/hyeryung_son/mucoco/outputs/toxicity/locate-edit-gpt2-loc-6toks--1steps-project-5steps-mrr_allsat/outputs_epsilon-3.txt.pkl')
    args.optim_steps = 10
    args.num_project_steps = 2
    main(args)