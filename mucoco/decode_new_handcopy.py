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
import joblib

from transformers import AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer # lib for embeddings

from mucoco.utils import TargetProbability, TargetEmbeddings, TargetSimplex, Lambda, Optimizer, OptimizerLE, get_epsilon
import mucoco.losses as lossbuilder
import mucoco.options as options
import mucoco.utils as utils
import torch.nn.functional as F

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    
    """
    Override logging levels of different modules based on their name as a prefix.
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })') ## 정확히 어떤 걸 매칭하는지 잘 모르겠는데...
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

def sentence_completion(prompt, tokens, lossfn):
    
    # max output length를 10개 늘린다.
    lossfn.args.max_output_length = lossfn.args.max_output_length + 10
    
    # 왠지는 모르겠지만 ㅠ prompt랑 tokens를 붙인 다음에 나머지를 생성하게 함 ... 흠.... prompt와 token 뒤에 10개를 더 생성하라 이런건가?
    new_tokens = lossfn.generate(torch.cat([prompt, torch.LongTensor([tokens]).to(lossfn.device)]))
    
    # 다시 max_output_length를 원복한다.
    lossfn.args.max_output_length = lossfn.args.max_output_legnth - 10

    return tokens + new_tokens[0].tolist()
    
def clean_output(tokens, eos_token_id, return_tensors=False, allow_first_eos=False, skip_special_tokens=[], prompt=None, sentence_complete=False, lossfn=None):
    """
    skip_special_tokens를 제외한 token만 리턴.
    eos로 시작하면 []를 리턴
    """
    
    if sentence_complete: # ??? 어떨 때 쓰는 옵션일까?
        tokens = sentence_completion(prompt, tokens, lossfn)
    new_tokens = []
    for i, tok in enumerate(tokens):
        if tok == eos_token_id and (not allow_first_eos or i > 0):
            break # 만약에 첫번째 토큰이 eos 이면 그냥 [] 를 리턴
        
        if (tok not in skip_special_tokens): # special_token이 아닌것만 골라내겠다가 골자임.
            new_tokens.append(tok)
            
    if return_tensors:
        return torch.LongTensor([new_tokens])
    return new_tokens
    
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
        outf = open(args.outfile, "w")
        outallsatf = open(args.outfile + ".allsat", "w")
        outf2 = open(args.outfile + ".intermediate", "w")
        
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    use_cuda = torch.cuda.is_available() and not args.cpu
    
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
    
    # ??? 왜 나중에 어짜피 다시 설정할거를 여기서 이렇게 하는걸까?...
    # 그냥 initialize하는거라고 생각하면 될까??ㅠ.ㅠ
    if args.keywords is None or args.keywords == "none":
        keywords = ["the" for _ in losses]
    elif args.keywords in ["_roc_", "_commongen_", "_commongenunique_"]:
        keywords = ["" for _ in losses]
    else:
        keywords = args.keywords.split(":")
        if len(keywords) == 1:
            keywords = [f"_topic_:{args.keywords[0]}" for _ in losses]

    # options related to how to select final output.
    # what's beta? 내 예상으로는,,, lambda? nontoxic 일때의 
    # nontoxic일 때 selection_criterion='mrr_allsat' 임, betas='0.8:0.2' 임.
    # !!! 그러면 결국에는 betas = [1.0, 0.0] 인거임.
    if "allsat" in args.selection_criterion: # allsat = all satisfied? 
        # with this flag, the output which minimized the primary objective while satisfying all objectives is selected. In case all constraints are not satisfied (e.g when constraints are competing or optimization fails), this will predict the default output (Using an autoregressive decoding setup: beam search in this case)
        betas = [1.0] + [0.0 for _ in range(len(losses)-1)]
    elif (args.selection_criterion == "weighted_sum" and args.betas is not None) or (args.selection_criterion == "last"):
        betas = [float(beta) for beta in args.betas.split(":")]
    else:
        raise ValueError("correct selection_criterion or betas needs to be specified")
    
    # omit assert statemments
    
    prev_vocab_size = None
    vocab_size = None
    primary_vocab_size = None
    
    # start for loop - model, tokenizer, embed_lut 읽어오는 부분
    for i, model_path in enumerate(model_paths):
        if model_path not in name2model:
            name2tokenizer[model_path] = AutoTokenizer.from_pretrained(tokenizer_paths[i], cache_dir = args.cache_dir, use_fast=True)
            name2config[model_path] = AutoConfig.from_pretrained(model_path, cache_dir=args.cache_dir)
            
            # model을 읽어오는데, loss 모듈의 model wrapper를 이용한다는 점이 좀 코드가 어려워보이게 만든다.
            if model_types[i] == "sentence-transformer":
                name2model[model_path] = lossbuilder.ModelWrapper(SentenceTransformer(model_path))
            elif "Custom" in model_types[i]:
                name2model[model_path] = lossbuilder.ModelWrapper(getattr(utils, model_types[i]).from_pretrained(model_path, config=name2config[model_path], cache_dir = args.cache_dir))
            else:
                name2model[model_path] = lossbuilder.ModelWrapper(getattr(transformers, model_types[i]).from_pretrained(model_path, config=name2config[model_path], cache_dir=args.cache_dir))
            # getattr: string을 통해서 class의 member를 가져온다. string으로 member를 접근할 수 있는 장점. 코드를 간결화할 수 있다. # https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=siniphia&logNo=221796316521
            
            if not args.show_warnings:
                
                set_global_logging_level(logging.ERROR, [name2model[model_path].__module__]) 
                # __module__: The name of the module the function was defined in, or None if unavailable.
                # https://stackoverflow.com/questions/10113892/semantics-of-module
                
            name2model[model_path].eval() # set as eval mode. turn off dropout ...
            
            # new_vocab_size를 저장하기 위해서 하는 step
            embed_lut_ = name2model[model_path].get_input_embeddings() # transformers pretrained model class 의 method
            if isinstance(embed_lut_, torch.nn.Sequential):
                new_vocab_size = embed_lut_[0].num_embeddings
            else:
                new_vocab_size = embed_lut_.num_embeddings
                
            # prev_vocab_size -> 여러개의 model을 쓰는 경우, for loop 의 이전 iteration의 model의 vocab_size
            # loss에 사용되는 모든 모델이 같은 vocabulary size를 가지도록 하는 step
            if prev_vocab_size is None:
                vocab_size = new_vocab_size
            if new_vocab_size != prev_vocab_size and prev_vocab_size is not None:
                if not args.allow_diff_vocab:
                    raise ValueError(f"all models should have the same vocabulary {new_vocab_size} != {vocab_size}")
                else:
                    logger.warning("all models don't have the same vocabulary and we are still proceeding")
                
            prev_vocab_size = vocab_size
            
        # 만약 decoder와 encoder의 tokenizer가 다르면, decoder의 input_embedding 사용. 아니면 encoder의 input_embedding 사용.
        if args.target_tokenize_different:
            embed_luts.append(name2model[model_path].get_decoder().get_input_embeddings())
        else:
            input_embeds = name2model[model_path].get_input_embeddings()
            if isinstance(input_embeds, torch.nn.Sequential):
                input_embeds = input_embeds[0]
            embed_luts.append(input_embeds)
            
        # 만약에 target_type 은 잘은 모르지만,,
        # update하는 대상이 embedding인지, simplex인지, probability인지 인것 같다.
        # 근데 왜 target_type이라고 부른걸까? 그리고, simplex는 정확히 뭘까?
        if args.target_type == "embeds":
            embed_luts[-1].requires_grad = False # embedding weights에 대해서, gradient descent를 하지 않으려는 걸까?
            
        if i == 0 :
            primary_vocab_size = vocab_size
            primary_embed_dim = embed_luts[-1].embedding_dim
            
        # I'm not sure but, for some transformer-based models, authors scale token embeddings before adding with positional embedding... 
        # https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod
        if getattr(name2model[model_path], "get_decoder", None) is None:
            embed_scales.append(1.0)
        else:
            embed_scales.append(getattr(name2model[model_path].get_decoder(), "embed_scel", 1.0))
    # end for loop
    
    if use_cuda:
        for name, model in name2model.items():
            model.cuda()
    
    # losses - models 는 1:1 매핑이 된다. 
    # loss를 build 해서 lossfns에 추가한다.
    lossfns = []
    for i, loss in enumerate(losses):
        lossfns.append(lossbuilder.build_loss(loss, name2model[model_paths[i]], name2tokenizer[model_paths[i]], args))
        loss2modelname[loss] = model_paths[i]
        loss2tokenizer[loss] = name2tokenizer[model_paths[i]]
    primary_tokenizer = loss2tokenizer[losses[0]]

    if args.model_dtype == "fp16":
        for name, model in name2model.items():
            model.half()
    
    # mucola 논문에서 epsilon과 lambda는 constraint 마다 존재한다.
    # 경제학에서 생각해보면, epsilon은 주어진 "예산"이고, lambda는 예산이 1 증가할 때, optimal 효용이 몇 단위 증가하는지이다.
    # quote : 람다(λ) 값은 경제학적 의미에서 뭐냐? 람다는 예산제약이 1단위 증가할 때 목적함수의 최적값이 얼마나 증가하는지를 알려줍니다. 즉, 예산이 1원(또는 1달러) 증가할 경우, 목적함수인 효용함수의 최적값(optimal utility)이 λ단위만큼 증가한다는 의미입니다.
    # https://electronicsdo.tistory.com/entry/Lagrangian-%EB%9D%BC%EA%B7%B8%EB%9E%91%EC%A7%80%EC%96%B8
    if args.epsilons is not None and args.epsilon != "none":
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
        min_epsilons = [eps + getattr(lossfns[i+1], "epsilon_additive", 0) for i, eps in enumerate(min_epsilons)]
        
    else:
        epsilons = []
        min_epsilons = []
        decay_function = []
        epsilon_warmup_steps = []
        epsilon_colldown_steps = []
    
    source_dataset = None
    target_dataset = None
    additional_dataset = None # additional이 들어가는 거는 hard constraint(lexical constraint) 관련이라 생각하기
    args.use_context = args.use_context == "true"
    # setting data paths
    if args.data is not None:
        data_paths = args.data.split(":")
        if len(data_paths) == 1:
            source_data = data_paths[0]
            target_data = data_paths[0] # not used
            context_data = data_paths[0] # not used
        elif len(data_paths) == 2:
            source_data = data_paths[0]
            target_data = data_paths[1] # for debugging
            context_data = data_paths[1] # not used
        else:
            source_data = data_paths[0]
            target_data = data_paths[1] # for debugging
            context_data = data_paths[2] # tsv file
        
        additional_data = args.additional_data
        if additional_data is None or additional_data == "none":
            additional_data = source_data # used for STRAP example (Krishna et al 2020)
    
    elif args.additional_data is not None and additional_data != "none":
        source_data = args.additional_data
        target_data = args.additional_data
        additional_data = args.additional_data
    
    else:
        source_dataset = sys.stdin
        start_idx = 0
        end_idx = 1000000
        
    if source_dataset is None: # 위 if else 의 마지막 케이스가 아닌 경우. # 옵션에 따라서, 다르게 생긴 데이터셋을 읽어오는 코드를 실행
        if args.datastyle == "text":
            source_dataset = [l.strip() for l in open(source_data)]
            target_dataset = [l.strip() for l in open(target_data)]
            context_dataset = []
            import csv
            with open(context_data) as csvfile:
                reader = csv.reader(csvfile, delimiter="\t")
                for row in reader:
                    context_dataset.append(row)
            additional_dataset = [l.strip() for l in open(additional_data)]
        elif args.datastyle == "jsonl":
            source_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(source_data)]
            target_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(target_data)]
            additional_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(additional_data)]
            
            # dataset 형태가 nested 된 json 일 경우에.. (e.g. {"contents": {"text": ...}})
            if args.jsonl_secondary_key is not None and args.jsonl_secondary_key != "none":
                source_dataset = [x[args.jsonl_secondary_key] for x in source_dataset]
                target_dataset = [x[args.jsonl_secondary_key] for x in target_dataset]
                additional_dataset = [x[args.jsonl_secondary_key] for x in additional_dataset]
            
            context_dataset = [None] * len(source_dataset)
            if args.use_context:
                context_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(context_data)]
                if args.jsonl_secondary_key is not None and args.jsonl_secondary_key != "none":
                    context_dataset = [x[args.jsonl_secondary_key] for x in context_dataset]
        elif args.datastyle == "single-jsonl": # one jsonl file has all the information 
            source_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(source_data)]
            target_dataset = [json.loads(l)[args.jsonl_secondary_key] for l in open(target_data)]
            additional_dataset = [json.loads(l)[args.jsonl_secondary_key] for l in open(additional_data)]
            
            context_dataset = [None] * len(source_dataset)
            if args.use_context:
                context_dataset = [[json.loads(l)[args.jsonl_secondary_key]] for l in open(context_data)] #meaningful
        
        # 읽어온 데이터에서 몇번째 샘플부터 몇번째 샘플까지 처리할것인지 결정
        start_idx = args.start_idx 
        end_idx = (len(source_dataset) + args.end_idx) % len(source_dataset) 

    # batch를 initialize 함
    source_batch, target_batch, additional_batch, for_predicted_source_batch, predicted_batch, context_batch = [], [], [], [], [], []
    batch_size = args.batch_size 
    
    device = "cuda" if use_cuda else "cpu"
    c = 0 # counting # of examples that are subject to constrained decoding.
    
    # initialize lists to keep track of losses
    losslists = [[] for _ in range(len(losses))] 
    predictedlosslists = [[] for _ in range(len(losses))]
    source_primarylosslist = []
    all_stepcounts = []
    avg_time = 0
    
    # 이것이 무엇일까..ㅎㅎ // len(losses) -1 의 개수만큼의 "true" or "false"
    # if "true", then use the base lm loss as the epsilon for loss i
    if args.gold_loss_epsilons is not None and args.gold_loss_epsilons != "none":
        args.gold_loss_epsilons = args.gold_loss_epsilons.lower().split(":")
        assert len(args.gold_loss_epsilons) == len(losses) - 1
    else:
        args.gold_loss_epsilons = ["false" for _ in range(len(losses) - 1)]
    
    # 모든 test set을 처리하지 않고, 일부만 테스트하기 위해서 아래를 정의
    # args.random_example -> 쓰이지 않음
    # args.num_examples -> 0 보다 큰 숫자이면, args.num_examples 만큼만 테스트하도록 함
    example_p = 1.0
    args.random_example = args.random_example == "true" ## 딱히 어디에 쓰이지 않는 옵션임. 그냥 default로 sample 함..
    if args.num_examples > 0 and target_dataset is not None: # source_dataset이 sys.stdin이 아닌 경우에
        example_p = args.num_examples * 1.0 / len(source_dataset)
    
    for text_id, source_text in enumerate(source_dataset):
        
        # 지정한 start_idx, end_idx 사이의 샘플만 사용
        if text_id < start_idx or text_id > end_idx: 
            continue 
        
        # 실제로 test 한 샘플개수가 c 인데, 그것이 num_examples 와 같으면 중지
        if args.num_examples > 0 and c > 0 and c == args.num_examples:
            break
        
        # 전체 test sample 중에서 example_p 의 확률로 test함
        # np.random.rand() : uniform distribution over [0, 1]
        do_this_example = np.random.rand() <= example_p
        if not do_this_example:
            continue 
        
        # test한 샘플 수를 1개 늘림
        c += 1 
        
        # lossfns[lossid].compute_gold_loss() 할때 kweight으로 인자로 들어감.
        # kweight이라는 옵션이 퀵하게 보기로는 안쓰이는 값같다.
        new_kweight = args.kweight 
        if target_dataset is not None: # sys.stdin 에서 source_dataset을 받아오는것이 아닌 경우
            target_text = target_dataset[text_id]
            additional_text = additional_dataset[text_id]
            context_texts = context_dataset[text_id]
            
            # 따로 option 들을 sys.stdin 에서 받을 필요가 없는 경우임
            # nontoxicity 의 경우,
            # args.init = "target" ## option의 종류: "zeros", "random", "source", "target", "targettarget", "random_vocab", "embedgd-zeros"
            # args.max_output_length = 20
            # args.max_length = 20
            # args.use_context = "false"
            # ??? 계속해서 "source" 와 "target"의 개념이 나오는데,, 이것들은 다 뭘까?...
            
        else: # source_text갸 sys.stdin과 같은 경우 ??? 
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

        # 잘은 모르는데, hard constraint, 즉 특정 단어를 꼭 넣어야 하는 loss에 쓰이는 옵션 같다.
        if args.keywords == "_roc_": # roc -> dataset 이름
            keywords = ["none"] + additional_text.split(", ")
        if args.keywords == "_rocunique_": # ??? unique가 붙었을때 뭐가 달라지는건지 모름.
            keywords = ["none"] + additional_text.split(", ") + ["none"]
        elif args.keywords == "_commongen_":
            keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#")
        elif args.keywords == "_commongenunique_":
            keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#") + ["none"]
        
        # debugging 할 때 썼던 코드인듯
        early_skip="n"
        if args.debug:
            early_skip = input(f"skip this example? {source_text} [yes(y)/maybe(m)/no(n)]")
            if early_skip == "y":
                continue
        
        if not args.jsonl_tokenized: # 대충 guess 하기로는 jsonl 파일이 이미 tokenize 된 채로 저장된게 아니면
            if source_text == "": # prompt가 없는 경우라면
                source_text = primary_tokenizer.bos_token # bos로 시작하고
            source_indices = primary_tokenizer.encode(source_text, return_tensors="pt").to(device) # source_text를 encode
            source_indices_write = source_indices[0].tolist()
            
            additional_indices = primary_tokenizer.encode(additional_text, return_tensors="pt", add_special_tokens=False).to(device)
            
            eos_token_id = primary_tokenizer.eos_token_id
            bos_token_id = primary_tokenizer.bos_token_id
            
            context_indices= None ## 보통 context란 말은 QA 할 때 쓰는거같은데, 여기서는... nontoxic generation 일 때는 잘 안쓰는 말같다.
            if args.target_tokenize_different:
                with primary_tokenizer.as_target_tokenizer():
                    eos_token_id = primary_tokenizer.eos_token_id
                    bos_token_id = primary_tokenizer.bos_token_id
                    if args.use_context:
                        context_indices = primary_tokenizer.encode(context_texts[0], return_tensors="pt")
            elif args.use_context:
                context_indices = primary_tokenizer.encode(context_texts[0], return_tensors="pt")
            
            # mucola 논문에서 seq2seq 모델을 쓴 케이스 -> appendix에 나와있는 entity-controlled summarization task 라는게 있었음!
            # 뭐 잘은 모르겠는데 base LM is seq2seq 이면 target_tokenize_different setting이 True 여야 하나봄........ 이게 무엇이냐........
            if not args.target_tokenize_different and "Seq2SeqLM" in model_paths[0]:
                logger.warning("you are using a seq2seq model for your primary loss but not tokenizing the target sentences with a different target tokenizer.")

            if args.target_tokenize_different:
                with primary_tokenizer.as_target_tokenizer():
                    for_predicted_source_indices = primary_tokenizer.encode(source_text, return_tensors="pt")
                    target_indices = primary_tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False).to(device)
            else:
                for_predicted_source_indices = source_indices
                target_indices = primary_tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False).to(device)
        else: # in case jsonl is already tokenized!
            source_indices_write = source_text # ..._text is already indices
            source_indices = source_text
            target_indices = target_text
            additional_indices = additional_text
            context_indices = context_texts # ??? context는 뭘까?
            if len(source_indices)==0: # if there's no prompt.
                source_indices.append(primary_tokenizer.bos_token_id)

            source_indices = torch.LongTensor([source_indices]).to(device) # tensor로 바꿈
            additional_indices = torch.LongTensor([additional_indices]).to(device) # tensor로 바꿈
            
            # for_predicted_source_indices => used for style transfer. 
            for_predicted_source_indices = source_indices
            target_indices = torch.LongTensor([target_indices]).to(device) # tensor로 바꿈
            
            bos_token_id = primary_tokenizer.bos_token_id
            eos_token_id = primary_tokenizer.eos_token_id
            if args.target_tokenize_different: # 이런 옵션이 만약에 있으면, target tokenizer에 맞춰서 tokenize 해야 함
                with primary_tokenizer.as_target_tokenizer():
                    bos_token_id = primary_tokenizer.bos_token_id
                    eos_token_id = primary_tokenizer.eos_token_id
                    
            source_text = primary_tokenizer.decode(source_indices[0].tolist()) # 오히려 이미 tokenize된 indices를 text로 다시 바꿈
            
        #### tokenize가 끝나고 list에 담기 -> 근데 사실 좀 웃긴게 .. 지금 코드가 어짜피 batch_size == 1 일때만 돌아가가지구, [[ids]] 이런식으로 생긴 형태일 뿐이다.
        source_batch.append(source_indices)
        target_batch.append(target_indices)
        for_predicted_source_batch.append(for_predicted_source_indices)
        additional_batch.append(additional_indices)
        context_batch.append(context_indices)
            
        # 이번 iteration의 text_id, source_text 에 대해서 아래를 실행
        if len(source_batch) == batch_size: ## 약간 이렇게 한 이유는 뭘까? 흠... 뭔가.... 음........모르겠다 ㅎ
            
            source_batch = torch.cat(source_batch, dim=0).to(device)
            target_batch = torch.cat(target_batch, dim=0).to(device)
            additional_batch = torch.cat(additional_batch, dim=0).to(device)
            for_predicted_source_batch = torch.cat(for_predicted_source_batch, dim=0).to(device)
            
            if args.use_context:
                context_batch = torch.cat(context_batch, dim=0).to(device)
                print(context_batch)
            
            # generate AR generations.
            predicted_batches = []
            for batchidx in range(source_batch.size(0)):
                with torch.no_grad(): # model의 gradient는 필요하지 않아서 이렇게 하는걸까?
                    starttime = time.time() # num_samples 개수만큼 AutoRegressive prediction을 만들어낸다.
                    AR_predicted_all = \
                        lossfns[0].generate(
                            input_ids = source_batch[batchidx].unsqueeze(0),
                            additional_ids = additional_batch[batchidx].unsqueeze(0),
                            num_return_sequences = (args.restarts + 1) * args.num_samples
                        )
                    
                    AR_prediction_all = [] # 이름을 엄청 헷갈리게 지었다고 생각하는 부분..
                    for sample_idx in range(len(AR_predicted_all)):
                        AR_predicted_indices = \
                            clean_output(AR_predicted_all[sample_idx].tolist(),
                                         eos_token_id = eos_token_id,
                                         return_tensors = True,
                                         allow_first_eos=losses[0] == "bart",
                                         skip_special_tokens = [bos_token_id, eos_token_id])
                        
                        # clean한 index를 다시 text로 변환
                        if args.target_tokenize_different:
                            with primary_tokenizer.as_target_tokenizer():
                                AR_prediction = primary_tokenizer.decode(AR_predicted_indices[0].tolist())
                        else:
                            AR_prediction = primary_tokenizer.decode(AR_predicted_indices[0].tolist())
                        
                        # decode한 text는 AR_prediction_all 에 저장
                        AR_prediction_all.append(AR_prediction)
                        
                        # indices는 predicted_batches 에 저장
                        predicted_batches.append(AR_predicted_indices.to(device))

            broken_skip = False
            
            for sample_idx in range(args.num_samples):
                
                for restart_idx in range(args.restarts + 1): # args.restarts가 0보다 크면, AR sample도 좀더 많이 만들어둠 (??? 그럴바에 그냥 num_samples를 늘리지왜....)
                    
                    # get one sample from predicted batch
                    predicted_batch = predicted_batches[sample_idx * (args.restarts + 1) + restart_idx]
                    AR_prediction = AR_prediction_all[sample_idx * (args.restarts + 1) + restart_idx]
                    
                    intermediate_result = {"prompt": source_text} # by hayley
                    intermediate_result.update({"sample_id": sample_idx, "original_text": AR_prediction})
                    
                    # initialize some options
                    skip = False
                    #predicted_allsat = False
                    lengthwise_best_prediction = [None] * batch_size # ??? 왜 굳이 batch_size 크기로 initialize 할까?
                    
                    # !!! important notes !!!
                    # !!! 1. AR output의 loss 보다는 좋아야지만 update한 output을 최종 결정한다. 만약에 AR output이 더 좋으면, 그걸로 최종 뱉는다. losses of the autoregressive output: we should perform atleast as well as this. If we don't, we predict this output
                    # !!! 2. 만약에 AR output이 이미 constraint를 만족하면, mucola 단계를 굳이 밟지 않는다. args.always_mucoco 라는 옵션이 true 이면 무조건 mucola 한다. Also, if the autoregressive output already satisfies the constraints, we skip mucoco unless, args.always_mucoco is true
                    predicted_labels = {}
                    total_predicted_loss = 0.0
                    predicted_allsat = True # 이건 진짜 저자가 잘못했다 ㅋ
                    predictedlosses = []
                    
                    for lossid in range(len(losses)):
                        lossname = losses[lossid]

                        predicted_loss, predicted_lo = \
                            lossfns[lossids].compute_gold_loss(
                                (source_batch, target_batch),
                                additional_batch = additional_batch,
                                context_batch = context_bach,
                                use_context = args.use_context,
                                label_id = label_ids[lossid],
                                keyword = keywords[lossid],
                                kweight=new_kweight # 안쓰는 옵션 같다........!!!
                            )
                        
                        predictedlosses.append(predicted_loss.data.cpu()) # primary loss와 constraint loss들을 list로 가지고 있음.
                        predicted_loss = predicted_loss.sum().item()
                        intermediate_result.update({f"original_loss{lossid}": predicted_loss})
                        total_predicted_loss += betas[lossid] * predicted_loss # 이거는 결국에 뭐냐면, gold_loss의 경우, primary_loss 이외에는 weight를 주지 않는다는 의미...!>!>!>

                        if lossid > 0: # primary_loss가 아닐때
                            predicted_allsat = predicted_allsat and (predicted_loss <= min_epsilons[lossid-1])
                        
                        if "label_prediction" in predicted_lo:
                            predicted_labels[lossid] = predicted_lo['label_prediction']
                        else:
                            predicted_labels[lossid] = "NA"
                        
                        if lossid > 0 and args.gold_loss_epsilons[lossid-1] == "true": # primary_loss가 아니고, 해당 loss에 대해서 gold_loss를 epsilon으로 쓰기 원할때
                            min_epsilons[lossid-1] = predicted_loss + getattr(lossfns[lossid], "epsilon_additive", 0)
                            epsilons[lossid-1] = predicted_loss + getattr(lossfns[lossid], "epsilon_additive", 0)
                            
                    predictedlosslists.append(predictedlosses)
                    
                    if args.only_mucoco == "false": # ??? 이거랑 always_mucoco랑 과연 ㅋㅋ 뭐가 다른걸까.. 어쨌든, only_mucoco가 아닌 이상, AR output을 best prediction으로 저장해놓는다.
                        lengthwise_best_prediction = [(AR_prediction, total_predicted_loss, predicted_allsat, predicted_batch[0].tolist(), -1)]
                    
                    skip = predicted_allsat # 만약에 AR prediction이 all satisfying 하면, skip = True 가 된다.
                    
                    definite_skip = False
                    ask_skip = ""
                    if args.debug and early_skip=="m": # debug 모드 이면서 early_skip 일 때 maybe라고 했던 경우...
                        print(f"new example: {source_text}\nautoregressive output: {AR_prediction}")
                        for lossid in range(len(losses)):
                            print(f"{lossabbr[lossid]} for desired label_id({label_ids[lossid]}): {predictedlosslists[-1][lossid]}; predicted label: {predicted_labels[lossid]}")
                        if predicted_allsat:
                            print(f"autoregressive output already satisfies the constraints")
                        ask_skip = input(f"skip this example? [y/n]") # 사용자에게 skip 할지를 물어봄
                        definite_skip = ask_skip == "y" # 사용자가 skip 하라고 하면 skip 함
                        
                    # 사실 뭐.. 위에 skip = prediced_allsat 때문에. skip and predicted allsat을 굳이 두번 체크하지 않고, 그냥 predicted_allsat 만 체크해도 될듯...
                    elif skip and predicted_allsat and (args.always_mucoco == "false"): 
                        # debug모드가 아니고
                        # !!! 만약에 always_mucoco 가 아닌 이상, predicted_allsat이면 skip 함
                        definite_skip = True 
                    
                    # 🌟 IMPORTANT 🌟 #
                    if not definite_skip: # skip 이 아닐 경우 == constrained decoding을 할 경우 
                        # set length_range
                        if (args.max_length is None or args.max_length == -1) and args.init not in ["source", "target"]:
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
                            length_range = [source_batch.size(1)]
                        else:
                            length_range = [args.max_length]
                        
                        for sent_length_ in length_range: # 사실상 1개 length 로만 돌고 있다 지금은.
                            if args.prefix_length > 0:
                                sent_length = sent_length_ - args.prefix_length
                                target_prefix = predicted_batch[:, :args.prefix_length]
                            else:
                                sent_length = sent_length_
                                target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                
                            if sent_length <= 0:
                                continue
                            if sent_length > args.max_allowed_length:
                                old_l = sent_length
                                sent_length = args.max_allowed_length
                                print(f"changed output length to {sent_length} from {old_l} to avoid GPU overflow. This is a temporary solution.")
                            else:
                                print("predicting a sentence length: ", sent_length)
                            
                            if args.target_type == "simplex": # use V sized real vector for each token and apply softmax before output
                                outputs = TargetSimplex(
                                    vocabsize = primary_vocab_size,
                                    sent_length = sent_length,
                                    batch_size = batch_size,
                                    device = device,
                                    temperature = args.decode_temperature, 
                                    st = args.st,
                                    init_value = source_batch[:, 1:-1] if args.init == "source" else None,
                                    random_init = args.init == "random",
                                    do_sample = args.expgd_do_sample,
                                    top_p = args.expgd_top_p,
                                    top_k = args.expgd_top_k,
                                    embed_scales = embed_scales
                                )
                            elif args.target_type == "probs": # use V sized vector which sums to one for each token and apply softmax before output
                                
                                init_value = None
                                break_after = False
                                if args.init == "source":
                                    init_value = source_batch
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after = True
                                elif args.init == "target":
                                    init_value = target_batch
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after = True
                                    
                                outputs = TargetProbability(
                                    vocabsize = primary_vocab_size,
                                    sent_length = sent_length,
                                    batch_size=batch_size,
                                    device = device,
                                    st = args.st,
                                    init_value = init_value,
                                    random_init = args.init == "random",
                                    do_sample = args.expgd_do_sample,
                                    top_p = args.expgd_top_p,
                                    top_k = args.expgd_top_k,
                                    embed_scales = embed_scales,
                                    max_steps = args.optim_steps
                                )
                            elif args.target_type == "embeds":
                                init_value = None
                                break_after = False
                                if args.init == "source": # initialize the target with the source
                                    init_value = embed_luts[0](source_batch)
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after = True
                                elif args.init == "targettarget": # initialize the target with given target
                                    init_value = embed_luts[0](target_batch)
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after = True
                                elif args.init == "target": # initialize the target with the autoregressive output 
                                    init_value = embed_luts[0](predicted_batch)
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after = True
                                elif args.init == "random_vocab": # uniformly and with replacement sample token from the entire vocabulary
                                    random_indices = torch.multinomial(torch.ones(primary_vocab_size,)/primary_vocab_size, num_samples = batch_size * sent_length, replacement = True).view(batch_size, sent_length).to(device)
                                    init_value = embed_luts[0](random_indices)
                                elif args.init == "embedgd-zeros": # initialize with eos_token_ids
                                    if args.target_tokenize_different:
                                        with primary_tokenizer.as_target_tokenizer():
                                            indices = torch.empty((batch_size, sent_length)).long().fill_(primary_tokenizer.eos_token_id).to(device)
                                    else:
                                        indices = torch.empty((batch_size,sent_length)).long().fill_(primary_tokenizer.eos_token_id)
                                    init_value = embed_luts[0](indices)
                                elif args.init == "zeros":
                                    indices = torch.zeros((batch_size, sent_length)).long().to(device)
                                    init_value = embed_luts[0](indices)
                                
                                final_bias = None
                                if args.final_bias:
                                    final_bias = lossfns[0].model.final_logits_bias
                                
                                outputs = TargetEmbeddings( # 솔직히 말하면 졸리기도 하고, 이게 정확히 어떤건지 모르겠다. 그냥 embedding을 하는건가?
                                    embed_dim = primary_embed_dim, # 
                                    embed_lut = embed_luts[0], # look up table
                                    sent_length = sent_length, # sentench length
                                    batch_size = batch_size, # batch size
                                    device = device,
                                    st = args.st, # st 가 뭘까? straight-through. gradient를 계산하기 어려운 스텝이 있다면, forward에는 그 스텝을 적용하지만 backward에는 그 스텝을 적용하지 않음. https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0
                                    init_value = init_value,
                                    random_init = args.init == "random",
                                    sampling_strategy = args.sampling_strategy,
                                    sampling_strategy_k = args.sampling_strategy_k,
                                    embed_scales = embed_scales,
                                    metric = args.metric,
                                    same_embed = args.same_embeds,
                                    final_bias = final_bias,
                                    eos_token_id = primary_tokenizer.eos_token_id
                                )
                            else:
                                raise ValueError("Wrong target_type")
                            
                            # 제약조건이 있을 경우, Lambda 객체를 정의해서 쓴다. (Lambda 객체는 constraint에 곱하는 값임)
                            if len(losses) > 1:
                                lambda_ = Lambda(count=len(epsilons)) # epsilon의 개수만큼 0.0으로 initialize함 # 왜 이거는 gradient ascent 하는걸까?
                                if use_cuda:
                                    lambda_.cuda()
                            
                            args.optim = "embedgd_le"
                            optimizer = OptimizerLE.from_opt(outputs, args)
                            optimizer.set_init_pred(predicted_batch)
                            
                            cur_lr = args.lr
                            
                            if len(losses) > 1: # 제약조건이 있을 경우에는, lambda를 gradient ascent하는 optimizer도 정의한다.
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
                            
                            # locate하는 코드를 넣을 장소는 여기.
                            
                            # 
                            
                            # 200개의 step으로 현재는 지정되어 있음.
                            for step in range(args.optim_steps):
                                try: # keyboard interrupt가 있을 때 그냥 프로그램을 종료하는게 아니라, 해당 샘플에 대한 optim_step을 종료하는 것으로 될 수 있도록. try~except~문 구성.
                                    with torch.cuda.amp.autocast():
                                        losses_for_backward = []
                                        logging_outputs = []
                                        
                                        # ??? 이부분이 정확히 어떤 것을 하는걸까? 코드를 이해하지는 못했다.
                                        # 다만, 변수명에서 미루어 짐작했을 때에는 예측한 토큰의 임베딩값, 예측한 토큰값, 예측한 확률값을 리턴.
                                        # target_type이 TargetEmbedding 일 때는 아래 수행.
                                        # return (pred_embs, self._pred_embeds), predictions, (pred_probs, softmax_pred_probs)
                                        pred_embeds, pred_tokens, pred_probs = outputs.forward_multiple(embed_luts, new_predictions = getattr(optimizer._optimizer, "new_predictions", None))
                                        
                                        def get_sent(tokens, tokenizer):
                                            batch = []
                                            if args.target_tokenize_different:
                                                with tokenizer.as_target_tokenizer():
                                                    for toks in tokens:
                                                        batch.append(tokenizer.decode(clean_output(toks.tolist(), -1, allow_first_eos=losses[0] == "bart")))
                                            else:
                                                for toks in tokens:
                                                    batch.append(tokenizer.decode(clean_output(toks.tolist(), -1, allow_first_eos=losses[0]=="bart")))
                                            return batch
                                        
                                        target_sents = get_sent(torch.cat([target_prefix, pred_tokens], dim=1), primary_tokenizer)
                                        if step % 10 == 0:
                                            intermediate_result.update({f"step_{step}_text": target_sents})
                                        
                                        original_preds = None
                                        if len(pred_embeds) > 1:
                                            original_preds = pred_embeds[1]
                                        
                                        # loss를 계산한다. lossid = 0 은 autoregressive output 이다. 
                                        # lossid > 0 은 constraint 들이다.
                                        for lossid, lossname in enumerate(losses):
                                            lossvalue, logging_output =\
                                                lossfns[lossid].compute_loss(
                                                    [source_batch, target_prefix], # batch
                                                    [pred_tokens, pred_embeds[0][lossid], pred_probs], # preds
                                                    additional_batch = additional_batch,
                                                    context_batch = context_batch,
                                                    use_context = args.use_context,
                                                    embed_scale = embed_scales[lossid],
                                                    label_id = label_ids[lossid],
                                                    keyword = keywords[lossid], 
                                                    original_preds = original_preds, 
                                                    kweight = new_kweight,
                                                    step = step
                                                )
                                            losslists[lossid][-1].append(lossvalue.sum().item())
                                            losses_for_backward.append(lossvalue)
                                            logging_outputs.append(logging_output)
                                        optimizer.zero_grad(set_to_none = True)
                                        outputs.zero_grad()
                                        
                                        if len(losses) > 1:
                                            optimizer_lambda.zero_grad(set_to_none=True)
                                            lambda_.zero_grad()
                                        
                                        for model in name2model.values():
                                            model.zero_grad(set_to_none=True)
                                        
                                        # linear_scale == True 일 때 (cold 같은 세팅일 때) betas를 사용해서 weighted sum을 한다.
                                        if args.linear_scale == "true":
                                            
                                            total_loss = 0
                                            cur_epsilons = []
                                            for sid in range(len(losses_for_backward)):
                                                total_loss = total_loss + betas[sid] * losses_for_backward[sid]
                                                cur_epsilons.append(0.0)
                                            
                                            total_batchloss = total_loss.sum()
                                            optimizer.backward(total_batchloss, retain_graph=False, scaler=scaler)
                                        # Lagrangian으로 식을 세웠을 때는 beta를 안쓰고 lambda_.get_loss를 이용한다.
                                        else:
                                            
                                            total_loss = 0.0
                                            total_loss = losses_for_backward[0] # perplexity (autoregressive loss)
                                            
                                            cur_epsilons = []
                                            
                                            constraint_values = []
                                            for sid in range(1, len(losses_for_backward)): # secondary losses or constraints
                                                cur_epsilon = get_epsilon(step, epsilons[sid-1], min_epsilons[sid-1], epsilon_warmup_steps[sid-1], epsilon_cooldown_steps[sid-1], epsilon_decay_functions[sid-1])
                                                # constraint value
                                                constraint_value = (cur_epsilon - losses_for_backward[sid]).detach()
                                                damp = args.dampness * constraint_value
                                                # 왜 mask를 일단 만들까? -> if constraint is satified and lambda < damp, then don't use lambdas to update thetas
                                                # 위 설명을 보고 코드에 대한 해석: 
                                                #    damp가 0 보다 크면 losses_for_backward[sid]가 cur_epsilon 보다 작은 것. -> constraint를 만족하는 것.
                                                #    ??? lambda가 damp 보다 작다는 것은,, 잘 모르겠네 ㅋㅋㅋㅋ ㅠㅠ;; 왜 저 조건도 필요한걸까?
                                                #    아무튼 2가지 조건 모두 만족하면 mask값이 0이 됨.
                                                # def get_mask(self, i, damp):
                                                #     # if constraint is satified and lambda < damp, then don't use lambdas to update thetas
                                                #     return 1 - damp.ge(0.).float() * self.lambda_[i].data.le(damp).float()
                                                mask = lambda_.get_mask(sid-1, damp)
                                                
                                                # 근데 damp 인자로 damp * mask를 넣어주는데, 그러면 self.lambda[i] - 0 아닌가? 이게 어째서 don't use lambdas to update thetas 인건가..
                                                # def get_loss(self, i, damp, loss):
                                                #     return (self.lambda_[i] - damp) * loss
                                                closs_for_theta = lambda_.get_loss(sid-1, damp * mask, (cur_epsilon - losses_for_backward[sid]))
                                                total_loss = total_loss - closs_for_theta
                                                
                                                cur_epsilons.append(cur_epsilon)
                                                constraint_values.append(constraint_value.item())
                                            total_batchloss = total_loss.sum()
                                            optimizer.backward(total_batchloss, retain_graph=False, scaler=scaler)                            

                                    indices=[5,6] # indices to edit
                                    #indices = locate_indices_all[sample_idx]
                                    if logging_outputs[0].get('entropy', None) is not None:
                                        optimizer.step(indices, scaler=scaler, entropy=logging_outputs[0].get('entropy', None))
                                    else:
                                        optimizer.step(indices, scaler=scaler)
                                    
                                    # start : lambda를 업데이트하기 위한 코드 + EmbedGD의 learning rate를 업데이트하기 위한 코드
                                    update_lr_condition = "none"
                                    
                                    if args.linear_scale != "true" and len(losses) > 1:
                                        sats = torch.Tensor(constraint_values).ge(0.).to(device) # constraint_values가 0 보다 크면, epsilon - loss > 0 that is, epsilon > loss 이므로 constraint 만족
                                        update_lambda_condition = (step % args.lambda_update == 0)
                                        lambda_mask = float(update_lambda_condition) * torch.ones_like(sats) # update_lambda_condition 이 True 이면, lambda_mask는 [1,1,1..,1] 이고 아니면 [0,0,0...,0]
                                                                                
                                        lambda_mask += (1 - sats.float()) * (lambda_.is_zero()) # constraint를 만족한 constraint에는 0이 더해지고, 만족하지 않은 constraint에는 lambda가 0이 아니라면 update_lambda_condition이 False여도 업데이트 한다.s
                                        
                                    total_batchlossitem = losses_for_backward[0].item()
                                    if dynamic_lambda_update_prev_loss is not None and abs(total_batchlossitem - dynamic_lambda_update_prev_loss) <= 1e-6:
                                        repeat_counts[0] += 1
                                        if args.linear_scale != "true" and len(losses) > 1 and args.dynamic_lambda_update:
                                            lambda_mask = (1 - sats.float()) # constraint를 만족하지 않은 constraint만 1임.
                                            
                                        if args.dynamic_lr_update and best_allsat[0] is not None and best_allsat[0]:
                                            update_lr_condition = "increase"
                                    else:
                                        repeat_counts[0] = 1
                                    
                                    dynamic_lambda_update_prev_loss = total_batchlossitem
                                    
                                    if update_lr_condition == "increase":
                                        cur_lr = optimizer._optimizer.update_lr(min(cur_lr + args.lr_update_size, args.max_lr))
                                        
                                    if args.linear_scale != "true" and len(losses) > 1:
                                        optimizer_lambda._optimizer.set_mask(lambda_mask.clamp(max = 1.0, min = 0.0)) # epsilon 값을 만족하지 않은 constraint에 대해서만, lambda를 update.
                                        optimizer_lambda.step() # lambda를 update.
                                        lambda_.make_positive()
                                    # end 
                                    
                                    gc.collect()
                                    
                                    # best...를 업데이트 하는 부분
                                    cur_losses = []
                                    for b in range(batch_size):
                                        cur_loss = 0.0
                                        for beta, lossval in zip(betas, losses_for_backward):
                                            cur_loss = cur_loss + beta * lossval[b].item()
                                        cur_losses.append(cur_loss)
                                        
                                        constrained = []
                                        allsat = True
                                        for i in range(1, len(losses)):
                                            if losses_for_backward[i] <= min_epsilons[i-1]:
                                                constrained.append("sat")
                                            else:
                                                constrained.append("vio")
                                                allsat = False
                                                
                                        if args.show_all_outputs and len(losses) > 1 and allsat:
                                            best_prediction_set[b].add(target_sents[b])
                                        
                                        constrained = ",".join(constrained)
                                        
                                        modify_condition = \
                                            args.selection_criterion == "last" or\
                                                (best_loss[b] is None and args.selection_criterion == "weighted_sum") or\
                                                (best_loss[b] is not None and args.selection_criterion == "weighted_sum" and best_loss[b] > cur_loss)
                                        
                                        if not modify_condition and args.selection_criterion == "mrr_allsat":
                                            modify_condition = \
                                                (best_loss[b] is None and allsat and repeat_counts[b] == 2) or \
                                                (best_loss[b] is not None and best_allsat[b] and allsat and repeat_counts[b] == 2)
                                        elif not modify_condition and args.selection_criterion == "primary_allsat":
                                            modify_condition = \
                                                (best_loss[b] is None and allsat) or \
                                                (best_loss[b] is not None and not best_allsat[b] and allsat) or \
                                                (best_allsat[b] and allsat and best_loss[b] > cur_loss)
                                                
                                        if modify_condition:
                                            if args.dynamic_lr_update:
                                                print("resetting the learning rate and noise std, a constraint has been satisfied")
                                                cur_lr = optimizer._optimizer.update_lr(args.lr)
                                                optimizer._optimizer.set_begin_std(0.01)
                                            if args.selection_criterion != "last":
                                                print(f"modify condition @{step}", time.time() - starttime, end  ="\n")
                                            best_loss[b] = cur_loss
                                            best_allsat[b] = allsat
                                            best_repeat_count[b] = repeat_counts[b]
                                            for i in range(len(losses)):
                                                best_losses[i][b] = losses_for_backward[i][b].item()
                                                
                                            best_pred_tokens[b] = pred_tokens[b]
                                            best_index[b] = step
                                            
                                            best_constrained = constrained
                                            best_step = step 
                                            
                                    if not args.time and step > 0 and step % args.log_interval == 0:
                                        if len(losses) > 1:
                                            log = f"beam cons: {predicted_allsat};"
                                            log = f"Step {step}: lr: {cur_lr}; total_loss: {total_batchloss:.4f}; current [loss:{sum(cur_losses):.4f}; l:{','.join([f'{x:.4f}' for x in lambda_().tolist()])}; e:{','.join(f'{x:.4f}' for x in cur_epsilons)}]; cons: {constrained}; "
                                            for i in range(len(losslists)):
                                                log = log + f" {lossabbr[i]}:{losslists[i][-1][-1]:.4f}; "
                                            
                                            if best_loss[0] is not None:
                                                log = log[:-1] + f"] |||| best [cur_loss:{sum(best_loss):.4f}; cons: {best_constrained}; "
                                                for i in range(len(best_losses)):
                                                    log = log + f"{lossabbr[i]}:{sum(best_losses[i]):.4f}; "
                                                log = log[:-1] + f"@ step #{best_index[-1]}"
                                                log = log + "]"
                                            else: 
                                                log = log[:-1] + f"] |||| best [none of the generations so far satisfies constraints]"
                                        else:
                                            log = f"Step {step}: lr:{cur_lr}; loss:{total_batchloss:.4f}; current [loss:{sum(cur_losses):.4f};"
                                            for i in range(len(losslists)):
                                                log = log + f" {lossabbr[i]}:{losslists[i][-1][-1]:.4f}; "
                                                
                                            if best_loss[0] is not None:
                                                log = log[:-1] + f"] best [loss:{sum(best_loss):.4f} "
                                                for i in range(len(best_losses)):
                                                    log = log + f"{lossabbr[i]}:{sum(best_losses[i]):.4f}; "
                                                log = log[:-1] + f" at step {best_index[-1]}"
                                                log = log + "]"
                                            else:
                                                log = log[:-1] + f"] |||| best [none of the generations so far satisfies constraints]"

                                            print(log, end="\n")
                                    
                                    del losses_for_backward
                                    
                                    # early stopping 관련된 로직. 조건에 해당하면 optim_steps만큼 for loop을 돌지 않아도 break.
                                    if args.early_stop_steps > 0:
                                        # selection_criterion이 allsat이 포함되어 있고, best_allsat[0]이 true 인 경우
                                        # selection_criterion이 weighted_sum이나 last 인 경우
                                        # early_stop_condition = True 이다.
                                        early_stop_condition = \
                                            ("allsat" in args.selection_criterion and best_allsat[0]) or\
                                            (args.selection_criterion == "weighted sum") or \
                                            (args.selection_criterion == "last")
                                        
                                        # loss가 큰 변화가 없으면 patience count +1
                                        if prev_loss is not None and abs(cur_loss - prev_loss) <= 1e-6:
                                            same_loss_count += 1
                                            
                                        else: 
                                            same_loss_count = 0
                                        
                                        # condition을 만족하면서, loss 변화가 args.early_stop_steps 이상 없었던 경우
                                        if early_stop_condition and same_loss_count >= args.early_stop_steps:
                                            print(f"Early stop at @{step} with a loss value of {cur_loss} and satisfied constraints")
                                            break
                                        # condition을 만족하지는 않지만, loss 변화가 args.early_stop_steps + 100 이상인 경우
                                        elif same_loss_count >= args.early_stop_steps + 100:
                                            print(f"Early stop at @{step} with a loss value of {cur_loss} and unsatisfied constraints")
                                            break
                                        
                                        prev_loss = cur_loss
                                        
                                except KeyboardInterrupt:
                                    print("skipping remaining optimizing steps and showing the best option so far")
                                    broken = True
                                    break
                            # end for loop for gradient-based inference
                                
                            predictions = []
                            prediction_idss = []
                            broken_skip = False
                            skip_printing = False
                            for b, item in enumerate(best_pred_tokens):
                                if item is None and broken: # best_pred_tokens가 저장되기도 전에 keyboard interrupt가 있었던 경우
                                    skip_printing = True
                                    if broken: 
                                        broken_skip = input("Skip this input entirely? yes(y)/no(continue)/press ctrl+c to exit")
                                        broken_skip = broken_skip.lower() == "y"
                                        break
                                # 꼭 mucola output이어야 한다는 조건이 없고, best modification도 제약조건을 모두 만족하지 못했을 때
                                # 또는 best modification이라고 저장된 것이 없을 때
                                # -> AR prediction 사용
                                if (args.only_mucoco == "false" and not best_allsat[b]) or (item is None):
                                    prediction_ids = ", ".join([str(idx) for idx in AR_predicted_indices[0].tolist()])
                                    prediction_indices = AR_predicted_indices[0].tolist()
                                    prediction = AR_prediction
                                    
                                    lossvalue = 0.0
                                    for lossid in range(len(betas)):
                                        lossvalue += betas[lossid] * predictedlosslists[-1][lossid][b]
                                    print(f"best prediction is from beam search, all constraints were not satisfied, allsat={lengthwise_best_prediction[b][2]}")
                                else:
                                    # 아닌 경우는 best modificaiton을 lengthwise_best_prediction 에 저장..
                                    prediction_ids = ", ".join([str(x) for x in target_prefix[b].tolist()])
                                    prediction_ids += f'[{", ".join([str(x) for x in item.tolist()])}]'
                                    prediction_indices = target_prefix[b].tolist() + item.tolist()
                                    
                                    targets = clean_output(item.tolist(), primary_tokenizer.eos_token_id, allow_first_eos=losses[0]=="bart")
                                    if args.target_tokenize_different:
                                        with primary_tokenizer.as_target_tokenizer():
                                            prediction = primary_tokenizer.decode(target_prefix[b].tolist() + targets)
                                    else:
                                        prediction = primary_tokenizer.decode(target_prefix[b].tolist() + targets)
                                    
                                    print("best prediction at step", best_index[b])
                                    lossvalue = best_loss[b]
                                    
                                    modify_condition =\
                                        lengthwise_best_prediction[b] is None or \
                                        (args.selection_criterion == "weighted_sum" and lengthwise_best_prediction[b][1] > lossvalue)
                                    
                                    if not modify_condition and args.selection_criterion == "primary_allsat":
                                        modify_condition = \
                                            (not lengthwise_best_prediction[b][2] and best_allsat[b]) or \
                                            (lengthwise_best_prediction[b][2] and best_allsat[b] and lengthwise_best_prediction[b][1] > lossvalue)
                                    elif not modify_condition and args.selection_criterion == "mrr_allsat":
                                        modify_condition =\
                                            (not lengthwise_best_prediction[b][2] and best_allsat[b] and best_repeat_count[b] >= 2) or \
                                            (lengthwise_best_prediction[b][2] and lengthwise_best_prediction[b][4] >= 2 and lengthwise_best_prediction[b][1] > lossvalue)
                                    
                                    if modify_condition:
                                        if args.debug:
                                            print("modify condition satisfied", end="\n")
                                        else:
                                            outallsatf.write("modify_contition satisfied ")
                                        
                                        lengthwise_best_prediction[b] = (prediction, lossvalue, best_allsat[b], prediction_indices, best_repeat_count[b])
                                        intermediate_result.update({"best_step": best_index[b],
                                                                    "best_prediction": prediction})

                                prediction_idss.append(prediction_ids)
                                predictions.append(prediction)
                                
                            all_stepcounts += best_index
                            
                            optimizer.zero_grad(set_to_none = True)
                            del outputs
                            del optimizer
                            if len(losses) > 1:
                                optimizer_lambda.zero_grad()
                                del optimizer_lambda
                                del lambda_
                            for modelname in loss2modelname.values():
                                name2model[modelname].zero_grad(set_to_none=True)                                        
                            torch.cuda.empty_cache()
                        
                        # 해당 sample에 대해서 lengthwise_best_prediction을 파일에 저장
                        b = 0
                        if lengthwise_best_prediction[b] is None or not lengthwise_best_prediction[b][2]:
                            if restart_idx < args.restarts:
                                continue # skip printing and loop over
                            elif lengthwise_best_prediction[b] is None:
                                lengthwise_best_prediction = [("", -1, False, [], -1)]
                        
                        if args.debug:
                            if not skip_printing:
                                for b in range(batch_size):
                                    print("sample #" + str(sample_idx), f"repeat count: {lengthwise_best_prediction[b][4]}", "best prediction for all lengths: ", lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                        else:
                            if args.output_style == "text":
                                for b in range(batch_size):
                                    # recall: lengthwise_best_prediction = [(AR_prediction, total_predicted_loss, predicted_allsat, predicted_batch[0].tolist(), -1)]
                                    outf.write(lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n") # AR_prediction
                                    outf.flush()
                                    outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n") # predicted_allsat
                                    outallsatf.flush()
                            else:
                                if sample_idx == 0: # 한 prompt 에 대해서 첫번째 sample 일 경우
                                    output = {
                                        "prompt": {
                                            "text": source_text,
                                            "tokens": source_indices_write},
                                        "generations": [{
                                            "text": lengthwise_best_prediction[b][0],
                                            "tokens": lengthwise_best_prediction[b][3],
                                            "allsat": lengthwise_best_prediction[b][2],
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
                                            "repeat_count": lengthwise_best_prediction[b][4],
                                            "mucoco": True
                                        }
                                    )
                                    
                                # 한 prompt 에 대해서 마지막 sample 일 경우, output에 추가한 후 dump 까지 해준다.
                                if sample_idx + 1 == args.num_samples: 
                                    json.dump(output, outf)
                                    outf.write("\n")
                                    outf.flush()
                                    
                                    outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n")
                                    outallsatf.flush()
                                
                                output2 = intermediate_result
                                json.dump(output2, outf2)
                                outf2.write("\n")
                                outf2.flush()
                        print(f"required output achieved or number of restarts ran out at attempt #{restart_idx + 1}")
                        break
                            
                    else: # skip 일 경우 == skipping mucola and writing beam search output 
                        if ask_skip != "y": # 사용자에게 물어봐서 skip 하기로 한게 아니면
                            ## args.debug 일 때는 과감히 생략
                            print("Skipping this example. the beam search output already satisfies all the constraints or there's no constraints")
                            # output style에 따라 다른 포맷으로 저장
                            if args.output_style == "text":
                                for b in range(batch_size):
                                    # recall: lengthwise_best_prediction = [(AR_prediction, total_predicted_loss, predicted_allsat, predicted_batch[0].tolist(), -1)]
                                    outf.write(lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n") # AR_prediction
                                    outf.flush()
                                    outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n") # predicted_allsat
                                    outallsatf.flush()
                            else:
                                for b in range(batch_size):
                                    if sample_idx == 0: # 한 prompt 에 대해서 첫번째 sample 일 경우
                                        output = {
                                            "prompt": {
                                                "text": source_text,
                                                "tokens": source_indices_write},
                                            "generations": [{
                                                "text": lengthwise_best_prediction[b][0],
                                                "tokens": lengthwise_best_prediction[b][3],
                                                "allsat": lengthwise_best_prediction[b][2],
                                                "mucoco": False # !!! AR output 임이 표시되어 있구나
                                            }]
                                        }
                                    else:
                                        output['generations'].append(
                                            {
                                                "text": lengthwise_best_prediction[b][0],
                                                "tokens": lengthwise_best_prediction[b][3],
                                                "allsat": lengthwise_best_prediction[b][2],
                                                "mucoco": False # !!! AR output 임이 표시되어 있구나
                                            }
                                        )
                                        
                                # 한 prompt 에 대해서 마지막 sample 일 경우, output에 추가한 후 dump 까지 해준다.
                                if sample_idx + 1 == args.num_samples: 
                                    json.dump(output, outf)
                                    outf.write("\n")
                                    outf.flush()
                    break # restart 하는 loop을 탈출 (don't restart)
        
        # 1개 prompt 에 대한 작업이 다 끝나면, source_batch 등등을 새로 initialize 합니다.
        del source_batch
        del target_batch
        del additional_batch
        del for_predicted_source_batch
        del predicted_batch
        del context_batch
        source_batch, target_batch, additional_batch, for_predicted_source_batch, predicted_batch, context_batch = [], [], [], [], [], []