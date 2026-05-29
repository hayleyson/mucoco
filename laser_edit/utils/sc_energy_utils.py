from typing import List, Tuple
import yaml, torch, pickle, sys, re, random
from pathlib import Path
from torch.utils.data import Dataset

sys.path.append("laser_edit/set_consistency_energy")
from energynets.energynet import energynet
from tasks.dataset_loader import concat_arbitrary_pairs


def load_sc_energy_model(config_path, device):
    
    model_config = yaml.load(open(config_path), 
                                Loader=yaml.FullLoader)
    model_config['device'] = device

    energy_net = energynet(params=model_config)
    energy_net.load_state_dict(torch.load(model_config["model_path"], 
                                        map_location=model_config['device'],
                                        weights_only=True)['state_dict'], strict=False)
    if 'threshold' in torch.load(model_config["model_path"],
                                map_location=model_config['device'],
                                weights_only=True):
        energy_net.threshold = torch.load(model_config["model_path"],
                                map_location=model_config['device'],
                                weights_only=True)['threshold']
    energy_net.eval()
    energy_net.to(device)
    
    return energy_net
    
def _pkl_path(dataset_name: str, split: str, name: str) -> Path:
    """
    set_consistency_dataset/{dataset_name}/ 경로의 피클 파일 경로를 반환.
    파일명 규칙: {dataset_name}_{split}_{NAME}_dataset.pickle
      예) lconvqa_test_C_dataset.pickle, lconvqa_test_CI_dataset.pickle
    name 인자는 "test_C", "test_CI" 등 split 접두사를 포함한 문자열을 기대.
    """
    base = Path(f"laser_edit/data/{dataset_name}")
    fname = f"{dataset_name}_{split}_{name}_dataset.pickle"
    return base / fname

def load_pickle_dataset(dataset_name: str, split: str, name: str):
    path = _pkl_path(dataset_name, split, name)
    with open(path, "rb") as f:
        ds = pickle.load(f)
    return ds

def load_sc_energy_dataset(dataset_name: str, split: str="eval2", use_only_incon: bool=True, n_samples: int=None, random_seed: int=42) ->Tuple[List[Dataset], List[str]]:
    eval2_con_dataset_arbitrary_pairs = load_pickle_dataset(dataset_name, split, "C")
    eval2_incon_dataset_arbitrary_pairs = load_pickle_dataset(dataset_name, split, "I")
    eval2_con_dataset_arbitrary_pairs.dataset = [t for t in eval2_con_dataset_arbitrary_pairs.dataset if len(t) >=4]
    eval2_incon_dataset_arbitrary_pairs.dataset = [t for t in eval2_incon_dataset_arbitrary_pairs.dataset if len(t) >=4]
    
    concat2_dataset, concat2_names, concat2_set_sizes = concat_arbitrary_pairs([eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs], concat_num=2)
    concat3_dataset, concat3_names, concat3_set_sizes = concat_arbitrary_pairs([eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs], concat_num=3)
    concat4_dataset, concat4_names, concat4_set_sizes = concat_arbitrary_pairs([eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs], concat_num=4)

    eval2_steps_names = ['con', 'incon'] + concat2_names+ concat3_names+ concat4_names
    eval2_datasets = [eval2_con_dataset_arbitrary_pairs, eval2_incon_dataset_arbitrary_pairs
            ] + concat2_dataset + concat3_dataset + concat4_dataset

    # Classification accuracy와 별개로 locate accuracy만 보고 싶기 때문에, incon인 샘플만 취한다.
    if use_only_incon:
        eval2_datasets = [eval2_datasets[i] for i in range(len(eval2_datasets)) if ('incon' in eval2_steps_names[i])]
        eval2_steps_names = [eval2_steps_name for eval2_steps_name in eval2_steps_names if ('incon' in eval2_steps_name)]
        print(f"Using samples from inconsistent datasets: {eval2_steps_names}")
    
    
    eval2_samples = []
    for dataset in eval2_datasets:
        eval2_samples.extend(dataset.dataset)
    print(f"Num samples: {len(eval2_samples)}")
    
    
    canonical_eval2_dataset = eval2_datasets[0]
    canonical_eval2_dataset.dataset = eval2_samples
    
    if n_samples is not None:
        
        random.seed(random_seed)
        eval2_samples = random.sample(eval2_samples, n_samples)
        print(f"Num samples after sampling: {len(eval2_samples)}")
        canonical_eval2_dataset.dataset = eval2_samples
        
    if use_only_incon:
        return [canonical_eval2_dataset], ['incon']
    else:
        return [canonical_eval2_dataset], ['total']




def _strip_leading_cls(text: str, cls_token: str = "<s>") -> str:
    text = text.strip()
    if text.startswith(cls_token):
        text = text[len(cls_token):].strip()
    return text


def _ensure_sentence_ending(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?":
        return text + "."
    return text


def _split_set_nli_ebm_text(text: str) -> List[str]:
    text = _strip_leading_cls(text)
    return [
        _ensure_sentence_ending(match.group(0))
        for match in re.finditer(r"[^.!?]+[.!?]*", text)
        if match.group(0).strip()
    ]


def _set_nli_sentence_text(element) -> str:
    if isinstance(element, str):
        return element
    return element[0]


def parse_set_text(text: str, source_mode="ebm", dataset="lconvqa") -> List[List[str]]:
    
    if dataset == "lconvqa":
        if source_mode == "ebm":
            set_elements = []
            for element in text.split(".")[:-1]:
                if "The answer is" not in element:
                    continue
                question, answer = element.split("The answer is", 1)
                set_elements.append([question.strip(), answer.strip(), None])
        elif source_mode == "llm":
            prefix = "question:"
            set_elements = []
            for chunk in re.split(r"\(\d+\)", text):
                chunk = chunk.strip()
                if not chunk or ", answer:" not in chunk:
                    continue
                question_part, answer_part = chunk.split(", answer:", 1)
                question_part = question_part.strip()
                if question_part[: len(prefix)].lower() == prefix:
                    question_part = question_part[len(prefix) :].lstrip()
                set_elements.append([question_part.strip(), answer_part.strip().rstrip('.'), None])
        else:
            raise ValueError(f"Invalid source mode: {source_mode}")
    elif dataset == "set_nli":
        if source_mode == "ebm":
            set_elements = _split_set_nli_ebm_text(text)
            set_elements = [[element] + [None] for element in set_elements]
        elif source_mode == "llm":
            set_elements = []
            chunks = [chunk.strip() for chunk in re.split(r"\(\d+\)", text)]
            if len([chunk for chunk in chunks if chunk]) <= 1:
                chunks = _split_set_nli_ebm_text(text)
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                set_elements.append([_ensure_sentence_ending(chunk), None])
        else:
            raise ValueError(f"Invalid source mode: {source_mode}")
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    return set_elements

def format_set_text(set_texts: List[List[str]], target_mode: str="ebm", dataset: str="lconvqa") -> str:
    if dataset == "lconvqa":
        if target_mode == "ebm":
            return " ".join([f"{text[0].strip()} The answer is {text[1].strip()}." for text in set_texts])
        elif target_mode == "llm":
            return " ".join([f"({i+1}) question: {text[0]}, answer: {text[1]}." for i, text in enumerate(set_texts)])
        else:
            raise ValueError(f"Invalid target mode: {target_mode}")
    elif dataset == "set_nli":
        sentences = [_ensure_sentence_ending(_set_nli_sentence_text(text)) for text in set_texts if text]
        if target_mode == "ebm":
            return " ".join(sentences)
        elif target_mode == "llm":
            return " ".join([f"({i+1}) {sentence}" for i, sentence in enumerate(sentences)])
        else:
            raise ValueError(f"Invalid target mode: {target_mode}")
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

def convert_format(set_string: str, source_mode: str="llm", target_mode: str="ebm", dataset: str="lconvqa") -> str:
    set_texts = parse_set_text(set_string, source_mode, dataset)
    return format_set_text(set_texts, target_mode, dataset)