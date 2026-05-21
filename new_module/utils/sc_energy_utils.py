from typing import List, Tuple
import yaml, torch, pickle, sys, re, random
from pathlib import Path
from torch.utils.data import Dataset

sys.path.append("new_module/set_consistency_energy")
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
    base = Path(f"new_module/data/{dataset_name}")
    fname = f"{dataset_name}_{split}_{name}_dataset.pickle"
    return base / fname

def load_pickle_dataset(dataset_name: str, split: str, name: str):
    path = _pkl_path(dataset_name, split, name)
    with open(path, "rb") as f:
        ds = pickle.load(f)
    return ds

def load_eval2_dataset(dataset_name: str, split: str="eval2", use_only_incon: bool=True, n_samples: int=None, random_seed: int=42) ->Tuple[List[Dataset], List[str]]:
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
    
    if n_samples is not None:
        eval2_samples = []
        for dataset in eval2_datasets:
            eval2_samples.extend(dataset.dataset)
        print(f"Num samples: {len(eval2_samples)}")
        
        canonical_eval2_dataset = eval2_datasets[0]
        random.seed(random_seed)
        eval2_samples = random.sample(eval2_samples, n_samples)
        canonical_eval2_dataset.dataset = eval2_samples
        
        if use_only_incon:
            return [canonical_eval2_dataset], ['incon']
        else:
            return [canonical_eval2_dataset], ['total']
    
    return eval2_datasets, eval2_steps_names




def parse_set_text(text: str, source_mode="ebm") -> List[List[str]]:
    if source_mode == "ebm":
        set_elements = text.split(".")[:-1]
        set_elements = [element.split("The answer is") for element in set_elements]
        set_elements = [x.strip() for element in set_elements for x in element]
        set_elements = [element + [None] for element in set_elements]
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
    return set_elements

def format_set_text(set_texts: List[List[str]], target_mode: str="ebm") -> str:
    if target_mode == "ebm":
        return ".".join([f"{text[0].strip()} The answer is {text[1].strip()}." for text in set_texts])
    elif target_mode == "llm":
        return "".join([f"({i+1}) question: {text[0]}, answer: {text[1]}." for i, text in enumerate(set_texts)])
    else:
        raise ValueError(f"Invalid target mode: {target_mode}")

def convert_format(set_string: str, source_mode: str="llm", target_mode: str="ebm") -> str:
    set_texts = parse_set_text(set_string, source_mode)
    return format_set_text(set_texts, target_mode)