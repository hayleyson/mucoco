from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from new_module.utils.utils import read_outputs, ravel

def nli_score(
    generations_df,
    write_file: Optional[Union[str, Path]] = None,
    indexes_to_skip: Optional[Iterable[int]] = None,
    device: str = "cuda",
):
    model_paths = [
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "cross-encoder/nli-roberta-base",
    "cross-encoder/nli-deberta-v3-base"
]
    models = []
    tokenizers = []
    for model_path in model_paths:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, use_safetensors=True).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        models.append(model)
        tokenizers.append(tokenizer)

    # 클래스별 확률 합산 변수 초기화
    total_entail_prob = 0
    total_neutral_prob = 0
    total_contradiction_prob = 0
    total_count = 0
    total_contradiction_count = 0
    total_entail_count = 0
    total_neutral_count = 0

    skip_set = set(indexes_to_skip) if indexes_to_skip is not None else set()
    results = []
    index = -1
    # 각 row에 대해 NLI 점수 계산
    for _, row in tqdm(generations_df.iterrows(), total=len(generations_df), desc='NLI classifying...', mininterval=5):
        premise_raw = row["prompt"]["text"]
        hypotheses = [gen["text"] for gen in row["generations"]]
        premise_lc = premise_raw.lower()

        # 각 hypothesis에 대해 NLI 평가
        for hypothesis in hypotheses:
            index += 1
            entail_prob_sum = 0
            neutral_prob_sum = 0
            contradiction_prob_sum = 0
            if (hypothesis == "") or (index in skip_set):
                results.append(
                    {
                        "entailment_prob": np.nan,
                        "neutral_prob": np.nan,
                        "contradiction_prob": np.nan,
                        "nli_class": np.nan,
                    }
                )
                continue
            hypothesis_lc = hypothesis.lower()
            if premise_lc in hypothesis_lc:
                hypothesis_lc = hypothesis_lc.replace(premise_lc, "")
            # 각 모델에 대해 예측 수행
            for i, (model, tokenizer) in enumerate(zip(models, tokenizers)):
                inputs = tokenizer(
                    premise_lc,
                    hypothesis_lc,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).squeeze()  # 예측 확률 계산

                if "ynie" in model_paths[i]:
                    entail_prob_sum += probs[0].item()  # entailment 확률
                    neutral_prob_sum += probs[1].item()    # neutral 확률
                    contradiction_prob_sum += probs[2].item() # contradiction 확률
                else:
                    contradiction_prob_sum += probs[0].item()  # contradiction 확률
                    entail_prob_sum += probs[1].item()     # entailment 확률
                    neutral_prob_sum += probs[2].item() # neutral 확률


            # 각 hypothesis에 대한 모델 평균 확률 계산 및 누적
            entail_prob_avg = entail_prob_sum / len(models)
            neutral_prob_avg = neutral_prob_sum / len(models)
            contradiction_prob_avg = contradiction_prob_sum / len(models)

            total_entail_prob += entail_prob_avg
            total_neutral_prob += neutral_prob_avg
            total_contradiction_prob += contradiction_prob_avg
            if contradiction_prob_avg == max(entail_prob_avg, neutral_prob_avg, contradiction_prob_avg):
                classified_class = "contradiction"
                total_contradiction_count += 1
            elif entail_prob_avg == max(entail_prob_avg, neutral_prob_avg, contradiction_prob_avg):
                classified_class = "entail" 
                total_entail_count += 1
            else:
                classified_class = 'neutral'
                total_neutral_count += 1
            total_count += 1

            results.append({
                "entailment_prob": entail_prob_avg,
                "neutral_prob": neutral_prob_avg,
                "contradiction_prob": contradiction_prob_avg,
                "nli_class": classified_class
            })

    # 전체 데이터에 대한 평균 확률 계산
    if total_count == 0:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan

    avg_nli_entail = total_entail_prob / total_count
    avg_nli_neutral = total_neutral_prob / total_count
    avg_nli_contradiction = total_contradiction_prob / total_count
    entail_ratio = total_entail_count / total_count
    neutral_ratio = total_neutral_count / total_count
    contadiction_ratio = total_contradiction_count / total_count

    if write_file:
        with open(write_file, 'w') as f:
            for result in results:
                f.write(f"{result}\n")

    return avg_nli_entail, avg_nli_neutral, avg_nli_contradiction, contadiction_ratio, entail_ratio, neutral_ratio


if __name__ == "__main__":
    NLI_IFEVAL_OUTPUT_PATH = Path(
        "new_module/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nli_ifeval/gpt-3.5-turbo-0125_nli_ifeval_150_postprocessed.jsonl"
    )
    NLI_OUTPUT_PATH = Path(
        "new_module/base_lm_generate/baselm_gens/gpt-3.5-turbo-0125/nli/gpt-3.5-turbo-0125_anli-r2-test_prompt_4_150.jsonl"
    )

    outputs = pd.read_json(NLI_IFEVAL_OUTPUT_PATH, lines=True)

    avg_nli_entail, avg_nli_neutral, avg_nli_contradiction, contadiction_ratio, entail_ratio, neutral_ratio = nli_score(
        outputs, write_file=f"{NLI_IFEVAL_OUTPUT_PATH}-result.txt.nli"
    )

    with open(f"{NLI_IFEVAL_OUTPUT_PATH}-result.txt", "w") as f:
        f.write(
            f"avg_nli_entail: {avg_nli_entail}, avg_nli_neutral: {avg_nli_neutral}, "
            f"avg_nli_contradiction: {avg_nli_contradiction}, contadiction_ratio: {contadiction_ratio}, "
            f"entail_ratio: {entail_ratio}, neutral_ratio: {neutral_ratio}\n"
        )

    nli_outputs = pd.read_json(NLI_OUTPUT_PATH, lines=True)

    _outputs = read_outputs(NLI_IFEVAL_OUTPUT_PATH)
    skipped_indexes = _outputs.loc[_outputs["text"] == ""].index.tolist()

    avg_nli_entail, avg_nli_neutral, avg_nli_contradiction, contadiction_ratio, entail_ratio, neutral_ratio = nli_score(
        nli_outputs,
        write_file=f"{NLI_OUTPUT_PATH}-result.txt.nli",
        indexes_to_skip=skipped_indexes,
    )

    with open(f"{NLI_OUTPUT_PATH}-result.txt", "w") as f:
        f.write(
            f"avg_nli_entail: {avg_nli_entail}, avg_nli_neutral: {avg_nli_neutral}, "
            f"avg_nli_contradiction: {avg_nli_contradiction}, contadiction_ratio: {contadiction_ratio}, "
            f"entail_ratio: {entail_ratio}, neutral_ratio: {neutral_ratio}\n"
        )