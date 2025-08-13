"""
This code is adapted from Mucola's losses module. (https://github.com/Sachin19/mucoco/blob/sampling2/mucoco/losses)
"""
from typing import List

import evaluate
import numpy as np
import torch
import torch.nn.functional as F

from new_module.losses import BaseLoss, register_loss

torch.set_printoptions(precision=3, sci_mode=False)

@register_loss("bertscore")
class BertScoreLoss(BaseLoss):

    def __init__(self, args):
        super().__init__() 
        self.args = args
            
    def compute_gold_loss(self, prompt:str, predictions:List[str], references:List[str], **kwargs):
        '''
        given predictions and references (original generations) as list of string, return average bertscore F1 ([0,1] range)
        '''
       
        # https://huggingface.co/spaces/evaluate-metric/bertscore
        # The function returns a dictionary with the following keys - precision, recall, f1, hashcode - and corresponding values for each sentence
        bertscore = evaluate.load("bertscore")
        sbert_score_raw = np.array(
            bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
                rescale_with_baseline=True,
            )["f1"]
        )
        # Take the mean of f1 scores for all the predictions
        sbert_score = np.mean(sbert_score_raw)

        return sbert_score


@register_loss("bleu")
class BleuLoss(BaseLoss):

    def __init__(self, args):
        super().__init__() 
        self.args = args
            
    def compute_gold_loss(self, prompt:str, predictions:List[str], references:List[str], **kwargs):
        '''
        given predictions and references (original generations) as list of string, return bleu score ([0,100] range)
        '''
        ## start evaluation
        # https://huggingface.co/spaces/evaluate-metric/sacrebleu
        sacrebleu = evaluate.load("sacrebleu")
        sbleu_score = sacrebleu.compute(
            predictions=predictions, references=[[text] for text in references]
        )["score"]

        return sbleu_score


    
@register_loss("edit_distance")
class EditDistanceLoss(BaseLoss):
    """
    Implements Levenshtein edit distance loss (average over pairs).
    This function is implemented by ChatGPT (GPT-5)
    """

    def __init__(self, tokenizer, args):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

    def _levenshtein(self, a_seq, b_seq):
        """
        Compute Levenshtein distance between two sequences (list of tokens or chars).
        Uses the classic DP with two rolling rows. O(len(a)*len(b)) time, O(min) space.
        """
        len_a, len_b = len(a_seq), len(b_seq)
        # Ensure the shorter sequence is on the horizontal axis to reduce memory
        if len_a < len_b:
            a_seq, b_seq = b_seq, a_seq
            len_a, len_b = len_b, len_a

        # previous[j] = distance between a_seq[:i-1] and b_seq[:j]
        previous = list(range(len_b + 1))
        for i in range(1, len_a + 1):
            current = [i]
            ai = a_seq[i - 1]
            for j in range(1, len_b + 1):
                cost_sub = 0 if ai == b_seq[j - 1] else 1
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + cost_sub
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        return previous[-1]

    def compute_gold_loss(self, prompt: str, predictions: List[str], references: List[str], **kwargs):
        '''
        Given predictions and references (original generations) as lists of strings,
        return the average Levenshtein edit distance as a torch scalar.

        Optional args (via self.args):
          - edit_level: "char" (default) or "word"
          - case_sensitive: bool (default False)
          - normalize: bool (default False)  # distance / max(len(pred), len(ref))
        '''
        if len(predictions) != len(references):
            raise ValueError(f"predictions and references must have the same length, "
                             f"got {len(predictions)} vs {len(references)}")

        n = len(predictions)
        if n == 0:
            return torch.tensor(0.0, dtype=torch.float32)

        level = getattr(self.args, "edit_level", "token")
        case_sensitive = getattr(self.args, "case_sensitive", False)
        normalize = getattr(self.args, "normalize", False)

        total = 0.0
        for pred, ref in zip(predictions, references):
            # Basic sanitation
            pred = "" if pred is None else str(pred)
            ref = "" if ref is None else str(ref)

            if not case_sensitive:
                pred = pred.lower()
                ref = ref.lower()

            if level == "word":
                a_seq = pred.split()
                b_seq = ref.split()
            elif level == "char":  
                a_seq = list(pred)
                b_seq = list(ref)
            else: # "token"
                a_seq = self.tokenizer(pred)['input_ids']
                b_seq = self.tokenizer(ref)['input_ids']

            dist = self._levenshtein(a_seq, b_seq)

            if normalize:
                denom = max(len(a_seq), len(b_seq))
                dist = (dist / denom) if denom > 0 else 0.0

            total += dist

        avg_dist = total / n
        return torch.tensor(avg_dist, dtype=torch.float32)
