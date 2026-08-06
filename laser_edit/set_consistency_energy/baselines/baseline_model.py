from .LLM.GPT import GPT
from .LLM.HFModel import HFModel
from .LLM.LiteLLM import LitellmLLM
from .LLM.VllmLLM import VllmLLM
class baseline_model():
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        self.baseline_type = None
        self.model_name = None
        self.model = None
        self.initialize()
        if 'gpt' in self.model_name.lower() or 'deepseek' in self.model_name.lower():
            self.locate = self.locate_gpt
        else:
            self.locate = self.locate_vllm_hf
                
    def initialize(self):
        self.baseline_type = self.params['baseline']['type']
        if self.baseline_type.lower() == 'llm':
            self.model_name = self.params['baseline']['model']
            if 'openai/' in self.model_name.lower() or 'anthropic/' in self.model_name.lower() or 'gemini/' in self.model_name.lower():
                self.model = LitellmLLM(self.params)
            elif 'gpt' in self.model_name.lower() or 'deepseek' in self.model_name.lower():
                print("GPT models")
                self.model = GPT(self.params)
            else:
                if self.params['baseline']['use_vllm']: 
                    self.model = VllmLLM(self.params)
                else:
                    self.model = HFModel(self.params)
        
        elif self.baseline_type.lower() == 'nli':
            raise NotImplementedError
        
        else:
            print(f"Invalid baseline_type. You gave '{self.baseline_type}'.")
            raise NotImplementedError
        
    def predict(self, pairs):

        pred = self.model.predict(pairs)

        gold = []
        for p in pairs:
            incon_list = p[1]
            if len(incon_list) == 0:
                gold += [0]
            else:
                gold += [1]

        return {
            "pred": pred,
            "gold": gold,
        }
    
    
    def locate_gpt(self, pair):
        # assert len(pair) == 1

        gold = pair[0][1]               # list of true inconsistent pairs
        gold = [g+1 for g in gold]
        result = self.model.locate(pair)  # list of predicted inconsistent pairs
        pred = result['pred_list']
        pred_str = result['pred_str']
        r_tok = result['reasoning_tokens']
        t_tok = result['total_generated_tokens']
        
        pair_num = len(pair[0][0])
        if set(pred) == set(gold):
            correct =1
        else:
            correct = 0

        # precision
        if len(pred) == 0:
            precision = 1
        else:
            precision = len(set(pred) & set(gold)) / len(set(pred))

        # recall
        if len(gold) == 0:
            recall = 1
        else:
            recall = len(set(pred) & set(gold)) / len(set(gold)) 

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2*precision*recall / (precision + recall)
        accuracy = correct
            
        return {
            "pair_num": pair_num,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "gold": gold,
            "pred": pred,
            "raw_response": pred_str,
            "reasoning_tokens": r_tok,
            "total_generated_tokens": t_tok
        }

    
    def locate_vllm_hf(self, pairs):
        
        if len(pairs) == 0:
            return {
                "precision": 1,
                "recall": 1,
                "f1": 1,
                "accuracy": 1,
                "gold_list": [],
                "pred_list": [],
                "raw_response_list": [],
                "reasoning_tokens_list": [],
                "total_generated_tokens_list": []
            }

        
        pair_nums = [len(x[0]) for x in pairs]
        gold_list = [x[1] for x in pairs] # list of true inconsistent pairs
        print(f"gold_list: {gold_list}")
        gold_list = [[g + 1 for g in gold] for gold in gold_list]

        pred_list, raw_response_list, r_toks_list, t_toks_list = self.model.locate(pairs)  # list of predicted inconsistent pairs

        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        for gold, pred in zip(gold_list, pred_list):

            if set(pred) == set(gold):
                correct =1
            else:
                correct = 0

            # precision
            if len(pred) == 0:
                precision = 1
            else:
                precision = len(set(pred) & set(gold)) / len(set(pred))

            # recall
            if len(gold) == 0:
                recall = 1
            else:
                recall = len(set(pred) & set(gold)) / len(set(gold)) 

            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2*precision*recall / (precision + recall)
            accuracy = correct
            
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
                
        return {
            "pair_num": pair_nums,
            "precision": sum(precision_list) / len(precision_list),
            "recall": sum(recall_list) / len(recall_list),
            "f1": sum(f1_list) / len(f1_list),
            "accuracy": sum(accuracy_list) / len(accuracy_list),
            "gold_list": gold_list,
            "pred_list": pred_list,
            "raw_response_list": raw_response_list,
            "reasoning_tokens_list": r_toks_list,
            "total_generated_tokens_list": t_toks_list
        }