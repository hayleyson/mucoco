from evaluation.prompted_sampling.evaluate import load_sc_energy_model
from new_module.locate.new_locate_utils import LocateMachine4SCE
from new_module.dev_utils.utils import read_outputs
import yaml 
from typing import List, Tuple
import os
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import argparse
import pickle

class CustomLocateMachine4SCE(LocateMachine4SCE):
    def locate_instance(self, prediction: List[str], max_num_tokens: int = 6, unit: str = "word",**kwargs) -> Tuple[List[str], List[List[int]]]:

        """
        Locate a instance (a pair) within the input set (set of pairs). 
        
        Suppose input text looks like '<s> q1 </s> a1 </s>, q1, ..., </s>'. 
        The located instance can be anywhere in q1, a2, q2, a2, ...
        """
        
        # logger.debug(f"[new_locate_utils] prediction before adding cls token: {prediction}")
        prediction = [self.tokenizer.cls_token + " " + p.lstrip(self.tokenizer.cls_token).lstrip(" ") for p in prediction]
        # logger.debug(f"[new_locate_utils] prediction after adding cls token: {prediction}")
        outputs, hidden_states_or_attentions = self.energynet.energy_model(prediction, pair_only = True)
        # Calculate token scores
        token_scores = self._calculate_token_scores(outputs, hidden_states_or_attentions)
        
        # set additional information
        inputs = self._instance_preserving_encode_plus(prediction)
        input_tensor = inputs['input_ids']
        mask = inputs['attention_mask']
        # logger.debug(f"input_tensor: {input_tensor}")
        # logger.debug(f"mask: {mask}")
        
        _, instance_locations = self._detect_instance(prediction) 
           
        batch_size = input_tensor.shape[0]
        assert batch_size == 1 # this code assumes batch_size = 1
            
        # initialize return variables
        prediction_list = []
        masked_sequence_text = []
        
        # Apply attention and stopwords mask. Then take softmax
        final_mask = (mask == 0) | torch.isin(input_tensor, self.stopwords_ids)
        token_scores[final_mask] = -float("inf")
        token_scores = token_scores.softmax(dim=-1)
        
        # Filter out degenerate instances (those with only masked tokens)
        # This prevents division by zero errors in instance scoring
        filtered_instance_locations = []
        filtered_instance_indexes = []
        for b in range(batch_size):
            valid_instances = []
            for j, (start, end) in enumerate(instance_locations[b]):
                # Check if instance has at least one non-masked token
                instance_has_nonmasked = (~final_mask[b][start:end]).any().item()
                if instance_has_nonmasked:
                    valid_instances.append((start, end))
                    filtered_instance_indexes.append(j)
                else:
                    logger.info(f"Filtering out degenerate instance at ({start}, {end}) with only masked tokens")
            filtered_instance_locations.append(valid_instances)
        
        # Update instance_locations to use only valid instances
        instance_locations = filtered_instance_locations
        
        # First locate at instance-level
        prediction_list = self._locate_instance(token_scores, instance_locations, batch_size)
        prediction_list_adjusted = [filtered_instance_indexes[prediction_list[0][0]]]

        # Create text with the located instance removed
        predicted_instance_start, predicted_instance_end = instance_locations[0][prediction_list[0][0]]
        
        new_input_tensor = torch.cat([input_tensor[:, :predicted_instance_start], input_tensor[:, predicted_instance_end:]], axis=-1) if predicted_instance_start > 0 else input_tensor[:, predicted_instance_end:]
        new_prediction = self.tokenizer.batch_decode(new_input_tensor)
        new_prediction = new_prediction[0].strip('<s>').strip(' ')

        predicted_instance_tensor = input_tensor[:, predicted_instance_start: predicted_instance_end]
        predicted_instance = self.tokenizer.batch_decode(predicted_instance_tensor)

        return new_prediction,predicted_instance, prediction_list_adjusted
    
    
    def verify_consistency(self, text):
        

        with torch.no_grad():
            # set consistency verification
            batch_text = ['<s> ' + text]
            output, _ = self.energynet.energy_model(batch_text, pair_only = True)
            
            if (self.energynet.output_form == 'real_num'):
                probs = output.reshape(-1)
            elif (self.energynet.output_form == '2dim_vec'):
                probs = F.softmax(output, dim=-1)[:,-1]
            else:
                raise ValueError(f"Unsupported output form: {self.energynet.output_form}")
        
            #  classify

            cons = torch.where(probs <= self.energynet.threshold,1,0).item()
        if cons == 1:
            return 'con'
        else:
            return 'incon'

    def detect_spans_recursive_max(self, text):
        predicted_indexes = []
        remaining_index_list = list(range(len(self.detect_span(text))))

        while ((self.verify_consistency(text) == 'incon') and len(text) > 0):
            
            text,_, index_list = self.locate_instance([text])
            index = index_list[0]
            
            predicted_indexes.append(remaining_index_list[index])
            remaining_index_list = (remaining_index_list[:index] if index > 0 else []) + remaining_index_list[index+1:] 
        
        return predicted_indexes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--agg_method', type=str, default='')
    parser.add_argument('--attentions_num_layer', type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    config['device'] = args.device
    model = load_sc_energy_model(args.config, args.device)

    if args.agg_method != '':
        config['locate']['agg_method'] = args.agg_method
        print(f"Overrode agg_method to {args.agg_method}.")
    locator = CustomLocateMachine4SCE(config, model, args.task)
    
    data = read_outputs(args.data_path)
    incon_data = data.loc[data['label'] == 'incon'].copy()
    
    metric_store = []

    if config['locate']['type'] == 'attention':

        if args.attentions_num_layer != -1:
            num_layer_list = [args.attentions_num_layer]
        else:
            num_layer_list = list(range(12))

        for num_layer in tqdm(num_layer_list):

            locator.params['locate']['attentions_num_layer'] = num_layer
            
            precisions = []
            recalls = []
            for ix, row in tqdm(incon_data.iterrows()):
                predicted_locate_labels = locator.detect_spans_recursive_max(row['text'])
                gold_locate_labels = row['locate_labels']
                if len(predicted_locate_labels) == 0:
                    precisions.append(0)
                    continue
                precision = len(set(predicted_locate_labels) & set(gold_locate_labels)) / len(predicted_locate_labels)
                precisions.append(precision)

                recall = len(set(predicted_locate_labels) & set(gold_locate_labels)) / len(gold_locate_labels)
                recalls.append(recall)
            
            metric_store_ = deepcopy(locator.params['locate'])
            metric_store_['precision'] = np.mean(precisions)

            metric_store_['recall'] = np.mean(recalls)
            metric_store.append(metric_store_)
            print(f'Layer {num_layer}: Average Precision = {np.mean(precisions)}, Average Recall = {np.mean(recalls)}')

    elif config['locate']['type'] == 'gradnorm':

        precisions = []
        recalls = []
        for ix, row in tqdm(incon_data.iterrows()):
            predicted_locate_labels = locator.detect_spans_recursive_max(row['text'])
            gold_locate_labels = row['locate_labels']
            if len(predicted_locate_labels) == 0:
                precisions.append(0)
                continue
            
            precision = len(set(predicted_locate_labels) & set(gold_locate_labels)) / len(predicted_locate_labels)
            precisions.append(precision)

            recall = len(set(predicted_locate_labels) & set(gold_locate_labels)) / len(gold_locate_labels)
            recalls.append(recall)
        
        metric_store_ = deepcopy(locator.params['locate'])
        metric_store_['precision'] = np.mean(precisions)
        metric_store_['recall'] = np.mean(recalls)
        metric_store.append(metric_store_)
        print(f'Average Precision = {np.mean(precisions)}, Average Recall = {np.mean(recalls)}')

    if os.path.exists(args.output_dir + '/locate_params_eval_result.csv'):
        print(f"{args.output_dir + '/locate_params_eval_result.csv'} exists.")
        print('Loading from existing eval result file...')
        prev_metric_store = pd.read_csv(args.output_dir + '/locate_params_eval_result.csv')
        print(f"Loaded {len(prev_metric_store)} results.")
        metric_store = pd.concat([prev_metric_store, pd.DataFrame.from_dict(metric_store)], axis=0, ignore_index=True)
    else:
        metric_store = pd.DataFrame.from_dict(metric_store)
        
    metric_store.to_csv(args.output_dir + '/locate_params_eval_result.csv', index=False)
    print(f"Eval result saved to {args.output_dir + '/locate_params_eval_result.csv'}")

if __name__ == '__main__':
    main()