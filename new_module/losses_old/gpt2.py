import numpy as np
import torch
import torch.nn.functional as F

from new_module.losses_old import BaseLoss, register_loss

torch.set_printoptions(precision=3, sci_mode=False)

@register_loss("gpt2")
class GPT2Loss(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 

        self.model = model
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device
        
        self.eos_token_id = self.tokenizer.eos_token_id    
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id # to remove the warning
    
    def compute_gold_loss(self, prompt, prediction, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        prompt = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True).to(self.device).long()
        # assuming batch size of 1 (prediction is a string instance.)
        prediction = self.tokenizer.encode(prediction, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True).to(self.device).long()
        input_tokens = torch.cat([prompt, prediction], dim=1)
        model_output = self.model(input_tokens)

        lm_logits = model_output[0][:, prompt.size(1)-1:-1, :]
        lm_logprobs = F.log_softmax(lm_logits, dim=-1)

        loss = F.nll_loss(lm_logprobs.squeeze(0), prediction.squeeze(0), reduction="none").sum(dim=-1)
        
        if self.args.length_normalize:
            loss /= lm_logprobs.size(1)

        return loss
    
    def generate(self, input_ids, **kwargs):
        prepared_input = self._prepare_input_for_generation(input_ids, **kwargs)
        output = self.model.generate(**prepared_input)
        
        return self._postprocess_output(prepared_input, output)

    def _prepare_input_for_generation(self, input_ids, **kwargs):
        max_output_length = getattr(self.args, "max_output_length", 10)
        batch_size = input_ids.size(0)
        #batch size is 1, padding and stuff needs to be modified for this to work for larger batches

        return_object = {'input_ids': input_ids,
                'max_length': input_ids.size(1) + max_output_length,
                'do_sample': True,
                'temperature': self.args.AR_temperature,
                'top_k': self.args.AR_top_k,
                'top_p': self.args.AR_top_p,
                'num_return_sequences': kwargs.get('num_return_sequences', 1)}
   
        return return_object
    
    def _postprocess_output(self, prepared_input, output_ids):
        return output_ids[:, prepared_input['input_ids'].size(1):, ]

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), filter_indices=[]):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            filter_indices: do not predict the given set of indices.
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(dim=2, index=sorted_indices, src=sorted_indices_to_remove)

        mask = torch.ones_like(logits)
        mask[indices_to_remove] = 0.0
        return mask

    elif top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if len(filter_indices) > 0:
        pass
    
@register_loss("gpt2-var-length")
class GPT2VarLengthLoss(GPT2Loss):
    def _prepare_input_for_generation(self, input_ids, **kwargs):
        max_output_length_mean = getattr(self.args, "max_output_length", 10)
        max_output_length = int(np.random.uniform(low = max_output_length_mean-10, high=max_output_length_mean+10, size=None))
        print(f"max_output_length: {max_output_length}")
        
        batch_size = input_ids.size(0)
        #batch size is 1, padding and stuff needs to be modified for this to work for larger batches

        return_object = {'input_ids': input_ids,
                'max_length': input_ids.size(1) + max_output_length,
                'do_sample': True,
                'temperature': self.args.AR_temperature,
                'top_k': self.args.AR_top_k,
                'top_p': self.args.AR_top_p,
                'num_return_sequences': 1}

        return return_object
    
    def generate(self, input_ids, **kwargs):
        num_sequences = kwargs.get('num_return_sequences', 1)
        outputs = []
        seq_lengths = []
        
        for i in range(num_sequences):
            
            prepared_input = self._prepare_input_for_generation(input_ids, **kwargs)
            output = self.model.generate(**prepared_input)
            outputs.append(self._postprocess_output(prepared_input, output))
            seq_lengths.append(prepared_input['max_length'] - prepared_input['input_ids'].size(1)) # subtract prompt length
            
        return outputs, seq_lengths