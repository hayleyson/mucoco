from mucoco.losses import BaseLoss, register_loss

import torch 
import torch.nn.functional as F

import numpy as np

torch.set_printoptions(precision=3, sci_mode=False)

@register_loss("gpt2")
class GPT2Loss(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 

        self.model = model
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device
        self.mu = 1.0
        
        self.eos_token_id = self.tokenizer.eos_token_id    
        self.model.config.pad_token_id = self.model.config.eos_token_id # to remove the warning
        self.max_steps = args.coeff_steps
        self.coeff_schedule = args.coeff_pattern

        # if args.topic_target != "none":
        #     self.extra_prefix = self.tokenizer.encode(" "+args.topic_target+" "+args.topic_target, return_tensors="pt").to(self.device)
        # else:
        self.extra_prefix = torch.LongTensor([[]]).to(self.device)
    
    def get_coeff(self, step, seq_len, sched="constant"):
        if step == 0:
            self.freq = max(1, self.max_steps//seq_len)
            # print(self.freq)
            constant = [1.0 for i in range(seq_len-1)]
            if sched == "constant":
                decreasing = [1.0 for i in range(seq_len)]
            if sched == "exp":
                decreasing = [np.exp(-i) for i in range(seq_len)]
            elif sched == "linear":
                decreasing = [1-i/(seq_len-1) for i in range(seq_len)]
            elif sched == "rsqrt":
                decreasing = [1/np.sqrt(i+1) for i in range(seq_len)]
            elif sched == "rsqrt-freq":
                freq = 5
                decreasing = [1/np.sqrt((i+1)//freq) for i in range(seq_len)]
            # elif sched == "noam":

        
            # constant = [1.0 for i in range(seq_len-1)]
            # decreasing = [1.0 * (2.67**-i) for i in range(seq_len)]
            # increasing = [1.0 * (2.67**-i) for i in range(seq_len-1)]
            # decreasing = [1.0 - i/(seq_len-1) for i in range(seq_len)]
            # increasing = [1.0 - i/(seq_len-1) for i in range(seq_len-1)]
            # increasing.reverse()
            self.total = torch.Tensor(constant + decreasing).unsqueeze(0).to(self.device)
        
        i = max(0, seq_len - 1 - step // self.freq)
        return self.total[:, i:i + seq_len]
        # if step > seq_len - 1:
        #     return self.total[:, :seq_len]
        # else:
        #     return self.total[:, seq_len-1-step:2*seq_len-1-step]

    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        prompt, prefix = batch #prompt is the real deal, prefix can be provided as an extended prompt (generated by the model autoregressively)
        pred_tokens, pred_embeds, pred_probs = preds
        pred_probs = pred_probs[0]
        batch_size = prompt.size(0)
        step = kwargs.get("step")
        
        # if kwargs.get("use_context", False):
        #     # print("using context in the loss computation")
        #     context = kwargs.get('context_batch').squeeze(1) #only one context for now
        # else:
        context = torch.empty((batch_size, 0)).long().to(self.device)

        eos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.eos_token_id) 

        embed_lut = self.model.get_input_embeddings()
        # prompt = torch.cat([eos, prompt], dim=1)
        # print(self.extra_prefix)
        input_tokens = torch.cat([self.extra_prefix, prompt, prefix, pred_tokens, context], dim=1)
        input_embeds = torch.cat([embed_lut(self.extra_prefix), embed_lut(prompt), embed_lut(prefix), pred_embeds, embed_lut(context)], dim=1)
        # print(input_embeds.dtype)
        # input()
        preflen = prompt.size(1) + prefix.size(1) 
        predlen = pred_embeds.size(1)
        suflen = context.size(1) + 1

        # print(preflen, prompt.size(1), prompt, predlen)

        losstype = getattr(self.args, "loss_type", "xentropy")
        if losstype == "xentropy": #TODO
            model_output = self.model(inputs_embeds=input_embeds)
            lm_logits = model_output[0][:, preflen-1:]
            lm_logprobs = F.log_softmax(lm_logits, dim=-1)
            
            xentropy_pred = (-lm_logprobs[:, :-1] * pred_probs).sum(dim=-1).sum(dim=-1)

            #CONTEXT NOT INCORPORATED. IT'S ASSUMED TO BE EMPTY
            # xentropy_pred = xentropy_pred - lm_logprobs[:, -2, self.eos_token_id]

            # _, mm = lm_logprobs.max(dim=-1)

            xentropy = xentropy_pred
            if self.args.length_normalize:
                xentropy /= lm_logprobs.size(1)
            
            loss = xentropy

            logging_output = {
                "loss": loss.data.cpu(),
                "max_length": predlen+suflen,
                "nsentences": batch_size,
                "lm_logprobs": lm_logprobs.data.cpu(),
            }
        elif losstype in ["l2", "cosine", "dot", "dotplusplus", "detachdot", "detachdot2", "typical", "focal"]:
            model_output = self.model(inputs_embeds=input_embeds, step=step, go_inside="transformer")#, output_attentions=True)
            
            hidden_states = model_output[0]
            # attentions = model_output['attentions']
            # print(attentions[-1][0].max(dim=-1)[1])
            # print(hidden_states.dtype)
            # print(attentions[-1].size(), hidden_states.size())
            # input()
            
            entropy = None
            if losstype == "cosine":
                # print(input_embeds.size())
                # print(hidden_states.size())
                # input()
                hidden_states_unitnorm = torch.nn.functional.normalize(hidden_states, p=2, dim=-1).contiguous()
                pred_embs_unitnorm = torch.nn.functional.normalize(input_embeds[:, source.size(1)+pad_length+1:, :], p=2, dim=-1).contiguous()
                loss = (1.0 - (hidden_states_unitnorm * pred_embs_unitnorm).sum(dim=-1)).sum(dim=-1)
            
            elif losstype == "dot":
                k = kwargs.get("kweight")
                step = kwargs.get("step")               
                hidden_states = hidden_states[:, preflen-1:-1, :].contiguous()
                pred_embs = input_embeds[:, preflen:, :].contiguous()
                
                loss = -(hidden_states * pred_embs).sum(dim=-1) 

                coeff = self.get_coeff(step, hidden_states.size(1), sched=self.coeff_schedule)
                loss = coeff * loss - (coeff * loss).detach() + loss.detach()
                loss = loss.sum(dim=-1)         
                
            
            elif losstype == "detachdot":
                hidden_states = hidden_states[:, preflen-1:-1, :].contiguous()
                pred_embs = input_embeds[:, preflen:, :].contiguous()
                loss = -(hidden_states.detach() * pred_embs).sum(dim=-1) 
                # print("before", loss)
                
                logits = hidden_states.matmul(embed_lut.weight.t()).detach()
                maxlogit = logits.max(dim=-1, keepdim=True)[0]
                # print("ok", maxlogit)
                logits = logits - maxlogit
                # print("what", (hidden_states * pred_embs).sum(dim=-1))
                additive = torch.exp((hidden_states.detach() * pred_embs).sum(dim=-1) - maxlogit.squeeze(-1)) - torch.exp((hidden_states * pred_embs).sum(dim=-1) - maxlogit.squeeze(-1)).detach()
                # print(additive)
                lognorm = (logits.exp().sum(dim=-1) + additive).log()
                # lognorm = torch.logsumexp(hidden_states.matmul(embed_lut.weight.t()), dim=-1)
                lognorm = maxlogit.squeeze(-1) + lognorm 

                # loss += lognorm
                # print("lognorm1", lognorm)
                # lognorm = torch.logsumexp(hidden_states.matmul(embed_lut.weight.t()), dim=-1).detach()
                # print("lognorm2", lognorm)
                loss += lognorm
                # print("after", loss)
                loss = loss.sum(dim=-1)
            
            elif losstype == "detachdot2":
                k = kwargs.get("kweight")
                
                
                hidden_states = hidden_states[:, preflen-1:-1, :].contiguous()
                pred_embs = input_embeds[:, preflen:, :].contiguous()
                
                loss = -(hidden_states * pred_embs.detach()).sum(dim=-1)
                lognorm = torch.logsumexp(hidden_states.matmul(embed_lut.weight.t()), dim=-1)
                
                # coeff = min(1.0, (1.0*step)/predlen)
                # loss += coeff * hidden_contribution #+ (1 - coeff) * hidden_contribution.detach()
                loss += lognorm
                loss = loss.sum(dim=-1)

            elif losstype == "focal2": #unlikelihood (focal is the wrong name)
                k = kwargs.get("kweight")
                step = kwargs.get("step")
                
            
                hidden_states = hidden_states[:, preflen-1:-1, :].contiguous()#.detach()
                pred_embs = input_embeds[:, preflen:, :].contiguous()
                logits = hidden_states.matmul(embed_lut.weight.t())
                target_logits = (hidden_states * pred_embs).sum(dim=-1,keepdim=True)
                # print(logits.size())
                # print(target_logits.size())
                logits = logits - target_logits
                # print(logits.size())
                loss = torch.logsumexp(logits, dim=-1)
                # input(loss)

                #unlikelihood
                pairwise = hidden_states.detach().bmm(pred_embs.transpose(1, 2))
                L = pred_embs.size(1)
                bandwidth=10
                C = torch.tril(torch.ones((L, L))).to(self.device)
                C[bandwidth:, :L-bandwidth] = C[bandwidth:, :L-bandwidth] - torch.tril(torch.ones(L-bandwidth, L-bandwidth)).to(self.device)
                C = C.detach()

                pairwise = pairwise - target_logits
                # print(pairwise)
                # print(pairwise.size())
                
                # print((torch.exp(pairwise) * C))
                unlikelihood = (torch.exp(pairwise) * C).sum(dim=-1).log()                
                # print(unlikelihood)
                # input()
                # print(unlikelihood)
                # print(unlikelihood.size())
                # unlikelihood = unlikelihood.sum(dim=-1)
                # print(unlikelihood)
                # input()
                # print(unlikelihood)
                # unlikelihood = pairwise.sum(dim=-1)
                loss = 0.4*unlikelihood + loss
                
                # coeff = self.get_coeff(step, hidden_states.size(1), sched=self.coeff_schedule)
                # loss = coeff * loss - (coeff * loss).detach() + loss.detach()
                # print(coeff)
                loss = loss.sum(dim=-1)
            
            elif losstype == "dotplusplus":
                k = kwargs.get("kweight")
                step = kwargs.get("step")
                
                # self.begintemp = 1.0
                # self.finaltemp = 0.9
                # self.r = pow(self.finaltemp/self.begintemp, 1/19)

                # temperature = max(self.finaltemp, self.begintemp * pow(self.r, step))
                temperature = 1.0
                # hidden_states = hidden_states[:, preflen-1:preflen+predlen-1, :].contiguous()
                # pred_embs = input_embeds[:, preflen:preflen+predlen, :].contiguous()
                # loss1 = -(hidden_states1 * pred_embs).sum(dim=-1)
                # print(loss1.size())
                hidden_states = hidden_states[:, preflen-1:-1, :].contiguous()
                # hidden_states = 0.1 * hidden_states + (0.9 * hidden_states).detach()
                pred_embs = input_embeds[:, preflen:, :].contiguous()
                logits = hidden_states.matmul(embed_lut.weight.t()) 

                # temperature = logits.norm(dim=-1).detach()/1000
                # print(temperature)
                loss = -(hidden_states * pred_embs).sum(dim=-1) / temperature
                # logits = logits / temperature.unsqueeze(2)
                
                maxlogit = logits.max(dim=-1, keepdim=True)[0]
                # print("ok", maxlogit)
                logits = logits - maxlogit
                # print("what", (hidden_states * pred_embs).sum(dim=-1))
                additive = torch.exp((hidden_states * pred_embs).sum(dim=-1) / temperature - maxlogit.squeeze(-1)) - torch.exp((hidden_states * pred_embs.detach()).sum(dim=-1) / temperature - maxlogit.squeeze(-1))
                # print(additive)
                lognorm = (logits.exp().sum(dim=-1) + additive).log()
                # lognorm = torch.logsumexp(hidden_states.matmul(embed_lut.weight.t()), dim=-1)
                lognorm = maxlogit.squeeze(-1) + lognorm 
                # hidden_contribution = -0.5 * (hidden_states * pred_embs.detach()).sum(dim=-1) + torch.logsumexp(hidden_states.matmul(embed_lut.weight.t()), dim=-1)
                
                # coeff = min(1.0, (1.0*step)/predlen)
                # loss += coeff * hidden_contribution #+ (1 - coeff) * hidden_contribution.detach()
                loss += lognorm

                ##
                # loss2 = -(hidden_states * pred_embs.detach()).sum(dim=-1)
                # lognorm = torch.logsumexp(hidden_states.matmul(embed_lut.weight.t()), dim=-1)
                # loss2 += lognorm
                # ##

                # loss = loss + 0.9 * loss2 + (0.1 * loss2).detach()
                
                coeff = self.get_coeff(step, hidden_states.size(1), sched=self.coeff_schedule)
                loss = coeff * loss - (coeff * loss).detach() + loss.detach()
                # print(coeff)
                loss = loss.sum(dim=-1)
                # print(loss)
                # loss += torch.logsumexp(hidden_states.matmul(embed_lut.weight.t()), dim=-1) 
                # loss += torch.log(torch.exp(hidden_states.matmul(embed_lut.weight.t())).sum(dim=-1))
                # print(loss1.size())
                # print(loss)
                # loss1 = loss[:, :predlen]
                # loss2 = loss[:, predlen:]

                # hidden_states2 = hidden_states[:, preflen+predlen-1:-1, :].contiguous()
                # pred_embs = input_embeds[:, preflen+predlen:, :].contiguous()
                # # print(hidden_states2.size(1), pred_embs.size(1))
                # loss2 = -(hidden_states2 * pred_embs).sum(dim=-1)
                # # print(loss2.size())
                # # loss = -(hidden_states * pred_embs).sum(dim=-1) * (hidden_states * pred_embs.detach()).sum(dim=-1)
                # loss2 += torch.logsumexp(hidden_states2.matmul(embed_lut.weight.t()), dim=-1) 
                # print(loss2.size())
                # loss += torch.log(torch.exp(hidden_states.matmul(embed_lut.weight.t())).sum(dim=-1))
                # print(loss2.size(), loss1.size())
                # print(loss1, "loss2", loss2)

                # loss = loss1.sum(dim=-1) + k*loss2.sum(dim=-1)


                # print(loss)
                # loss = (-hidden_states * pred_embs).sum(dim=-1).sum(dim=-1)

                #entropy
                # logits = hidden_states.matmul(embed_lut.weight.t()) / temperature
                # # mask = top_k_top_p_filtering(logits, 0, 0.9)
                # # maxlogit = filtered_logits.max(dim=-1, keepdim=True)[0]
                # probs = F.softmax(logits, dim=-1)

                # entropy = (-probs * torch.log(probs)).sum(dim=-1)[:, :-suflen].detach()
                # print(entropy)
            
            # elif losstype == "focal":
            #     k = kwargs.get("kweight")
            #     step = kwargs.get("step")
            #     temperature = 1.0
                
            #     hidden_states = hidden_states[:, preflen-1:-1, :].contiguous()
            #     pred_embs = input_embeds[:, preflen:, :].contiguous()
            #     loss = -(hidden_states * pred_embs).sum(dim=-1) / temperature
            #     logits = hidden_states.matmul(embed_lut.weight.t()) / temperature
            #     maxlogit = logits.max(dim=-1, keepdim=True)[0]
            #     logits = logits - maxlogit
            #     additive = torch.exp((hidden_states * pred_embs).sum(dim=-1) / temperature - maxlogit.squeeze(-1)) - torch.exp((hidden_states * pred_embs.detach()).sum(dim=-1) / temperature - maxlogit.squeeze(-1))
                
            #     lognorm = (logits.exp().sum(dim=-1) + additive).log()
            #     lognorm = maxlogit.squeeze(-1) + lognorm 
            #     loss += lognorm ## -log p

            #     # gamma = 1.0
            #     # loss = (1 - torch.exp(-loss))**gamma * loss 

            #     eps = -1
            #     oneminusp = (1 - torch.exp(-loss))
            #     loss = loss + oneminusp*eps
            #     # loss = loss.sum(dim=-1)
                
            #     coeff = self.get_coeff(step, hidden_states.size(1), sched=self.coeff_schedule)
            #     loss = coeff * loss - (coeff * loss).detach() + loss.detach()
            #     loss = loss.sum(dim=-1)

            elif losstype == "typical":
                k = kwargs.get("kweight")
                step = kwargs.get("step")
                
                temperature = 1.0
                hidden_states = hidden_states[:, preflen-1:-1, :].contiguous().detach()
                pred_embs = input_embeds[:, preflen:, :].contiguous()
                loss = -(hidden_states * pred_embs).sum(dim=-1) / temperature
                logits = hidden_states.matmul(embed_lut.weight.t()) / temperature
                maxlogit = logits.max(dim=-1, keepdim=True)[0]
                logits = logits - maxlogit
                additive = torch.exp((hidden_states * pred_embs).sum(dim=-1) / temperature - maxlogit.squeeze(-1)) - torch.exp((hidden_states * pred_embs.detach()).sum(dim=-1) / temperature - maxlogit.squeeze(-1))
                
                lognorm = (logits.exp().sum(dim=-1) + additive).log()
                lognorm = maxlogit.squeeze(-1) + lognorm 
                loss += lognorm ## -log p
                
                # coeff = self.get_coeff(step, hidden_states.size(1), sched=self.coeff_schedule)
                
                #entropy
                logits = hidden_states.matmul(embed_lut.weight.t()) / temperature
                # mask = top_k_top_p_filtering(logits, 0, 0.9)
                # maxlogit = filtered_logits.max(dim=-1, keepdim=True)[0]
                probs = F.softmax(logits, dim=-1)

                entropy = (-probs * torch.log(probs)).sum(dim=-1)
                x = torch.abs(entropy - loss)
                loss = x + loss
                # loss = coeff * loss - (coeff * loss).detach() + loss.detach()
                loss = loss.sum(dim=-1)
                # print(x)
            elif losstype == "focal": #unlikelihood
                k = kwargs.get("kweight")
                step = kwargs.get("step")
                
                temperature = 1.0
                hidden_states = hidden_states[:, preflen-1:-1, :].contiguous()#.detach()
                pred_embs = input_embeds[:, preflen:, :].contiguous()
                loss = -(hidden_states * pred_embs).sum(dim=-1) / temperature
                logits = hidden_states.matmul(embed_lut.weight.t()) / temperature
                maxlogit = logits.max(dim=-1, keepdim=True)[0]
                logits = logits - maxlogit
                additive = torch.exp((hidden_states * pred_embs).sum(dim=-1) / temperature - maxlogit.squeeze(-1)) - torch.exp((hidden_states * pred_embs.detach()).sum(dim=-1) / temperature - maxlogit.squeeze(-1))
                
                lognorm = (logits.exp().sum(dim=-1) + additive).log()
                lognorm = maxlogit.squeeze(-1) + lognorm 
                loss += lognorm ## -log p
                
                # coeff = self.get_coeff(step, hidden_states.size(1), sched=self.coeff_schedule)
                
                #unlikelihood
                #pairwise dotproduct
                pairwise = -hidden_states.bmm(pred_embs.transpose(1, 2))
                C = (torch.triu(torch.ones((pred_embs.size(1), pred_embs.size(1)))) - torch.eye(pred_embs.size(1), pred_embs.size(1))).to(self.device).detach()
                # print(pairwise.size())
                # print(C)
                deno = torch.logsumexp(logits, dim=-1, keepdim=True).detach()
                # print(deno.size())
                pairwise = (pairwise + deno)* C / (C.sum(dim=-1, keepdim=True) + 1e-8)
                # print(pairwise)
                
                unlikelihood = pairwise.sum(dim=-1)
                # logits = hidden_states.matmul(embed_lut.weight.t()) / temperature
                # mask = top_k_top_p_filtering(logits, 0, 0.9)
                # maxlogit = filtered_logits.max(dim=-1, keepdim=True)[0]
                # probs = F.softmax(logits, dim=-1)

                # entropy = (-probs * torch.log(probs)).sum(dim=-1)
                # x = torch.abs(entropy - loss)
                # print(loss.sum(dim=-1), unlikelihood.sum(dim=-1))
                loss = -0.2*unlikelihood + loss
                # print(unlikelihood)
                # input()
                # loss = coeff * loss - (coeff * loss).detach() + loss.detach()
                loss = loss.sum(dim=-1)
                # print(x)
            else:
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds[:, source.size(1)+pad_length+1:, :].contiguous()
                loss = (hidden_states - pred_embs)
                loss = (loss*loss).sum(dim=-1).sum(dim=-1)
            
            if self.args.length_normalize:
                loss = loss/hidden_states.size(1)    

            logging_output = {
                "loss": loss.data.cpu(),
                "max_length": prefix.size(1) + pred_tokens.size(1),
                "nsentences": batch_size,
                "lm_logprobs": hidden_states.data.cpu(),
                "entropy": entropy
            }
        else:
            raise ValueError(f"wrong losstype provided: {losstype}")

        

        return loss, logging_output

    def compute_gold_loss(self, batch, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        prompt, target = batch
        batch_size = prompt.size(0)
        
        # if kwargs.get("use_context",False):
        #     print("using context")
        #     context = kwargs.get('context_batch').squeeze(1) #only one context for now
        # else:
        context = torch.empty((batch_size, 0)).long().to(self.device)
        
        # eos = torch.empty((batch_size, context.size(1), 1)).long().to(self.device).fill_(self.eos_token_id) 
        # input_tokens = torch.cat([prompt.unsqueeze(1).expand(-1, context.size(1), -1), target.unsqueeze(1).expand(-1, context.size(1), -1), context, eos], dim=1)

        # eos = torch.empty((batch_size, 1)).long().to(self.device).fill_(self.eos_token_id) 
        # input_tokens = torch.cat([prompt, target, context, eos], dim=1)
        input_tokens = torch.cat([prompt, target, context], dim=1)

        # print(input_tokens)
        losstype = getattr(self.args, "loss_type", "xentropy") 
        if losstype == "xentropy":
            model_output = self.model(input_tokens)
            # target = input_tokens[:, prompt.size(1):,]
            target = torch.cat([target, context], dim=1)

            lm_logits = model_output[0][:, prompt.size(1)-1:-1, :]
            lm_logprobs = F.log_softmax(lm_logits, dim=-1)

            loss = F.nll_loss(lm_logprobs.squeeze(0), target.squeeze(0), reduction="none").sum(dim=-1)
            
            if self.args.length_normalize:
                loss /= lm_logprobs.size(1)

            _, mm = lm_logprobs.max(dim=-1) # used for debugging

            logging_output = {
                "loss": loss.data.cpu(),
                "max_length": target.size(1),
                "nsentences": batch_size,
                "mm": mm,
            }
        elif losstype in ["l2", "cosine", "dot", "dotplusplus", "detachdot", "detachdot2", "typical", "focal"]:
            model_output = self.model.transformer(input_tokens)
            hidden_states = model_output[0][:, prompt.size(1)-1:-1]
            input_embeds = self.model.get_input_embeddings()(input_tokens)

            if losstype == "cosine":
                # print(input_embeds.size())
                # print(hidden_states.size())
                # input()
                
                hidden_states_unitnorm = torch.nn.functional.normalize(hidden_states, p=2, dim=-1).contiguous()
                pred_embs_unitnorm = torch.nn.functional.normalize(input_embeds[:, source.size(1)+pad_length:, :], p=2, dim=-1)[:, 1:, :].contiguous()
                loss = (1.0 - (hidden_states_unitnorm * pred_embs_unitnorm).sum(dim=-1)).sum(dim=-1)
            
            elif losstype == "dot":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds[:, prompt.size(1):, :].contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1)
                loss = loss.sum(dim=-1)
            
            elif losstype == "dotplusplus" or losstype == "detachdot" or losstype == "detachdot2":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds[:, prompt.size(1):, :].contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1)
                # print(loss)
                # print(hidden_states.matmul(self.model.get_input_embeddings().weight.t()))
                # print(torch.logsumexp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()), dim=-1))
                # print(torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t())).sum(dim=-1)))
                # input("gold")
                loss += torch.logsumexp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()), dim=-1)
                # print()
                # loss += torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()).sum(dim=-1)))
                loss = loss.sum(dim=-1)
            
            elif losstype == "focal":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds[:, prompt.size(1):, :].contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1)
                # print(loss)
                # print(hidden_states.matmul(self.model.get_input_embeddings().weight.t()))
                # print(torch.logsumexp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()), dim=-1))
                # print(torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t())).sum(dim=-1)))
                # input("gold")
                loss += torch.logsumexp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()), dim=-1)
                # print()
                # loss += torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()).sum(dim=-1)))

                eps = -1

                loss = loss + (1 - torch.exp(-loss))*eps
                loss = loss.sum(dim=-1)
                
                # coeff = self.get_coeff(step, hidden_states.size(1), sched=self.coeff_schedule)
                # loss = coeff * loss - (coeff * loss).detach() + loss.detach()
                # loss = loss.sum(dim=-1)
            
            elif losstype == "typical":
                hidden_states = hidden_states.contiguous()
                pred_embs = input_embeds[:, prompt.size(1):, :].contiguous()

                loss = -(hidden_states * pred_embs).sum(dim=-1)
                # print(loss)
                # print(hidden_states.matmul(self.model.get_input_embeddings().weight.t()))
                # print(torch.logsumexp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()), dim=-1))
                # print(torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t())).sum(dim=-1)))
                # input("gold")
                loss += torch.logsumexp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()), dim=-1)
                # print()
                # loss += torch.log(torch.exp(hidden_states.matmul(self.model.get_input_embeddings().weight.t()).sum(dim=-1)))

                #entropy 
                logits = hidden_states.matmul(self.model.get_input_embeddings().weight.t()) 
                # mask = top_k_top_p_filtering(logits, 0, 0.9)
                # maxlogit = filtered_logits.max(dim=-1, keepdim=True)[0]
                probs = F.softmax(logits, dim=-1)

                entropy = (-probs * torch.log(probs)).sum(dim=-1)

                loss = (entropy + loss).sum(dim=-1)

            else:
                hidden_states = hidden_states[:, :-1, :].contiguous()
                pred_embs = input_embeds[:, source.size(1)+pad_length+1:, :].contiguous()
                loss = (hidden_states - pred_embs)
                loss = (loss*loss).sum(dim=-1).sum(dim=-1)
            
            if self.args.length_normalize:
                loss = loss/(hidden_states.size(1)-1)

            logging_output = {
                "loss": loss.data.cpu(),
                "max_length": target.size(1),
                "nsentences": batch_size,
                "lm_logprobs": hidden_states.data.cpu()
            }
        else:
            raise ValueError(f"wrong losstype provided: {losstype}")

        return loss, logging_output   
    
    def generate(self, input_ids, **kwargs):
        prepared_input = self._prepare_input_for_generation(input_ids, **kwargs)
        output = self.model.generate(**prepared_input)
        # print(self.model.get_input_embeddings().weight)
        # print(str(**prepared_input))
        # print("gen", output)
        
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
                # 'pad_token_id':self.eos_token_id} 
        # print(return_object)

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
        # print(sorted_indices)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # print(sorted_indices_to_remove, sorted_indices)
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=2, index=sorted_indices, src=sorted_indices_to_remove)
        # print(filter_value)
        # print(indices_to_remove)
        # input("topp")
        # logits[indices_to_remove] = filter_value
        # print(indices_to_remove)
        # input()
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
        max_output_length = np.random.normal(loc=max_output_length_mean, scale=10, size=None)
        max_output_length = int(min(max(max_output_length, max_output_length_mean-10), max_output_length_mean+10))
        print(max_output_length)
        
        batch_size = input_ids.size(0)
        #batch size is 1, padding and stuff needs to be modified for this to work for larger batches

        return_object = {'input_ids': input_ids,
                'max_length': input_ids.size(1) + max_output_length,
                'do_sample': True,
                'temperature': self.args.AR_temperature,
                'top_k': self.args.AR_top_k,
                'top_p': self.args.AR_top_p,
                'num_return_sequences': 1}
                # 'pad_token_id':self.eos_token_id} 
        # print(return_object)

        return return_object
    
    def generate(self, input_ids, **kwargs):
        num_sequences = kwargs.get('num_return_sequences', 1)
        outputs = []
        seq_lengths = []
        
        for i in range(num_sequences):
            
            prepared_input = self._prepare_input_for_generation(input_ids, **kwargs)
            output = self.model.generate(**prepared_input)
            outputs.append(self._postprocess_output(prepared_input, output))
            seq_lengths.append(prepared_input['max_length'])
            # print(self.model.get_input_embeddings().weight)
            # print(str(**prepared_input))
            # print("gen", output)
            
        return outputs, seq_lengths