import math
import torch
import torch.nn as nn

class NCE(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.decomposition = None
        self.loss_one_pos_vs_many_neg = self.params['energynet']['one_pos_vs_many_neg']
        self.loss_many_pos_vs_one_neg = self.params['energynet']['many_pos_vs_one_neg']
        self.ReLU = nn.ReLU()
        
    def forward(self, pos_pair, neg_pair = None):
        """
        pos_pair, neg_pair : [xypairs, inconsistent_pair_indices]
            xypairs: list of xypairs(=set).
                list length = batch_size
            inconsistent_pair_indiecs: list of list.
                Outer list length = batch_size
                Inner list: indices of inconsistent pairs
        """
        loss = 0
        e_pos = self.decomposition(pos_pair)["predictions"] # get the energy value for true pair
        e_neg = self.decomposition(neg_pair)["predictions"] # get the energy value for false pair
        e_pos_exp_minus = torch.exp(-e_pos)
        e_neg_exp_minus = torch.exp(-e_neg)

        pos_sum = torch.sum(e_pos_exp_minus)
        neg_sum = torch.sum(e_neg_exp_minus)
        
        # positive part
        if self.loss_one_pos_vs_many_neg:
            loss = loss + self.one_pos_vs_many_neg(e_pos_exp_minus, e_neg_exp_minus, pos_sum, neg_sum) / len(e_pos)

        # negative part
        if self.loss_many_pos_vs_one_neg:
            loss = loss - self.many_pos_vs_one_neg(e_pos_exp_minus, e_neg_exp_minus, pos_sum, neg_sum) / len(e_neg)

        return loss, {"e_pos": torch.sum(e_pos)/len(e_pos), "e_neg": torch.sum(e_neg)/len(e_neg)}
    
    def one_pos_vs_many_neg(self, e_pos_exp_minus, e_neg_exp_minus, pos_sum, neg_sum):
        denominators_pos = e_pos_exp_minus + neg_sum # size: (bat_size)
        L_pos = -pos_sum + torch.log(torch.sum(denominators_pos))
        return L_pos

    def many_pos_vs_one_neg(self, e_pos_exp_minus, e_neg_exp_minus, pos_sum, neg_sum):
        denominators_neg = e_neg_exp_minus + pos_sum # size: (bat_size)
        L_neg = -neg_sum + torch.log(torch.sum(denominators_neg))
        return L_neg
        

    def concat_energy(self, input_matrix):
        """
        input_matrix: list of list of str, shape: (batch_size, batch_size)
        """

        e_mat = [self.decomposition(m)["predictions"] for m in input_matrix]

        return torch.stack(e_mat, dim = 0)