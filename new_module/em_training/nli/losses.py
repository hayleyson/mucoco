import torch
import torch.nn as nn


class MSE_MarginRankingLoss(nn.Module):
    
    def __init__(self, weights, margin):
        
        super(MSE_MarginRankingLoss, self).__init__()
        self.margin = margin
        self.weights = weights
        self.mse_loss = torch.nn.MSELoss()
        self.margin_ranking_loss = CustomMarginRankingLoss(self.margin)
        
    def forward(self, predictions, labels, fy_i, fy_1_i):
        mse_val = self.mse_loss(predictions, labels)
        mrl_val = self.margin_ranking_loss(fy_i, fy_1_i)
        loss_sum = self.weights[0] * mse_val + self.weights[1] * mrl_val
        return loss_sum
        

class KendallsTauLoss(nn.Module):
    
    def __init__(self):
        super(KendallsTauLoss, self).__init__()
    
    def forward(self, predictions, true_labels):
        
        # Assumes predictions and true_labels are 1D tensors of equal length
        n = predictions.size(0)
        numerator = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                numerator += torch.sign(predictions[i] - predictions[j]) * torch.sign(true_labels[i] - true_labels[j])
        loss = 1.0 - (2 * numerator / (n * (n - 1)))  # Normalizing Kendall's Tau to be in [0, 1]
        return loss


class PairwiseLogisticLoss(nn.Module):
    """
    Implementation of loss from "Learning to Summarize from Human Feedback" paper.
    """
    
    def __init__(self):
        super(PairwiseLogisticLoss, self).__init__()

    def forward(self, fy_i, fy_1_i):
        return - torch.mean(torch.log(torch.sigmoid(fy_i - fy_1_i)))

class CustomMarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super(CustomMarginRankingLoss, self).__init__()
        self.margin = margin
        self.loss_func = nn.MarginRankingLoss(margin=self.margin)
    
    def forward(self, fy_i, fy_1_i):
        return self.loss_func(fy_i, fy_1_i, torch.ones_like(fy_i, dtype=torch.long))
        

def create_pairs_for_ranking(logits, labels):
    """ Given model predictions(logits or probabilities) and ground truth labels of a list of examples, 
    fetch all possible pairs from the examples, compare the ground truth values of each pair, 
    and return two lists where the first list contains the model predictions for items in pairs that have higher g.t. and
    the second list that contains the predictions for items with lower g.t..
    """
    first = []
    second = []
    num_samples = len(labels)
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            if labels[i] > labels[j]:
                first.append(i)
                second.append(j)
            elif labels[i] < labels[j]:
                first.append(j)
                second.append(i)
    better_logits = logits[first]
    worse_logits = logits[second]

    return better_logits, worse_logits


    
    