"""
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, 
                 alpha=1, 
                 gamma=2, 
                 logits=False, 
                 reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, 
                inputs, 
                targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduce=False)
        else:
            bce_loss = F.binary_cross_entropy(
                inputs, targets, reduce=False)
        pt = torch.exp(- bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss