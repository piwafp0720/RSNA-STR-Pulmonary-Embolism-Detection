import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import models

from . import rexnet
from .layers import pooling


class SimpleClassifier(nn.Module):
    def __init__(self,
                 base,
                 pretrained: bool = True,
                 pool_type: str = 'avg',
                 n_features: int = 320,
                 n_hiddens: int = 256,
                 n_classes: int = 1,
                 drop_rate: float = 0.):
        super().__init__()
        if 'rexnet' in base:
            self.base = getattr(rexnet, base)(pretrained=pretrained)
        else:
            self.base = getattr(models, base)(pretrained=pretrained)

        if pool_type == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'max':
            self.pooling = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gem':
            self.pooling = pooling.GeM()
        elif pool_type == 'mac':
            self.pooling = pooling.MAC()
        else:
            raise KeyError

        self.drop_rate = drop_rate
        self.cls_layer1 = nn.Linear(n_features, n_hiddens)
        self.cls_bn1 = nn.BatchNorm1d(n_hiddens)
        self.cls_layer2 = nn.Linear(n_hiddens, n_classes)
    
    def forward_until_pooling(self, x):
        x = self.base.forward_features(x)
        x = self.pooling(x).squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        x = self.forward_until_pooling(x)

        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        x = torch.relu(self.cls_bn1(self.cls_layer1(x)))

        return self.cls_layer2(x)

