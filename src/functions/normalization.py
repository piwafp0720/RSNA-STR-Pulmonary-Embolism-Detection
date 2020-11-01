
import torch


def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


def powerlaw(x, eps=1e-6):
    x = x + eps
    return x.abs().sqrt().mul(x.sign())
