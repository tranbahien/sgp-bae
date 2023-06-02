import torch


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def inv_sigmoid(x):
    return torch.log(x) - torch.log(1 - x)
