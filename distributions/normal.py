import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Distribution
from utils import sum_except_batch


class DiagonalNormal(Distribution):
    """A multivariate Normal with diagonal covariance."""

    def __init__(self, shape, trainable=False):
        super(DiagonalNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.loc = nn.Parameter(torch.zeros(shape), requires_grad=trainable)
        self.log_scale = nn.Parameter(torch.zeros(shape), requires_grad=trainable)

    def log_prob(self, x):
        log_base =  - 0.5 * math.log(2 * math.pi) - self.log_scale
        log_inner = - 0.5 * torch.exp(-2 * self.log_scale) * ((x - self.loc) ** 2)
        return sum_except_batch(log_base+log_inner)

    def sample(self, num_samples):
        eps = torch.randn(num_samples, *self.shape, device=self.loc.device, dtype=self.loc.dtype)
        return self.loc + self.log_scale.exp() * eps

