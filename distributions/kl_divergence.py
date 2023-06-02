import torch

from torch.distributions import kl_divergence, register_kl, Normal

from distributions import DiagonalNormal
from utils import sum_except_batch


@register_kl(Normal, DiagonalNormal)
def _kl_normal_normal(q, p):
    q_loc = q.loc
    p_loc = p.loc

    q_logvars = 2 * torch.log(q.scale)
    p_logvars = 2 * p.log_scale

    kl = 0.5 * sum_except_batch(
        p_logvars - q_logvars
        + (torch.pow(q.loc - p.loc, 2) * torch.exp(-p_logvars))
        + torch.exp(q_logvars - p_logvars) - 1.0
    )

    return kl
