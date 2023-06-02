import torch
import torch.nn as nn

from distributions import Distribution


class ConditionalDistribution(Distribution):
    """ConditionalDistribution base class"""

    def __init__(self, net):
        super(ConditionalDistribution, self).__init__()
        self.net = net

    def get_parameters(self):
        return filter(lambda p: p.requires_grad, self.net.parameters())

    def log_prob(self, x, context):
        """Calculate log probability under the distribution.

        Args:
            x: Tensor, shape (batch_size, ...).
            context: Tensor, shape (batch_size, ...).

        Returns:
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def sample(self, context):
        """Generates samples from the distribution.

        Args:
            context: Tensor, shape (batch_size, ...).

        Returns:
            samples: Tensor, shape (batch_size, ...).
        """
        raise NotImplementedError()

    def sample_with_log_prob(self, context):
        """Generates samples from the distribution together with their log probability.

        Args:
            context: Tensor, shape (batch_size, ...).

        Returns::
            samples: Tensor, shape (batch_size, ...).
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()


