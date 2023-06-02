import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch._six import container_abcs
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.have_bias = bias

        W_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        b_shape = (self.out_channels)

        self.weight = nn.Parameter(torch.zeros(W_shape), requires_grad=True)
        if self.have_bias:
            self.bias = nn.Parameter(torch.zeros(b_shape), requires_grad=True)
        else:
            self.register_buffer("bias", torch.zeros(b_shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        W = self.weight
        if self.have_bias:
            b = self.bias
        else:
            b = torch.zeros((self.out_channels), device=self.weight.device)
        return F.conv2d(X, W, b, self.stride, self.padding, self.dilation)
    
    
class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1, bias=True):
        super(ConvTranspose2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.have_bias = bias

        W_shape = (self.in_channels, self.out_channels, *self.kernel_size)
        b_shape = (self.out_channels)

        self.weight = nn.Parameter(torch.zeros(W_shape), requires_grad=True)
        if self.have_bias:
            self.bias = nn.Parameter(torch.zeros(b_shape), requires_grad=True)
        else:
            self.register_buffer("bias", torch.zeros(b_shape))
        self.reset_parameters()
        
    def _output_padding(self, input, output_size,
                        stride, padding, kernel_size, dilation):
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    "output_size must have {} or {} elements (got {})"
                    .format(k, k + 2, len(output_size)))

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(k):
                dim_size = ((input.size(d + 2) - 1) * stride[d] -
                            2 * padding[d] +
                            (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                            output_size, min_sizes, max_sizes, input.size()[2:]))

            res = torch.jit.annotate(List[int], [])
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X):
        output_size = None
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
              
        W = self.weight
        if self.have_bias:
            b = self.bias
        else:
            b = torch.zeros((self.out_channels), device=self.weight.device)
            
        return F.conv_transpose2d(X, W, b, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
