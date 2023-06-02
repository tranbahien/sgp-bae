import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.layers import LambdaLayer
from nn.layers import act_module
from torch.nn import Linear, Conv2d, ConvTranspose2d
from torch.nn import BatchNorm2d, Flatten, ConstantPad2d


class ConvNet(nn.Module):
    def __init__(self, input_size, output_size, n_channels,
                 kernel_sizes, strides, paddings,
                 activation='relu', in_lambda=None, out_lambda=None,
                 last_fc_layer=True):
        super(ConvNet, self).__init__()
        self.last_fc_layer = last_fc_layer
        
        conv_layers = []
        if in_lambda: conv_layers.append(LambdaLayer(in_lambda))
        
        idxs = list(range(len(n_channels)))
        for idx in range(len(n_channels)):
            if idx == 0:
                in_size, out_size = input_size, n_channels[0]
            else:
                in_size, out_size = n_channels[idx-1], n_channels[idx]
            conv_layers.append(Conv2d(in_size, out_size,
                                      kernel_size=kernel_sizes[idx-1],
                                      padding=paddings[idx-1],
                                      stride=strides[idx-1]))
#             conv_layers.append(BatchNorm2d(out_size, affine=False,
#                                            track_running_stats=False))
            conv_layers.append(act_module(activation))
        
        self.convs = nn.Sequential(*conv_layers)
        
        if last_fc_layer:
            fc_layers = []
            fc_layers.append(Linear(n_channels[-1], output_size))
            if out_lambda: fc_layers.append(LambdaLayer(out_lambda))

            self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.convs(x)
        x = x.squeeze()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        if self.last_fc_layer:
            x = self.fc(x)
        
        return x

    
class InvConvNet(nn.Module):
    def __init__(self, input_size, output_size, n_hiddens, n_channels,
                 kernel_sizes, strides, paddings,
                 activation='relu', in_lambda=None, out_lambda=None,
                 mid_lambda=None):
        super(InvConvNet, self).__init__()

        fc_layers = []
        if in_lambda: fc_layers.append(LambdaLayer(in_lambda))
        for in_size, out_size in zip([input_size] + n_hiddens[:-1], n_hiddens):
            fc_layers.append(Linear(in_size, out_size))
            
            # TODO:
            fc_layers.append(act_module(activation))
        if mid_lambda: fc_layers.append(LambdaLayer(mid_lambda))

        conv_layers = []
        
        for idx in range(len(n_channels)-1):
            in_size, out_size = n_channels[idx], n_channels[idx+1]
            conv_layers.append(ConvTranspose2d(in_size, out_size,
                                               kernel_size=kernel_sizes[idx],
                                               stride=strides[idx],
                                               padding=paddings[idx]))
#             conv_layers.append(BatchNorm2d(out_size, affine=False,
#                                            track_running_stats=False))
            conv_layers.append(act_module(activation))

        conv_layers.append(ConvTranspose2d(n_channels[-1], output_size,
                                           kernel_size=kernel_sizes[-1],
                                           stride=strides[-1],
                                           padding=paddings[-1]))
        if out_lambda: conv_layers.append(LambdaLayer(out_lambda))

        self.fc = nn.Sequential(*fc_layers)
        self.convs = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.fc(x)
        x = self.convs(x)
        return x

