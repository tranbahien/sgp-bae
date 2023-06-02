import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.layers import LambdaLayer
from nn.layers import act_module
from nn.layers import Linear


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_units,
                 activation='relu', in_lambda=None, out_lambda=None):
        super(MLP, self).__init__()
        
        layers = []
        
        if in_lambda: layers.append(LambdaLayer(in_lambda))
        
        if len(hidden_units) > 1:
            for in_size, out_size in zip([input_size] + hidden_units[:-1], hidden_units):
                layers.append(Linear(in_size, out_size))
                layers.append(act_module(activation))

            layers.append(Linear(hidden_units[-1], output_size))
        else:
            layers.append(Linear(input_size, hidden_units[0]))
            layers.append(act_module(activation))
            layers.append(Linear(hidden_units[0], output_size))
        
        if out_lambda: layers.append(LambdaLayer(out_lambda))
            
        self.layers = nn.Sequential(*layers)

        
    def forward(self, X):
        return self.layers(X)