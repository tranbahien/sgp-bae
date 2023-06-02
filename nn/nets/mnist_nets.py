import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.layers import LambdaLayer
from nn.layers import act_module
from torch.nn import Linear, Conv2d, ConvTranspose2d
from torch.nn import BatchNorm2d, Flatten, ConstantPad2d


class Encoder(nn.Module):

    def __init__(self, in_channels=1, latent_dim=16):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        conv_layers = []
        conv_layers.append(Conv2d(in_channels, 8, kernel_size=3, stride=2))
        conv_layers.append(nn.ELU())
        conv_layers.append(Conv2d(8, 8, kernel_size=3, stride=2))
        conv_layers.append(nn.ELU())
        conv_layers.append(Conv2d(8, 8, kernel_size=3, stride=2))
        conv_layers.append(nn.ELU())
        conv_layers.append(nn.Flatten())

        fc_layers = []
        fc_layers.append(Linear(32, latent_dim))

        self.convs = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):

    def __init__(self, latent_dim=16):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        fc_layers = []
        fc_layers.append(Linear(latent_dim, 128))

        conv_layers = []
        conv_layers.append(nn.Upsample(scale_factor=(2,2)))
        conv_layers.append(Conv2d(8, 8, kernel_size=3, stride=1, padding=1))
        conv_layers.append(nn.ELU())
        conv_layers.append(nn.Upsample(scale_factor=(2,2)))
        conv_layers.append(Conv2d(8, 8, kernel_size=3, stride=1, padding=0))
        conv_layers.append(nn.ELU())
        conv_layers.append(nn.Upsample(scale_factor=(2,2)))
        conv_layers.append(Conv2d(8, 1, kernel_size=3, stride=1, padding=1))
        conv_layers.append(nn.ELU())        

        self.fc = nn.Sequential(*fc_layers)
        self.convs = nn.Sequential(*conv_layers)
    
    def forward(self, x):
        x = self.fc(x)
        batch_size = x.shape[0]
        x = x.reshape([batch_size, 8, 4, 4])
        x = self.convs(x)
        return x
