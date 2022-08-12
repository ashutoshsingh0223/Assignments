from typing import Optional

import torch.nn as nn

from fraunhofer.utils import get_activation_and_params


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int,
                 batch_norm: bool, identity: bool, activation: Optional[str] = None):
        super(ConvBlock, self).__init__()

        self.block = nn.ModuleList()

        self.block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, padding=padding, stride=stride))
        if batch_norm:
            self.block.append(nn.BatchNorm2d(num_features=out_channels))
        if activation:
            activation, params = get_activation_and_params(activation)
            self.block.append(activation(**params))

        self.identity = None
        if identity:
            if in_channels == out_channels:
                self.identity = nn.Sequential()
            else:
                self.identity = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(out_channels),
                )

    def forward(self, x):
        out = self.block(x)
        if self.identity is not None:
            out = out + self.identity(x)

        return out


class FullyConnectedSet(nn.Module):
    def __init__(self, fully_connected_cfg):
        super(FullyConnectedSet, self).__init__()

        self.fc_layers = nn.ModuleList()
        for index, layer_info in enumerate(fully_connected_cfg):
            activation = None
            in_features = layer_info['in_features']

            # Do not put an activation to last layer
            if index != len(fully_connected_cfg) - 1 and layer_info['activation']:
                activation_cls, activation_params = get_activation_and_params(name=layer_info['activation'])
                activation = activation_cls(**activation_params)

            self.fc_layers.append(nn.Linear(in_features=in_features,
                                            out_features=layer_info['out_features']))

            if layer_info.get('dropout'):
                self.fc_layers.append(nn.Dropout(p=layer_info['dropout']))

            if activation is not None:
                self.fc_layers.append(activation)

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return x
