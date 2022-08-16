
import torch
import torch.nn as nn

from helsing.blocks import ConvBlock, FullyConnectedSet
from helsing.abstract import Model


class Encoder(nn.Module):
    def __init__(self, config, in_channels=1):
        """
        Convolutional Encoder. Built for images.
        Args:
            config: Config of encoder. Contains conv layers and everything around them
            Example: (
                       {'kernel': 3, 'out_channel_factor': None, 'out_channels': 64, 'batch_norm': True, 'pool': True,
                       'type': 'max', 'pool_stride': 2, 'padding': 1, 'identity': True, 'activation': 'relu', 'stride': 1},

                       {'kernel': 3, 'out_channel_factor': 2, 'out_channels': None, 'batch_norm': True, 'pool': True,
                       'type': 'max', 'pool_stride': 2, 'padding': 1, 'identity': True, 'activation': 'relu', 'stride': 1},
                       .......
                      )

            in_channels: Number of channels.
        """
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()

        self.cfg = config
        self.in_channels = in_channels
        self.out_channels = None

        for layer_info in self.cfg:
            if isinstance(layer_info.get('out_channels'), int):
                out_channels = layer_info.get('out_channels')
            else:
                out_channels = int(in_channels * layer_info['out_channel_factor'])

            self.layers.append(ConvBlock(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=layer_info['kernel'], padding=layer_info['padding'],
                                         batch_norm=layer_info.get('batch_norm', False),
                                         identity=layer_info.get('identity', False),
                                         activation=layer_info.get('activation'), stride=layer_info.get('stride')))

            if layer_info.get('pool'):
                if layer_info['type'] == 'max':
                    self.layers.append(nn.MaxPool2d(kernel_size=layer_info['pool_stride']))
                elif layer_info['type'] == 'avg':
                    self.layers.append(nn.AvgPool2d(kernel_size=layer_info['pool_stride']))
                else:
                    raise ValueError('Only supported pooling operations are max and avg')

            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DirectAdditionModel(nn.Module, Model):

    def __init__(self, config):
        """
        Image classifier
        Args:
            config: model config, contains encoder config, number of input channels etc.
            Example:
                    {
                        'in_channels': 3,
                        'encoder': (
                                 {'kernel': 3, 'out_channel_factor': None, 'out_channels': 64, 'batch_norm': True, 'pool': True,
                                 'type': 'max', 'pool_stride': 2, 'padding': 1, 'identity': True, 'activation': 'relu', 'stride': 1},
                                ,.....),
                        'fc_classifier': (
                            {"in_features": 128, "out_features": 128, "activation": 'relu6', 'dropout': 0.3},
                            ...)
        }

        """
        super(DirectAdditionModel, self).__init__()

        self.config = config
        self.encoder = Encoder(config=self.config['encoder'], in_channels=self.config['in_channels'])

        # self.bottleneck = nn.Conv2d(in_channels=self.encoder.out_channels, out_channels=1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.sigmoid = nn.Sigmoid()
        self.fc_classifier = FullyConnectedSet(fully_connected_cfg=self.config['fc_classifier'])

    def forward(self, x1, x2, regress=False):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1 = self.flatten(x1)
        x2 = self.flatten(x2)

        x = x1 + x2
        x = self.fc_classifier(x)

        if regress:
            x = self.sigmoid(x)
        return x


class BilinearModel(nn.Module, Model):

    def __init__(self, config):
        """
        Image classifier
        Args:
            config: model config, contains encoder config, number of input channels etc.
            Example:
                    {
                        'in_channels': 3,
                        'encoder': (
                                 {'kernel': 3, 'out_channel_factor': None, 'out_channels': 64, 'batch_norm': True,'pool': True,
                                 'type': 'max', 'pool_stride': 2, 'padding': 1, 'identity': True, 'activation':
                                 'relu', 'stride': 1},
                                ,.....),
                        'fc_classifier': (
                            {"in_features": 128, "out_features": 128, "activation": 'relu6', 'dropout': 0.3},
                            ...)
        }

        """
        super(BilinearModel, self).__init__()

        self.config = config
        self.encoder = Encoder(config=self.config['encoder'], in_channels=self.config['in_channels'])

        # self.bottleneck = nn.Conv2d(in_channels=self.encoder.out_channels, out_channels=1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.bilinear = nn.Bilinear(in1_features=self.config['fc_classifier'][0]['in_features'],
                                    in2_features=self.config['fc_classifier'][0]['in_features'],
                                    out_features=self.config['fc_classifier'][0]['in_features'])

        self.sigmoid = nn.Sigmoid()

        self.fc_classifier = FullyConnectedSet(fully_connected_cfg=self.config['fc_classifier'])

    def forward(self, x1, x2, regress=False):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1 = self.flatten(x1)
        x2 = self.flatten(x2)

        x = self.bilinear(x1, x2)
        x = self.fc_classifier(x)

        if regress:
            x = self.sigmoid(x)
        return x


class OpAwareModel(nn.Module, Model):
    def __init__(self, config):
        super(OpAwareModel, self).__init__()

        self.config = config
        self.encoder = Encoder(config=self.config['encoder'], in_channels=self.config['in_channels'])

        # self.bottleneck = nn.Conv2d(in_channels=self.encoder.out_channels, out_channels=1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # For mapping sign of the second input
        self.bilinear = nn.Bilinear(in1_features=self.config['fc_classifier'][0]['in_features'],
                                    in2_features=2,
                                    out_features=self.config['fc_classifier'][0]['in_features'])

        self.sigmoid = nn.Sigmoid()
        self.fc_classifier = FullyConnectedSet(fully_connected_cfg=self.config['fc_classifier'])

    def forward(self, x1, x2, op, regress=False):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1 = self.flatten(x1)
        x2 = self.flatten(x2)

        # Perform sign insertion
        x2 = self.bilinear(x2, op)

        # Add
        x = x1 + x2
        x = self.fc_classifier(x)

        if regress:
            x = self.sigmoid(x)
        return x


class OpAwareBilinear(nn.Module, Model):
    def __init__(self, config):
        super(OpAwareBilinear, self).__init__()

        self.config = config
        self.encoder = Encoder(config=self.config['encoder'], in_channels=self.config['in_channels'])

        # self.bottleneck = nn.Conv2d(in_channels=self.encoder.out_channels, out_channels=1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # For mapping sign of the second input
        self.sign = nn.Bilinear(in1_features=self.config['fc_classifier'][0]['in_features'],
                                in2_features=2,
                                out_features=self.config['fc_classifier'][0]['in_features'])

        self.bilinear = nn.Bilinear(in1_features=self.config['fc_classifier'][0]['in_features'],
                                    in2_features=self.config['fc_classifier'][0]['in_features'],
                                    out_features=self.config['fc_classifier'][0]['in_features'])

        self.sigmoid = nn.Sigmoid()
        self.fc_classifier = FullyConnectedSet(fully_connected_cfg=self.config['fc_classifier'])

    def forward(self, x1, x2, op, regress=False):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x1 = self.pool(x1)
        x2 = self.pool(x2)

        x1 = self.flatten(x1)
        x2 = self.flatten(x2)

        # Perform sign insertion
        x2 = self.sign(x2, op)

        x = self.bilinear(x1, x2)
        x = self.fc_classifier(x)

        if regress:
            x = self.sigmoid(x)
        return x


model_index = {

    'addition': DirectAdditionModel,
    'bilinear': BilinearModel,
    'subtraction': OpAwareModel,
    'op_aware_bilinear': OpAwareBilinear
}