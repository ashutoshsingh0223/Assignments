from typing import Union
import json
from pathlib import Path

import torch
import torch.nn as nn

from fraunhofer.blocks import ConvBlock, FullyConnectedSet
from fraunhofer.constants import BASE_DIR


class Encoder(nn.Module):
    def __init__(self, config, in_channels=3):
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


class Classifier(nn.Module):
    def __init__(self, config, num_classes=3):
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

            num_classes: number of classes for the classifier. Default 3
        """
        super(Classifier, self).__init__()

        self.config = config
        self.num_classes = num_classes
        self.encoder = Encoder(config=self.config['encoder'], in_channels=self.config['in_channels'])

        self.bottleneck = nn.Conv2d(in_channels=self.encoder.out_channels, out_channels=1, kernel_size=1)
        self.flatten = nn.Flatten()

        self.fc_classifier = FullyConnectedSet(fully_connected_cfg=self.config['fc_classifier'])

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.flatten(x)
        x = self.fc_classifier(x)
        return x

    def save(self, run_id, best=False):
        """

        Args:
            run_id: id of train run for which saving was donw
            best: if this save is best model of the run

        Returns:
            None
        """
        if best:
            path = BASE_DIR / run_id / "best-classifier.pt"
        else:
            path = BASE_DIR / run_id / "classifier.pt"
        state = {
            'state_dict': self.state_dict(),
            'params': {'config': self.config, 'num_classes': self.num_classes}
        }

        torch.save(state, path)
        print(f'Model saved at: {path}')

    @classmethod
    def load(cls, path: Union[Path, str]):
        """

        Args:
            path: Path of model file. Check `.pt` files in run-id dir

        Returns:

        """
        data = json.loads(open(path, 'r').read())
        model = cls(**data['params'])

        state_dict = torch.load(data['state_dict'])
        model.load_state_dict(state_dict)

        return model
