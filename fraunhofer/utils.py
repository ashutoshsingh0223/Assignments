from typing import Tuple, Any, Dict
from collections import OrderedDict

import torch.nn as nn


# Activation funcs
def get_activation_and_params(name) -> Tuple[Any, Dict[str, Any]]:
    index = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'relu6': nn.ReLU6,
        'leaky_relu': nn.LeakyReLU,
        'softmax': nn.Softmax
    }

    params_index = {
        'relu': {'inplace': True},
        'sigmoid': {},
        'relu6': {'inplace': True},
        'leaky_relu': {'negative_slope': 0.01, 'inplace': True},
        'softmax': {'dim': 1}
    }
    return index[name], params_index[name]


def disp_metrics_for_epoch(res_dict):
    for key in res_dict.keys():
        if res_dict[key]:
            value = str(round(res_dict[key][-1], 3))
            pad = 5 - len(value)
            value += pad * " "
            print("{}: {} | ".format(key, value), end="")
    print('\n')


# Results dict for classification
def get_res_dict():
    classification_res_dict = OrderedDict()
    metrics = ['loss', 'acc', 'f1', 'roc_auc']
    for m in metrics:
        classification_res_dict[f'val_{m}'] = []
        classification_res_dict[f'train_{m}'] = []

    return classification_res_dict
