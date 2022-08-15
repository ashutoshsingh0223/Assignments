from typing import Tuple, Any, Dict
from collections import OrderedDict
from datetime import datetime

import torch.nn as nn

import matplotlib.pyplot as plt

from helsing.constants import BASE_DIR


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


def plot_result_metrics(res_dict, path, mode='val'):

    assert mode in ['val', 'train'], 'Mode should be in (val, train)'
    fig, axs = plt.subplots(1, 4, sharey=False, figsize=(15, 6))
    fig.tight_layout()
    axs = axs.ravel()
    i = 0
    for x in res_dict:
        if mode in x:
            axs[i].set_ylim([min(res_dict[x]), max(res_dict[x])])
            axs[i].set_xlabel('epochs')
            axs[i].set_xticks(list(range(0, 15, 2)))
            axs[i].plot(res_dict[x])
            axs[i].set_title(x)
            i += 1
    plt.savefig(path / f'{mode}_metrics.png', dpi=120)


# Results dict for classification
def get_res_dict():
    classification_res_dict = OrderedDict()
    metrics = ['loss', 'acc', 'f1', 'roc_auc']
    for m in metrics:
        classification_res_dict[f'val_{m}'] = []
        classification_res_dict[f'train_{m}'] = []

    return classification_res_dict


def create_run_dir(name: str):
    # Create run_id and a directory
    now = datetime.now()  # current date and time
    run_id = f"{name}-{now.strftime('%d-%m-%Y-%H-%M')}"
    run_path = BASE_DIR / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    return run_id, run_path