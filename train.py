from datetime import datetime
import json

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, Softmax
import math

from torchmetrics import AUROC, Accuracy, F1Score

from helsing.model import Classifier, BilinearClassifier
from helsing.sum.dataset import PairSample
from helsing.utils import get_res_dict, disp_metrics_for_epoch, plot_result_metrics
from helsing.constants import BASE_DIR


accuracy = Accuracy(top_k=1, num_classes=3, average='macro')
auc_roc = AUROC(num_classes=3, average='macro')
f1_score = F1Score(top_k=1, num_classes=3, average='macro')

config = {
    'in_channels': 1,
    'encoder': ({'kernel': 3, 'out_channel_factor': None, 'out_channels': 64, 'batch_norm': True, 'pool': True,
                 'type': 'max', 'pool_stride': 2, 'padding': 1, 'identity': True, 'activation': 'relu', 'stride': 1},
                # 14x14x64
                {'kernel': 3, 'out_channel_factor': 2, 'out_channels': None, 'batch_norm': True, 'pool': True,
                 'type': 'max', 'pool_stride': 2, 'padding': 1, 'identity': True, 'activation': 'relu', 'stride': 1},
                # 7x7x128
                {'kernel': 3, 'out_channel_factor': 1, 'out_channels': None, 'batch_norm': True, 'pool': False,
                 'type': 'max', 'pool_stride': 2, 'padding': 1, 'identity': True, 'activation': 'relu', 'stride': 1},
                # 7x7x128
                ),
    # Concatenation yields 128 and 128 = 256
    # For bilinear layer will just be 128
    'fc_classifier': (
        {"in_features": 256, "out_features": 256, "activation": None, 'dropout': 0.4},
        {"in_features": 256, "out_features": 19, "activation": None, 'dropout': None},
    )
}


def train_classifier(train_batch_size=64, test_batch_size=64, learning_rate=0.0001, epochs=15, download=False):
    # Create run_id and a directory
    now = datetime.now()  # current date and time
    run_id = f"run-{now.strftime('%d-%m-%Y-%H-%M')}"
    run_path = BASE_DIR / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    # Dump all hyperparams and configs
    with open(run_path / 'hyperparams.json', 'w') as f:
        hyperparams = {
            'config': config, 'train_batch_size': train_batch_size, 'learning_rate': learning_rate, 'epochs': epochs
        }
        f.write(json.dumps(hyperparams))

    # Create datasets and dataloaders
    train_set = PairSample(root=BASE_DIR, train=True, download=download)
    val_set = PairSample(root=BASE_DIR, train=False, download=download)

    train_loader = DataLoader(dataset=train_set, shuffle=True, pin_memory=True, num_workers=2,
                              batch_size=train_batch_size)
    val_loader = DataLoader(dataset=val_set, shuffle=False, batch_size=test_batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define model, optimizer and criterion
    model = Classifier(config=config)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss(reduction='mean')

    softmax = Softmax(dim=1)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Get result dict
    res_dict = get_res_dict()

    best_val_loss = math.inf
    for epoch in range(epochs):
        # setup some more variables to save train stats
        run_loss_tr = 0.0
        run_loss_val = 0.0

        model.train()
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out = model(data)
                loss = criterion(out, labels)

                loss.backward()
                optimizer.step()

            #  For metrics and loss
            run_loss_tr += loss.item()
            # Need to softmax `out` because CrossEntropy in torch requires non-normalized scores
            detached_preds = softmax(out.detach().clone()).cpu()
            labels = labels.cpu()
            # Update batch metrics
            acc = accuracy(detached_preds, labels)
            f1 = f1_score(detached_preds, labels)
            roc = auc_roc(detached_preds, labels)

        # Finaly compute metrics for an epoch
        res_dict['train_acc'].append(accuracy.compute().item())
        res_dict['train_f1'].append(f1_score.compute().item())
        res_dict['train_roc_auc'].append(auc_roc.compute().item())
        res_dict['train_loss'].append(run_loss_tr / len(train_loader))

        # Reset torchmetrics instances to reuse during validation
        accuracy.reset()
        f1_score.reset()
        auc_roc.reset()

        model.eval()
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                out = model(data)
                loss = criterion(out, labels)

            detached_preds = softmax(out.detach().clone()).cpu()
            labels = labels.cpu()
            acc = accuracy(detached_preds, labels)
            f1 = f1_score(detached_preds, labels)
            roc = auc_roc(detached_preds, labels)
            run_loss_val += loss.item()

        res_dict['val_acc'].append(accuracy.compute().item())
        res_dict['val_f1'].append(f1_score.compute().item())
        res_dict['val_roc_auc'].append(auc_roc.compute().item())
        res_dict['val_loss'].append(run_loss_val / len(val_loader))

        accuracy.reset()
        f1_score.reset()
        auc_roc.reset()

        # save best model so far
        if res_dict['val_loss'][-1] <= best_val_loss:
            model.save(run_id=run_id, best=True)
            best_val_loss = res_dict['val_loss'][-1]

        # display metrics for epoch
        disp_metrics_for_epoch(res_dict)

    # Dump metrics
    with open(run_path / 'metrics.json', 'w') as f:
        f.write(json.dumps(res_dict))

    # Save final model
    model.save(run_id=run_id)

    # Save val metrics charts
    plot_result_metrics(res_dict, path=run_path)


def train_bilinear_classifier(train_batch_size=64, test_batch_size=64, learning_rate=0.0001, epochs=15, download=False):

    global config

    config['fc_classifier'][0]['in_features'] = 128
    # Create run_id and a directory
    now = datetime.now()  # current date and time
    run_id = f"bilinear-run-{now.strftime('%d-%m-%Y-%H-%M')}"
    run_path = BASE_DIR / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    # Dump all hyperparams and configs
    with open(run_path / 'hyperparams.json', 'w') as f:
        hyperparams = {
            'config': config, 'train_batch_size': train_batch_size, 'learning_rate': learning_rate, 'epochs': epochs
        }
        f.write(json.dumps(hyperparams))

    # Create datasets and dataloaders
    train_set = PairSample(root=BASE_DIR, train=True, download=download)
    val_set = PairSample(root=BASE_DIR, train=False, download=download)

    train_loader = DataLoader(dataset=train_set, shuffle=True, pin_memory=True, num_workers=2,
                              batch_size=train_batch_size)
    val_loader = DataLoader(dataset=val_set, shuffle=False, batch_size=test_batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define model, optimizer and criterion
    model = BilinearClassifier(config=config)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss(reduction='mean')

    softmax = Softmax(dim=1)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Get result dict
    res_dict = get_res_dict()

    best_val_loss = math.inf
    for epoch in range(epochs):
        # setup some more variables to save train stats
        run_loss_tr = 0.0
        run_loss_val = 0.0

        model.train()
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out = model(data)
                loss = criterion(out, labels)

                loss.backward()
                optimizer.step()

            #  For metrics and loss
            run_loss_tr += loss.item()
            # Need to softmax `out` because CrossEntropy in torch requires non-normalized scores
            detached_preds = softmax(out.detach().clone()).cpu()
            labels = labels.cpu()
            # Update batch metrics
            acc = accuracy(detached_preds, labels)
            f1 = f1_score(detached_preds, labels)
            roc = auc_roc(detached_preds, labels)

        # Finaly compute metrics for an epoch
        res_dict['train_acc'].append(accuracy.compute().item())
        res_dict['train_f1'].append(f1_score.compute().item())
        res_dict['train_roc_auc'].append(auc_roc.compute().item())
        res_dict['train_loss'].append(run_loss_tr / len(train_loader))

        # Reset torchmetrics instances to reuse during validation
        accuracy.reset()
        f1_score.reset()
        auc_roc.reset()

        model.eval()
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                out = model(data)
                loss = criterion(out, labels)

            detached_preds = softmax(out.detach().clone()).cpu()
            labels = labels.cpu()
            acc = accuracy(detached_preds, labels)
            f1 = f1_score(detached_preds, labels)
            roc = auc_roc(detached_preds, labels)
            run_loss_val += loss.item()

        res_dict['val_acc'].append(accuracy.compute().item())
        res_dict['val_f1'].append(f1_score.compute().item())
        res_dict['val_roc_auc'].append(auc_roc.compute().item())
        res_dict['val_loss'].append(run_loss_val / len(val_loader))

        accuracy.reset()
        f1_score.reset()
        auc_roc.reset()

        # save best model so far
        if res_dict['val_loss'][-1] <= best_val_loss:
            model.save(run_id=run_id, best=True)
            best_val_loss = res_dict['val_loss'][-1]

        # display metrics for epoch
        disp_metrics_for_epoch(res_dict)

    # Dump metrics
    with open(run_path / 'metrics.json', 'w') as f:
        f.write(json.dumps(res_dict))

    # Save final model
    model.save(run_id=run_id)

    # Save val metrics charts
    plot_result_metrics(res_dict, path=run_path)


if __name__ == '__main__':
    main(train_batch_size=64, test_batch_size=8, learning_rate=0.0001)
