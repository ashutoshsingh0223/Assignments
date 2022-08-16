
##Project Structure
    .
    ├── helsing              
        ├── sum
            ├── dataset.py - Dataset class for SuMNIST subtask
        ├── sum_diff 
            ├── dataset.py - Dataset class for DiffSuMNIST subtask
        ├── abstract.py
        ├── blocks.py 
        ├── constants.py 
        ├── utils.py
        ├── model.py 
    ├── train.py - Contains all training methods
    ├── scratch.py - All scratch methods
    └── HelsingReadme.md

##Artefacts Location
    
    .
    ├── helsing
        ├── op_aware_bilinear-15-08-2022-21-56
            ├── best-classifier.pt - File containing best classifier so far. Everything required to reload the model
            ├── classifier.pt - File containing final classifier.
            ├── hyperparams.json - Configs and hyperparams
            ├── metrics.json - Epoch wise Training and Validation metrics
            ├── val_metrics.png - Grapha for validation metrics - epoch wise


##Training and testing a model
Please check `demo.ipynb`


## More stuff

### Configs used

We use this type of config for parametrizing the model.
```
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
    # For bilinear layer will just be 128
    'fc_classifier': (
        {"in_features": 128, "out_features": 256, "activation": None, 'dropout': 0.4},
        {"in_features": 256, "out_features": 19, "activation": None, 'dropout': None},
    )
}

```

#### Encoder Config details
Encoded config is an iterable of following dict
```
{'kernel': 3, 'out_channel_factor': None, 'out_channels': 64, 'batch_norm': True, 'pool': True,
 'type': 'max', 'pool_stride': 2, 'padding': 1, 'identity': True, 'activation': 'relu', 'stride': 1},
```

Most of the explainations are simple in context of CNNs. `type` refers to type of pooling, and `identity` is something similar to a residual connection.

#### Fully Connected classifier(fc_classifier) Config details
It is an iterable of following details
```
{"in_features": 128, "out_features": 256, "activation": None, 'dropout': 0.4},
```
The attributes are self-expalanatory.


### model_index
Something like a  model-zoo to facilitate use of models with similar outputs by training methods.
Found in `./helsing/model.py`. Keys from this `dict` are used as model names.
