# MnasNet implementation using PyTorch

Inspired by: "MnasNet: Platform-Aware Neural Architecture Search for Mobile" Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le

This repository contains implementation of MnasNet-A1 and B1 architectures from scratch using PyTorch.

In order to train the network from scratch on the dataset provided by torchvision use following command:

```
python app.py parameters.model="mnasnet-a1"
```

In order to continue training from the saved ckpts use following command:

```
python app.py parameters.model="mnasnet-a1" parameters.resume="/path_to_ckpts.pth"
```

### Model summaries:

|Model|Number of parameters|
|-----|--------------------|
|A1|3,449,568| 
|B1|3,115,122|


### Experimental settings:

For details regarding the hyperparameters please check ```defaults.yaml``` file.

* Number of epochs: 200

* Batch size: 128

* Optimizer: RMSprop

* Initial Learning rate: 1.e-3

* LR scheduler: ReduceLROnPlateau

* Dropout: 0.2

### Results:

|Model|Dataset|Test Acc|
|-----|-------|--------|
|A1|cifar10|91.09%|
|B1|cifar10|90.60%|
|A1|cifar100|70.01%|
|B1|cifar100|66.88%|
