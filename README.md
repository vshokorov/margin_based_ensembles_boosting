#### Main scripts:

__train.py__: A script for training (and evaluating) the ensembles of VGG and WideResNet of different network sizes and ensemble sizes, for several times (e. g. for averaging the results). The script is based on [this repository](https://github.com/timgaripov/swa).

Training VGG on CIFAR100/CIFAR10:
```(bash)
python3 train.py --dir=logs/oct --data_path=./data/ --dataset=CIFAR100 --use_test --transform=VGG --model=VGG16 --save_freq=200 --print_freq=5 --epochs=200 --wd=0.001 --lr=0.05 --dropout 0.5 --comment width64 --seed 25477 --width 64 --num-nets 8 --num-exps=5 --not-save-weights
```

Training WideResNet on CIFAR100/CIFAR10:
```(bash)
python3 train.py --dir=logs/oct --data_path=./data/ --dataset=CIFAR100 --use_test --transform=ResNet --model=WideResNet28x10 --save_freq=200 --print_freq=5 --epochs=200 --wd=0.0003 --lr=0.1 --dropout 0.0 --comment width160 --seed 25477 --width 160 --num-nets 8 --num-exps=5 --not-save-weights
```
Parameters:
* --dir: where to save logs / models
* --data_path: a dir to the data (if not exist, the data will be downloaded ito this directory)
* --dataset: CIFAR100 / CIFAR10
* --model: VGG16 / WideResNet28x10
* --width: width factor (to vary network sizes); options are listed in [consts.py](https://github.com/nadiinchi/power_laws_deep_ensembles/blob/main/hypers.py)
* --num-nets: number of networks to train
* --num-exps: number of ensembles to train
* --wd, --lr, --dropout: hyperparameters, listed in [consts.py](https://github.com/nadiinchi/power_laws_deep_ensembles/blob/main/hypers.py) for different width factors for VGG and WideResNet
* --comment: additional string to be used in the name of the folder containing the results of the run
* --epochs: number of trainign epochs (we always use 200)
* --use_test: if specified, test set is used, otherwise validation set (a part of training set) is used; needed for tuning hyperparameters
* --transform: data transformation and augmentation to use (VGG / ResNet); should be specified according to the model chosen
* --save_freq / print_freq: how frequently to save the model / log
* --not-save-weights: if specified, the weights are not saved (useful when training huge ensembles)
