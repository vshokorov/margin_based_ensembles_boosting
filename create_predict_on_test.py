import torch
import numpy as np
import models
import data
import utils
from matplotlib import pyplot as plt


"--dir=logs",
data_path="./data/"
dataset="CIFAR10"
transform="VGG"
model="VGG16"
batch_size=128
num_workers=4
use_test=False
"--save_freq=200",
"--print_freq=5",
"--epochs=200",
"--wd=0.001",
"--lr=0.05",
dropout=0.5
"--comment=width64",
"--seed=25477",
width=64
"--num-nets=8",
"--num-exps=5"

loaders, num_classes = data.loaders(dataset,
                                    data_path,
                                    batch_size,
                                    num_workers,
                                    transform,
                                    use_test
                                    )

architecture = getattr(models, model)()
architecture.kwargs["k"] = width
if "VGG" in model or "WideResNet" in model:
    architecture.kwargs["p"] = dropout
model = architecture.base(num_classes=num_classes, **architecture.kwargs)


def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda() # async=True
        output = model(input, **kwargs)
        probs = output #F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)

work_dirs = ['/home/tyuzhakov/power_laws_deep_ensembles/logs/oct/train.py-CIFAR10_VGG16/width16_bootstrapped_DS_64_upd-12-11-22:40:46/',
             '/home/tyuzhakov/power_laws_deep_ensembles/logs/oct/train.py-CIFAR10_VGG16/width16_all_DS_64_upd-12-12-13:59:33/', 
             '/home/tyuzhakov/power_laws_deep_ensembles/logs/oct/train.py-CIFAR10_VGG16/width16_bootstrapped_noisy_DS_upd-12-10-15:30:14/',
             '/home/tyuzhakov/power_laws_deep_ensembles/logs/oct/train.py-CIFAR10_VGG16/width16_all_noisy_DS_upd-12-10-15:14:18/']

for wd in work_dirs:
    try:
        for num_model in range(100):
            saved_data = torch.load(wd+'model_run' + str(num_model) + '.cpt', map_location='cuda')
            model.load_state_dict(saved_data['model_state'])

            predictions_logits, targets = predictions(loaders['test'], model)
            np.save(wd + 'predictions_run' + str(num_model), predictions_logits)
    except FileNotFoundError:
        pass