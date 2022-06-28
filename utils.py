import numpy as np
import os
import torch
import torch.nn.functional as F

import curves
import metrics
import data


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = dir
    
    torch.save(state, filepath)

def get_params_vector(net):
    params_lst = [param.clone().detach().reshape(-1) for param in net.parameters()]
    params_tensor = torch.cat(params_lst)
    return params_tensor

def get_grad_vector(net):
    params_grad_lst = [param.grad.detach().reshape(-1) for param in net.parameters()]
    params_grad_tensor = torch.cat(params_grad_lst)
    return params_grad_tensor

def train_with_grad_log(train_loader, model, optimizer, criterion, device, regularizer=None, grad_clip=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0
    grad_norm = []
    opt_step_norm = []

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True) # async=True

        output = model(input)
        
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        
        params_before_step = get_params_vector(model)
        loss.backward()
        optimizer.step()
        params_after_step = get_params_vector(model)
        opt_step_norm.append(torch.norm(params_after_step - params_before_step).item())
        grad_norm.append(torch.norm(get_grad_vector(model)).item())

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
        'grad_norm': grad_norm,
        'opt_step_norm': opt_step_norm
    }



def train(train_loader, model, optimizer, criterion, device, regularizer=None, grad_clip=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, batch in enumerate(train_loader):
        input = batch['input'].to(device, non_blocking=True)
        gap_size = batch['gap'].to(device, non_blocking=True) if 'gap' in batch else None
        target = batch['target'].to(device, non_blocking=True) # async=True

        output = model(input)
        loss = criterion(output, target, gap_size)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, device, regularizer=None, **kwargs):
    model.eval()
    predictions_logits, targets = predictions(test_loader, model, device)
    nll = criterion(torch.from_numpy(predictions_logits), \
                    torch.from_numpy(targets))
    loss = nll.clone()
    if regularizer is not None:
        loss += regularizer(model)
    
    nll_ = -metrics.metrics_kfold(predictions_logits, targets, n_splits=2, n_runs=5,\
                                      verbose=False, temp_scale=True)["ll"]

    return {
        'nll': nll_,
        'loss': loss.item(),
        'accuracy': (np.argmax(predictions_logits, axis=1)==targets).mean() * 100.0,
    }


@torch.no_grad()
def predictions(test_loader, model, device, **kwargs):
    model.eval()
    preds = []
    targets = []
    for batch in test_loader:
        input = batch['input'].to(device) # async=True
        target = batch['target']
        output = model(input, **kwargs)
        probs = output #F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, device, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for batch in loader:
        input = batch['input']
        input = input.to(device) # async=True
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

def train_test_predictions_from_scratch(dataset, data_path, model, device, transform,
                                        batch_size = 1000, num_workers=4):
    loaders, _ = data.loaders(dataset,
                              data_path,
                              batch_size,
                              num_workers,
                              transform if 'noDA' in transform else transform + '_noDA',
                              use_test=False,
                              shuffle_train=False
                              )
    predictions_train_logits, targets_train = predictions(loaders['train'], model, device)
    predictions_test_logits, targets_test = predictions(loaders['test'], model, device)

    return {'train': (predictions_train_logits, targets_train),
            'test': (predictions_test_logits, targets_test)}
                                        