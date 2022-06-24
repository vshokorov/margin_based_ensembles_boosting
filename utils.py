import numpy as np
import os
import torch
import torch.nn.functional as F

import curves
import metrics


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


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
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
    for iter, (input, gap_size, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.to(device, non_blocking=True)
        gap_size = gap_size.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True) # async=True

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



def predictions(test_loader, model, device, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.to(device) # async=True
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
    for input, _ in loader:
        input = input.to(device) # async=True
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))

def lr_schedule(lr_shed_type):
        if lr_shed_type == "standard":
            def learning_rate_schedule(base_lr, epoch, total_epochs):
                alpha = epoch / total_epochs
                if alpha <= 0.5:
                    factor = 1.0
                elif alpha <= 0.9:
                    factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
                else:
                    factor = 0.01
                return factor * base_lr
        elif lr_shed_type == "warmup":
            def learning_rate_schedule(base_lr, epoch, total_epochs):
                alpha = epoch / total_epochs
                if alpha <= 0.2:
                    factor = alpha * 5
                else:
                    factor = ((1 - alpha) / (0.8)) ** 2
                return factor * base_lr
        elif lr_shed_type == "standard_0.3":
            def learning_rate_schedule(base_lr, epoch, total_epochs):
                alpha = epoch / total_epochs
                if alpha <= 0.3:
                    factor = 1.0
                elif alpha <= 0.8:
                    factor = 1.0 - (alpha - 0.3) / 0.5 * 0.99
                else:
                    factor = 0.01
                return factor * base_lr
        elif lr_shed_type == "stair":
            def learning_rate_schedule(base_lr, epoch, total_epochs):
                if epoch < total_epochs / 2:
                    factor = 1.0
                else:
                    factor = 0.1
                return factor * base_lr
        elif lr_shed_type == "exp":
            def learning_rate_schedule(base_lr, epoch, total_epochs):
                factor = 0.9885 ** epoch
                return factor * base_lr
        elif lr_shed_type == "standard_fixed_min":
            def learning_rate_schedule(base_lr, epoch, total_epochs, min_value = 0.0005):
                alpha = epoch / total_epochs
                if alpha <= 0.5:
                    return base_lr
                elif alpha <= 0.9:
                    factor = (alpha - 0.5) / 0.4
                    return (1 - factor) * base_lr + factor * min_value
                else:
                    return min_value
        elif lr_shed_type == "none":
            def learning_rate_schedule(base_lr, *args, **kw_args):
                return base_lr
        return learning_rate_schedule

class OptimizerWithSchedule():
    def __init__(self, optimizer, lr_schedule, lr_scheduler_args) -> None:
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.base_lr = lr_scheduler_args['max_lr']
        self.epochs = lr_scheduler_args['epochs']
        self.steps_per_epoch = lr_scheduler_args['steps_per_epoch']
        self._step = 0
        
    def step(self):
        self.optimizer.step()
        
        if isinstance(self.lr_schedule, torch.optim.lr_scheduler.OneCycleLR):
            self.lr_schedule.step()
        elif self._step % self.steps_per_epoch == 0:
            self.__end_epoch()
        self._step += 1
        

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        raise NotImplementedError()
        return self.optimizer.load_state_dict(state_dict)

    def __end_epoch(self):
        lr = self.lr_schedule(self.base_lr, self._step//self.steps_per_epoch, self.epochs)
        adjust_learning_rate(self.optimizer, lr)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

def optimizer_lrscheduled(opt_kwargs, optimizer_name, lr_shed_type, lr_scheduler_args= {}):

    if 'nobiasWD' in optimizer_name:
        bias_params = [value for key, value  in opt_kwargs['params'] if key[-4:] == 'bias' and value.requires_grad]
        nbias_params = [value for key, value  in opt_kwargs['params'] if key[-4:] != 'bias' and value.requires_grad]
        opt_kwargs['params'] = [{'params': bias_params},
                                {'params': nbias_params, 'weight_decay': opt_kwargs['weight_decay']}]
        opt_kwargs.pop('weight_decay')

    if 'SGD' in optimizer_name:
        optimizer = torch.optim.SGD(**opt_kwargs)
    elif 'Adam' in optimizer_name:
        opt_kwargs.pop('momentum')
        optimizer = torch.optim.Adam(**opt_kwargs)
    else:
        raise
    
    if lr_shed_type == 'OneCycleLR':
        lr_scehd = torch.optim.lr_scheduler.OneCycleLR(optimizer, **lr_scheduler_args)
    else:
        lr_scehd = lr_schedule(lr_shed_type)

    return OptimizerWithSchedule(optimizer, lr_scehd, lr_scheduler_args)

