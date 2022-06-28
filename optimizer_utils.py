import torch

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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