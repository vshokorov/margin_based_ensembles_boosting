import utils
import numpy as np
import wandb
from scipy import stats
import re

def get_margin(logits, target):
    logits = np.array(logits).copy()
    index = np.arange(len(logits))
    target_logits = logits[index, target]
    logits[index, target] = logits.min() - 10
    return target_logits - logits.max(1)

# def get_margin(logits, target):
#     logits = np.array(logits).copy()
#     index = np.arange(len(logits))
#     target_logits = logits[index, target]
#     logits[index, target] = 0
#     num_classes = logits.shape[1]
#     mean_other = logits.sum(1) / (num_classes - 1)
#     return target_logits - mean_other

BASE_MARGIN_STD = 3.5

class Ens_gap_editor():
    def __init__(self, editor_type):
        self.normalize_margin = False

        if editor_type == 'None':
            self.initial_gap = lambda size: np.zeros(size)
            self.__get_next_gap = getattr(self, 'const_gap')
            return 
        elif editor_type.replace('.','',1).replace('-','',1).isdigit():
            self.initial_gap = lambda size: np.ones(size) * float(editor_type)
            self.__get_next_gap = getattr(self, 'const_gap')
            return 
        elif editor_type.startswith('randnorm_'):
            result = re.match(r'randnorm_[0-9.]+', editor_type)
            gap = float(result.group(0).replace('randnorm_', ''))
            self.initial_gap = lambda size: np.random.normal(loc=0.0, scale=gap, size=size)
            self.__get_next_gap = getattr(self, 'const_gap')
            return 
        elif editor_type.startswith('randexp_'):
            result = re.match(r'randexp_[0-9.]+', editor_type)
            gap = float(result.group(0).replace('randexp_', ''))
            self.initial_gap = lambda size: np.random.exponential(scale=gap, size=size) - gap
            self.__get_next_gap = getattr(self, 'const_gap')
            return 
        elif editor_type.startswith('randmexp_'):
            result = re.match(r'randmexp_[0-9.]+', editor_type)
            gap = float(result.group(0).replace('randmexp_', ''))
            self.initial_gap = lambda size: gap - np.random.exponential(scale=gap, size=size)
            self.__get_next_gap = getattr(self, 'const_gap')
            return 
        elif editor_type.startswith('base_'):
            result = re.match(r'base_[0-9.]+', editor_type)
            gap = float(result.group(0).replace('base_', ''))
            self.initial_gap = lambda size: np.ones(size) * gap

            if result.group(0) == editor_type:
                self.__get_next_gap = getattr(self, 'const_gap')
                return 
            
            editor_type = editor_type.replace(result.group(0) + '_', '')
        else:
            self.initial_gap = lambda size: np.zeros(size)

        if editor_type.startswith('norm_'):
            self.__get_next_gap = getattr(self, editor_type.replace('norm_', ''))
            self.normalize_margin = True
        elif 'simple_gap_marginscale_' in editor_type:
            self._scale = float(editor_type.replace('simple_gap_marginscale_', ''))
            self.__get_next_gap = getattr(self, 'simple_gap_marginscale')
        else:
            self.__get_next_gap = getattr(self, editor_type)



    def __normalize_margin(self, m, num_model):
        if num_model == 0:
            self.__base_margin_std = m.std()
            return m
        return m * self.__base_margin_std / m.std()

    def get_next_gap(self, last_gap, logits, target, num_model):
        margin = get_margin(logits, target)
        if self.normalize_margin: margin = self.__normalize_margin(margin, num_model)

        val, logval = self.__get_next_gap(margin, last_gap, num_model, logits, target)
        if logval is not None:
            wandb.log({'mean_gap_size': logval})
        
        # val /= logits[np.arange(logits.shape[0]), target] # scale by true logit
        
        return val

    def const_gap(self, margin, last_gap, *args):
        mean_margin = margin.mean()
        return self.initial_gap(len(margin)), mean_margin

    def simple_gap(self, margin, *args):
        mean_margin = margin.mean()
        return self.initial_gap(len(margin)) + mean_margin - margin, mean_margin

    def save_mean_reverse_gap(self, margin, *args):
        mean_margin = margin.mean()
        return self.initial_gap(len(margin)) + 2 * mean_margin - margin, mean_margin
    
    def max2min_min2max(self, margin, *args):
        mean_margin = margin.mean()
        return self.initial_gap(len(margin)) + margin.min() - margin + margin.max(), mean_margin

    def gap_is_meanmargin(self, margin, last_gap, num_model, *args):
        # One model in ensemble means that num_model = 0
        margin = (last_gap * num_model  + margin) / (num_model + 1)
        mean_margin = margin.mean()
        return self.initial_gap(len(margin)) + margin, mean_margin

    def simple_gap_center_by_mode(self, margin, *args):
        mean_margin = margin.mean()
        mode_margin = stats.mode(margin).mode
        return self.initial_gap(len(margin)) + mode_margin - margin, mean_margin
    
    def simple_gap_marginscale(self, margin, *args):
        mean_margin = margin.mean()
        # mode_margin = stats.mode(margin).mode
        margin = mean_margin - margin
        margin *= self._scale

        return self.initial_gap(len(margin)) + margin, mean_margin
    
    def reverse_gap(self, margin, *args):
        mean_margin = margin.mean()
        return self.initial_gap(len(margin)) + margin - mean_margin, mean_margin

    def cummulative_gap(self, margin, last_gap, *args):
        margin += last_gap
        mean_margin = margin.mean()
        return self.initial_gap(len(margin)) + mean_margin - margin, mean_margin

    def mean_cummulative_gap(self, margin, last_gap, num_model, *args):
        # One model in ensemble means that num_model = 0
        margin = (last_gap * num_model  + margin) / (num_model + 1)
        mean_margin = margin.mean()
        return self.initial_gap(len(margin)) + mean_margin - margin, mean_margin

    def simple_gap_all_logits(self, margin, last_gap, num_model, logits, target):
        mean_margin = margin.mean()
        out = logits - logits[np.arange(logits.shape[0]), target].reshape(-1, 1)
        out[np.arange(out.shape[0]), target] -= mean_margin

        return self.initial_gap(len(margin))[:, None] + out.mean(1, keepdims=True) - out, mean_margin


# del loaders['train'].dataset.gap_size
# predictions_logits, targets = utils.predictions(loaders['train'], model, device)
# loaders['train'].dataset.gap_size = mean_gap_size - gap_size