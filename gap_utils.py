import utils
import numpy as np
import wandb

def get_margin(logits, target):
    logits = np.array(logits).copy()
    index = np.arange(len(logits))
    target_logits = logits[index, target]
    logits[index, target] = logits.min() - 10
    return target_logits - logits.max(1)



class Ens_gap_editor():
    def __init__(self, editor_type):
        self.__get_next_gap = getattr(self, editor_type)

    def get_next_gap(self, last_gap, logits, target, num_model):
        margin = get_margin(logits, target)
        val, logval = self.__get_next_gap(margin, last_gap, num_model)
        if logval is not None:
            wandb.log({'mean_gap_size': logval})
        return val

    def save_gap(self, margin, last_gap, *args):
        return last_gap, None

    def simple_gap(self, margin, *args):
        mean_margin = margin.mean()
        return mean_margin - margin, mean_margin

    def cummulative_gap(self, margin, last_gap, *args):
        margin += last_gap
        mean_margin = margin.mean()
        return mean_margin - margin, mean_margin

    def mean_cummulative_gap(self, margin, last_gap, num_model):
        margin = (last_gap * (num_model - 1)  + margin) / num_model
        margin += last_gap
        mean_margin = margin.mean()
        return mean_margin - margin, mean_margin

    def zeropos_no_mean(self, margin, last_gap, num_model):
        margin = np.clip(margin, None, 0)
        mean_margin = margin.mean()
        return - margin, mean_margin
    
    def zeropos(self, margin, last_gap, num_model):
        margin = np.clip(margin, None, 0)
        mean_margin = margin.mean()
        return mean_margin - margin, mean_margin
    
    def zeroneg_no_mean(self, margin, last_gap, num_model):
        margin = np.clip(margin, 0, None)
        mean_margin = margin.mean()
        return - margin, mean_margin
    
    def zeroneg(self, margin, last_gap, num_model):
        margin = np.clip(margin, 0, None)
        mean_margin = margin.mean()
        return mean_margin  - margin, mean_margin

# del loaders['train'].dataset.gap_size
# predictions_logits, targets = utils.predictions(loaders['train'], model, device)
# loaders['train'].dataset.gap_size = mean_gap_size - gap_size