import torch
import numpy as np
import models
import data
import utils
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
from eval_eigs import eval_eigs, eval_trace
import re
import pandas as pd

import wandb
from tqdm import tqdm
from io import StringIO 
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

@torch.no_grad()
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

api = wandb.Api(timeout=19)
entity, project = "vetrov_disciples", "power_laws_deep_ensembles"  # set to your entity and project 
runs = api.runs(entity + "/" + project) 

run_names = []
run_configs = {}
for r in runs:
    if len(r.config) > 0 and r.state == 'finished':
        run_names.append(r.name)
        run_configs[r.name] = r.config
        
work_dirs = list(set(map(lambda x: 'logs/oct/' + x[:x.find('_', -4)] + '/', run_names)))

for work_dir in tqdm(work_dirs):
    if not os.path.exists(work_dir):
        print(f'[ERROR] No such directory {work_dir}')
        continue
    
    for num_model in [int(i.replace('model_run', '', ).replace('.cpt', '')) 
                          for i in filter(lambda x: re.match('^model_run\d+.cpt$', x) is not None, os.listdir(work_dir))]:
        
        try:
            if (os.path.exists(work_dir + 'predictions_test_run%d.npy' % num_model) and
                os.path.exists(work_dir + 'predictions_train_run%d.npy' % num_model)):
                continue

            if not os.path.exists(work_dir + f'model_run{num_model}.cpt'):
                continue
                
            c = run_configs[work_dir.replace('logs/oct/', '')[:-1] + '_' + str(num_model)]
            with Capturing():
                loaders, num_classes = data.loaders(c['dataset'],
                                        c['data_path'],
                                        c['batch_size'],
                                        c['num_workers'],
                                        c['transform'] if 'noDA' in c['transform'] else c['transform'] + '_noDA',
                                        use_test=False,
                                        shuffle_train=False
                                        )
            
            architecture = getattr(models, c['model'])()
            if 'VGG' in c['model']:
                architecture.kwargs["k"] = c['width']
            # architecture.kwargs["use_InstanceNorm"] = use_InstanceNorm
            if "VGG" in c['model'] or "WideResNet" in c['model']:
                architecture.kwargs["p"] = c['dropout']
            model = architecture.base(num_classes=num_classes, **architecture.kwargs).cuda()
            _ = model.eval()

            saved_data = torch.load(work_dir+f'model_run{num_model}.cpt', map_location='cpu')
            model.load_state_dict(saved_data['model_state'])
            
            if not os.path.exists(work_dir + 'predictions_test_run%d.npy' % num_model):
                predictions_logits, targets = predictions(loaders['test'], model)
                np.save(work_dir + 'predictions_test_run%d' % num_model, predictions_logits)
            
            if not os.path.exists(work_dir + 'predictions_train_run%d.npy' % num_model):
                predictions_logits, targets = predictions(loaders['train'], model)
                np.save(work_dir + 'predictions_train_run%d' % num_model, predictions_logits)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except BaseException as ex:
            print(ex, f'in {work_dir}, {num_model}')
            # print(f'Something wrong with {work_dir}, {num_model}')
