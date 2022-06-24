import argparse
import sys
import string
import time
import torch
import torch.nn.functional as F
import numpy as np

import data
import models
import utils
import random

import warnings
warnings.filterwarnings("ignore")

import logger
import wandb

MAX_FAIL_RUN_NUMBER = 2

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

def main():


    parser = argparse.ArgumentParser(description='DNN curve training')
    parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                        help='training directory (default: /tmp/curve/)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', action='store_true',
                        help='switches between validation and test set (default: validation)')
    parser.add_argument('--bootstrapping', action='store_true',
                        help='use bootstrapping while training')
    parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                        help='transform name (default: VGG)')
    parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                        help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')
    parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                        help='model name (default: None)')
    parser.add_argument('--comment', type=str, default="", metavar='T', help='comment to the experiment')
    parser.add_argument('--wandb_group', type=str, default=None, help='wandb group of expirement')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                        help='save frequency (default: 50)')
    parser.add_argument('--print_freq', type=int, default=1, metavar='N',
                        help='print frequency (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer type')
    parser.add_argument('--grad_clip', type=none_or_float, default='None',
                        help='optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', type=none_or_int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--width', type=int, default=64, metavar='N', help='width of 1 network')
    parser.add_argument('--num-nets', type=int, default=8, metavar='N', help='number of networks in ensemble')
    parser.add_argument('--num-exps', type=int, default=3, metavar='N', help='number of times for executung the whole script')
    parser.add_argument('--not-random-dir', action='store_true',
                        help='randomize dir')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='WD',
                        help='dropout rate for fully-connected layers')
    parser.add_argument('--train_temperature', type=float, default=1,
                        help='temperature in SoftMax')
    parser.add_argument('--logit_norm_type', type=str, default=None,
                        help='type of logit normalization. InstanceNorm/L2/L1/L_inf')
    parser.add_argument('--not-save-weights', action='store_true',
                        help='not save weights')
    parser.add_argument('--lr-shed', type=str, default='standard', metavar='LRSHED',
                        help='lr shedule name (default: standard)')
    parser.add_argument('--shorten_dataset', action='store_true',
                        help='same train set of size N/num_nets for each net')
    parser.add_argument('--initialization', type=str, default='standart',
                        help='initialization name (default: standard), available also: PATH')
    parser.add_argument('--noisy_data', action='store_true',
                        help='create noisy dataset, p_{idx is noise}=0.2')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                        help='wandb api key')

    args = parser.parse_args()
    wandb.login(key=args.wandb_api_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    letters = string.ascii_lowercase
    
    exp_label = "%s_%s/%s"%(args.dataset, args.model, args.comment)
    if args.num_exps > 1:
        if not args.not_random_dir:
            exp_label += "_%s/"%''.join(random.choice(letters) for i in range(5))
        else:
            exp_label += "/"
    
    np.random.seed(args.seed)
    
    attempt_number = 0
    for exp_num in range(args.num_exps):

        fmt_list = [('lr', "3.4e"), ('tr_loss', "3.3e"), ('tr_acc', '9.4f'), \
                    ('te_nll', "3.3e"), ('te_acc', '9.4f'), ('ens_acc', '9.4f'),   
                    ('ens_nll', '3.3e'), ('time', ".3f")]
        fmt = dict(fmt_list)
        log = logger.Logger(exp_label, fmt=fmt, base=args.dir)

        log.print(" ".join(sys.argv))
        log.print(args)

        torch.backends.cudnn.benchmark = True
        
        
        if args.seed is not None: 
            args.seed = np.random.randint(1000)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
        else:
            torch.manual_seed(25477)
            torch.cuda.manual_seed(25477)

        loaders, num_classes = data.loaders(
            args.dataset,
            args.data_path,
            args.batch_size,
            args.num_workers,
            args.transform,
            args.use_test,
            args.bootstrapping,
        )
        
        if args.shorten_dataset:
            loaders["train"].dataset.targets = loaders["train"].dataset.targets[:5000]
            loaders["train"].dataset.data = loaders["train"].dataset.data[:5000]

        architecture = getattr(models, args.model)()
        if "VGG" in args.model or "WideResNet" in args.model:
            architecture.kwargs["k"] = args.width
            architecture.kwargs["norm_type"] = args.logit_norm_type
            architecture.kwargs["p"] = args.dropout
                
        def criterion(x, y, gap_size=0):
            x[torch.arange(x.size(0)), y] -= gap_size
            return F.cross_entropy(x, y)

        regularizer = None

        gap_size = np.zeros((len(loaders['train'].dataset), ))
        loaders['train'].dataset.gap_size = np.zeros((len(loaders['train'].dataset), ))
        
        num_model = 0
        while num_model < args.num_nets:
            
            if args.bootstrapping:
                raise DeprecationWarning()
                
            model = architecture.base(num_classes=num_classes, **architecture.kwargs)
            model.temperature = args.train_temperature

            if args.initialization != 'standart':
                weights_load_status = model.load_state_dict(torch.load(args.initialization)['model_state'])
                log.print("Model weights:", weights_load_status)

            model = model.to(device)

            optimizer = utils.optimizer_lrscheduled(
                {'params': filter(lambda param: param.requires_grad, model.parameters()),
                 'lr': args.lr,
                 'momentum':args.momentum,
                 'weight_decay':args.wd},
                args.optimizer,
                args.lr_shed,
                {'max_lr': args.lr,
                 'epochs': args.epochs,
                 'steps_per_epoch': len(loaders['train'])}
            )

            start_epoch = 1
            if args.resume is not None:
                print('Resume training from %s' % args.resume)
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])

            wandb_group = log.full_run_name if args.wandb_group is None else args.wandb_group

            run = wandb.init(project='power_laws_deep_ensembles', 
                             entity='vetrov_disciples', 
                             group=wandb_group,
                             name=log.full_run_name + '_' + str(num_model),
                             resume=False,
                             reinit=True)
            wandb.config.update(args)
            run.save()

            has_bn = utils.check_bn(model)
            test_res = {'loss': None, 'accuracy': None, 'nll': None}
            for epoch in range(start_epoch, args.epochs + 1):
                time_ep = time.time()

                train_res = utils.train(loaders['train'], model, optimizer, criterion, device, regularizer)
                
                ens_acc = np.NaN
                ens_nll = np.NaN
                # if epoch == args.epochs:
                #     predictions_logits, targets = utils.predictions(loaders['test'], model, device)
                #     predictions = F.softmax(torch.from_numpy(predictions_logits), dim=1).numpy()
                #     predictions_sum = ensemble_size/(ensemble_size+1) \
                #                       * predictions_sum+\
                #                       predictions/(ensemble_size+1)
                #     ensemble_size += 1
                #     ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)
                #     predictions_sum_log = np.log(predictions_sum+1e-15)
                #     ens_nll = -metrics.metrics_kfold(predictions_sum_log, targets, n_splits=2, n_runs=5,\
                #                                     verbose=False, temp_scale=True)["ll"]

                if not args.not_save_weights and epoch % args.save_freq == 0:
                    utils.save_checkpoint(
                        log.path+f'/model_run{num_model}_ep{epoch}.cpt',
                        epoch,
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict()
                    )

                time_ep = time.time() - time_ep
                
                test_res = utils.test(loaders['test'], model, \
                                        criterion, device, regularizer)
                lr = optimizer.get_lr()
                values = [lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
                            test_res['accuracy'], ens_acc, ens_nll, time_ep]
                
                wandb_log_dict = {k:v for (k, _), v in zip(fmt_list, values)}
                wandb_log_dict['epoch'] = epoch
                wandb.log(wandb_log_dict)

                if epoch % args.print_freq == 0:
                    for (k, _), v in zip(fmt_list, values):
                        log.add(epoch, **{k:v})
                    log.iter_info()
                    log.save(silent=True)

            if not args.not_save_weights:
                utils.save_checkpoint(
                    log.path+'/model_run%d.cpt' % num_model,
                    args.epochs,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict()
                )

            del loaders['train'].dataset.gap_size
            predictions_logits, targets = utils.predictions(loaders['train'], model, device)
            index = np.arange(len(predictions_logits))
            target_predictions_logits = predictions_logits[index, targets]
            predictions_logits[index, targets] = predictions_logits.min() - 10
            local_gap_size = target_predictions_logits - predictions_logits.max(1)
            gap_size += local_gap_size
            loaders['train'].dataset.gap_size = gap_size.mean() - gap_size
            wandb.log({'mean_gap_size': gap_size.mean()})
            wandb.log({'local_mean_gap_size': local_gap_size.mean()})
            
            run.finish()
            if test_res['accuracy'] >= 20:
                num_model += 1
                attempt_number = 0

            else:
                attempt_number += 1
                if attempt_number == MAX_FAIL_RUN_NUMBER:
                    num_model += 1
                    attempt_number = 0

    return log.path    
        
if __name__ == "__main__":
    main()
