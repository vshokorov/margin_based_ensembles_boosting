import argparse
from os import device_encoding
import sys
import string
import time
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

import data
import gap_utils
import models
import utils
import optimizer_utils
import random

import warnings
warnings.filterwarnings("ignore")

import logger
import wandb

import arcface_verification
import pickle
from datetime import datetime

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
    parser.add_argument('--wandb_group', type=str, default='', help='wandb group of expirement')
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
    parser.add_argument('--grad_clip', type=utils.none_or_float, default='None',
                        help='optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', type=utils.none_or_int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--width', type=int, default=64, metavar='N', help='width of 1 network')
    parser.add_argument('--num-nets', type=int, default=8, metavar='N', help='number of networks in ensemble')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='WD',
                        help='dropout rate for fully-connected layers')
    parser.add_argument('--train_temperature', type=float, default=1,
                        help='temperature in SoftMax')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Dimension of the model embeddings')
    parser.add_argument('--not-save-weights', action='store_true',
                        help='not save run')
    parser.add_argument('--lr-shed', type=str, default='standard', metavar='LRSHED',
                        help='lr shedule name (default: standard)')
    parser.add_argument('--shorten_dataset', action='store_true',
                        help='same train set of size N/num_nets for each net')
    parser.add_argument('--initialization', type=str, default='standart',
                        help='initialization name (default: standard), available also: PATH')
    parser.add_argument('--noisy_data', action='store_true',
                        help='create noisy dataset, p_{idx is noise}=0.2')
    parser.add_argument('--gap_size', type=str, default='None',
                        help='additional gap in logits')
    parser.add_argument('--aug_predictions', action='store_true',
                        help='use augmentation to calculate next gap')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='save embeddings instead of logits')
    parser.add_argument('--attempts_number', type=int, default=1,
                        help='set maximum number of attempts to get acceptable run. Must be 1 for gap_ensembles')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                        help='wandb api key')

    args = parser.parse_args()
    wandb.login(key=args.wandb_api_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_label = "%s_%s/%s"%(args.dataset, args.model, args.comment)
    
    attempt_number = 0

    fmt_list = [('lr', "3.4e"), ('tr_loss', "3.3e"), ('tr_acc', '9.4f'),
                # ('te_nll', "3.3e"), 
                ('te_acc', '9.4f'), ('time', ".3f")]
    fmt = dict(fmt_list)
    log = logger.Logger(exp_label, fmt=fmt, base=args.dir)

    log.print(" ".join(sys.argv))
    log.print(args)

    torch.backends.cudnn.benchmark = True

    if args.seed is None: 
        args.seed = 25477
    np.random.seed(args.seed)
    random.seed(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    def get_loaders(args):
        loaders, num_classes = data.loaders(
            dataset = args.dataset, 
            path = args.data_path, 
            batch_size = args.batch_size, 
            num_workers = args.num_workers, 
            transform_name = args.transform, 
            use_test = args.use_test,
            use_bootstrapping = args.bootstrapping,
            noisy_data=args.noisy_data,
        )
        if args.shorten_dataset:
            loaders["train"].dataset.targets = loaders["train"].dataset.targets[:5000]
            loaders["train"].dataset.data = loaders["train"].dataset.data[:5000]
        
        gap_editor = gap_utils.Ens_gap_editor(args.gap_size)
        loaders['train'].dataset.gap_size = gap_editor.initial_gap(len(loaders['train'].dataset))

        return loaders, num_classes, gap_editor
    
    loaders, num_classes, gap_editor = get_loaders(args)

    architecture = getattr(models, args.model)()
    if "VGG" in args.model or "WideResNet" in args.model:
        architecture.kwargs["k"] = args.width
        architecture.kwargs["p"] = args.dropout
    architecture.kwargs["emb_dim"] = args.emb_dim
    architecture.kwargs["img_size"] = loaders['train'].dataset.img_size
    
    def criterion(x, y, gap_size=None):
        scale = 1 / args.train_temperature
        x_out = x * 1.
        if not gap_size is None:
            if x_out.shape == gap_size.shape:
                x_out -= gap_size
            else:
                x_out[torch.arange(x_out.size(0)), y] -= gap_size
        
        x_out *= scale
        return F.cross_entropy(x_out, y)

    regularizer = None

    num_model = 0
    while num_model < args.num_nets:
        
        if args.bootstrapping:
            loaders, num_classes, gap_editor = get_loaders(args)
            
        model = architecture.base(num_classes=num_classes, **architecture.kwargs)

        if args.initialization != 'standart':
            weights_load_status = model.load_state_dict(torch.load(args.initialization)['model_state'])
            log.print("Model weights:", weights_load_status)

        model = model.to(device)

        optimizer = optimizer_utils.optimizer_lrscheduled(
            opt_kwargs = {'params': filter(lambda param: param.requires_grad, model.parameters()),
                          'lr': args.lr,
                          'momentum':args.momentum,
                          'weight_decay':args.wd},
            optimizer_name = args.optimizer,
            lr_shed_type = args.lr_shed,
            lr_scheduler_args = {'max_lr': args.lr,
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

        wandb_group = "%s_%s/%s"%(args.dataset, args.model, args.wandb_group)
        run = wandb.init(project='power_laws_deep_ensembles', 
                            entity='vetrov_disciples', 
                            group=wandb_group,
                            name=log.full_run_name + '_' + str(num_model),
                            resume=False,
                            reinit=True)
        wandb.config.update(args)
        # wandb.watch(model, log="all")
        run.log_code("./", include_fn=lambda path: (path.endswith(".py") and
                                                    'ipynb_checkpoints' not in path))

        has_bn = utils.check_bn(model)
        for epoch in range(start_epoch, args.epochs + 1):
            time_ep = time.time()

            train_res = utils.train(loaders['train'], model, optimizer, criterion, device, regularizer)

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
            values = [lr, train_res['loss'], train_res['accuracy'], # test_res['nll'],
                        test_res['accuracy'], time_ep]
            
            wandb_log_dict = {k:v for (k, _), v in zip(fmt_list, values)}
            wandb_log_dict['cos_self_train']  = train_res['cos_self']
            wandb_log_dict['cos_other_train'] = train_res['cos_other']
            wandb_log_dict['margin_train']    = train_res['margin']

            for k, v in model.get_weight_norms().items():
                wandb_log_dict[k] = v
            
            wandb_log_dict['correlation_linear'] = model.get_cos_for_last_linear()
            wandb_log_dict['cos_self_test']   = test_res['cos_self']
            wandb_log_dict['cos_other_test']  = test_res['cos_other']
            wandb_log_dict['margin_test']     = test_res['margin']
            wandb_log_dict['epoch']           = epoch
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

        predictions = utils.train_test_predictions_from_scratch(
            dataset = args.dataset, 
            data_path = args.data_path, 
            model = model,
            device = device,
            transform = args.transform,
            aug_predictions = args.aug_predictions)
            
        if not args.not_save_weights:
            np.save(log.path + '/initial_gap%d.npy' % num_model, loaders['train'].dataset.gap_size)

        if hasattr(loaders['train'].dataset, 'gap_size'):
            loaders['train'].dataset.gap_size = gap_editor.get_next_gap(loaders['train'].dataset.gap_size, 
                                                                        predictions['train'][0], 
                                                                        predictions['train'][1], 
                                                                        num_model)
            
        run.finish()
        attempt_number += 1
        if test_res['accuracy'] >= 20 or attempt_number == args.attempts_number:
            if not args.not_save_weights:
                if args.save_embeddings:
                    model.fc.dict_projector.return_embeddings(True)

                if args.aug_predictions or args.save_embeddings:
                    predictions = utils.train_test_predictions_from_scratch(
                        dataset = args.dataset, 
                        data_path = args.data_path, 
                        model = model,
                        device = device,
                        transform = args.transform,
                        aug_predictions = False)
                
                c = OrderedDict()
                c['train'] = torch.from_numpy(predictions['train'][0])
                c['test'] = torch.from_numpy(predictions['test'][0])
                
                if args.save_embeddings:
                    c['kernel'] = model.fc.dict_projector.kernel_tensor
                    model.fc.dict_projector.return_embeddings(False)

                torch.save(c, log.path + '/predictions_run%d.pth' % num_model)
                np.save(log.path + '/initial_gap%d.npy' % num_model, loaders['train'].dataset.gap_size)
            
            num_model += 1
            attempt_number = 0


    return log.path    
        
if __name__ == "__main__":
    main()
