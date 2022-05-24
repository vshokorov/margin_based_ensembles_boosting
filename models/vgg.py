"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
from turtle import forward
import torch.nn as nn
import torch
import numpy as np

import curves

__all__ = ['VGG16', 'VGG16BN', 'VGG19', 'VGG19BN', 'get_size', 'compute_k', 'VGG16_phi_r', 'VGG16_dict', 'VGG16_dict_big']

def get_size(k, num_classes=100):
    return (3*9+2+2*2+4*3+8*8+8*num_classes)*k+\
          ((1+2*1+2*2+4*2+2*4*4+8*4+5*8*8)*9+\
           2*8*8)*k*k+num_classes
def compute_k(N, num_classes=100):
    a = ((1+2*1+2*2+4*2+2*4*4+8*4+5*8*8)*9+\
           2*8*8)
    b = (3*9+2+2*2+4*3+8*8+8*num_classes)
    c = num_classes
    return np.round((-b + np.sqrt(4*a*N+(b**2-4*a*c)))/2/a)


def get_config(depth, k=64):
    if depth == 16:
        return [[k, k], [k*2, k*2], [k*4, k*4, k*4], [k*8, k*8, k*8], [k*8, k*8, k*8]]
    else:
        return [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]]


def make_layers(config, batch_norm=False, fix_points=None):
    layer_blocks = nn.ModuleList()
    activation_blocks = nn.ModuleList()
    poolings = nn.ModuleList()

    kwargs = dict()
    conv = nn.Conv2d
    bn = nn.BatchNorm2d
    if fix_points is not None:
        kwargs['fix_points'] = fix_points
        conv = curves.Conv2d
        bn = curves.BatchNorm2d

    in_channels = 3
    for sizes in config:
        layer_blocks.append(nn.ModuleList())
        activation_blocks.append(nn.ModuleList())
        for channels in sizes:
            layer_blocks[-1].append(conv(in_channels, channels, kernel_size=3, padding=1, **kwargs))
            if batch_norm:
                layer_blocks[-1].append(bn(channels, **kwargs))
            activation_blocks[-1].append(nn.ReLU(inplace=True))
            in_channels = channels
        poolings.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layer_blocks, activation_blocks, poolings


class VGGBase(nn.Module):
    def __init__(self, num_classes, depth=16, batch_norm=False, k=64, p=0.5, norm_type=None):
        super(VGGBase, self).__init__()
        config = get_config(depth, k=k)
        layer_blocks, activation_blocks, poolings = make_layers(config, batch_norm)
        self.layer_blocks = layer_blocks
        self.activation_blocks = activation_blocks
        self.poolings = poolings

        self.classifier = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(config[-1][-1], 8*k),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(8*k, 8*k),
            nn.ReLU(inplace=True),
            nn.Linear(8*k, num_classes)
        )
        if norm_type == 'InstanceNorm':
            def norm(x, t=1):
                x = x.view(x.size(0), 1, -1)
                x = nn.functional.instance_norm(x) / t
                x = x.view(x.size(0), -1)
                return x
        elif norm_type == 'L2':
            def norm(x, t=1):
                return x / torch.norm(x, p = 2, dim=1, keepdim=True) / t
        elif norm_type == 'L1':
            def norm(x, t=1):
                return x / torch.norm(x, p = 1, dim=1, keepdim=True) / t
        elif norm_type == 'L_inf':
            def norm(x, t=1):
                return x / torch.norm(x, p = torch.inf, dim=1, keepdim=True) / t
        elif norm_type is None:
            def norm(x, t=1):
                return x / t
        else:
            raise NotImplementedError()
        self.logit_norm = norm
        self.temperature = 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        for layers, activations, pooling in zip(self.layer_blocks, self.activation_blocks,
                                                self.poolings):
            for layer, activation in zip(layers, activations):
                x = layer(x)
                x = activation(x)
            x = pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.logit_norm(x, self.temperature)

        return x


class VGGBase_dict(VGGBase):
    def __init__(self, big_dim, **args):
        super(VGGBase_dict, self).__init__(**args)
        
        if big_dim:
            hidden_dim = self.classifier[-1].in_features
            self.classifier = self.classifier[:-2]
            self.logit_norm = Dict_projector(hidden_dim)
        else:
            self.logit_norm = Dict_projector(10)

    # def forward(self, x):
    #     x = super().forward(x)
    #     return x

class VGGBase_phi_r(VGGBase):
    def __init__(self, **args):
        super(VGGBase_phi_r, self).__init__(**args)
        self.classifier = self.classifier[:-2]

        self.lphi = nn.Linear(self.classifier[-1].out_features, 10)
        self.lr = nn.Linear(self.classifier[-1].out_features, 1)
        
    def forward(self, x):
        x = super(VGGBase_phi_r, self).forward(x)
        phi = self.lphi(x)
        phi = phi.div(torch.norm(phi, dim=1, keepdim=True))
        r = self.lr(x)
        return phi * r


class VGGCurve(nn.Module):
    def __init__(self, num_classes, fix_points, depth=16, k=64, batch_norm=False):
        super(VGGCurve, self).__init__()
        layer_blocks, activation_blocks, poolings = make_layers(get_config(depth, k=k),
                                                                batch_norm,
                                                                fix_points=fix_points)
        self.layer_blocks = layer_blocks
        self.activation_blocks = activation_blocks
        self.poolings = poolings

        self.dropout1 = nn.Dropout()
        self.fc1 = curves.Linear(512, 512, fix_points=fix_points)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc2 = curves.Linear(512, 512, fix_points=fix_points)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = curves.Linear(512, num_classes, fix_points=fix_points)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
                    getattr(m, 'bias_%d' % i).data.zero_()

    def forward(self, x, coeffs_t):
        for layers, activations, pooling in zip(self.layer_blocks, self.activation_blocks,
                                                self.poolings):
            for layer, activation in zip(layers, activations):
                x = layer(x, coeffs_t)
                x = activation(x)
            x = pooling(x)
        x = x.view(x.size(0), -1)

        x = self.dropout1(x)
        x = self.fc1(x, coeffs_t)
        x = self.relu1(x)

        x = self.dropout2(x)
        x = self.fc2(x, coeffs_t)
        x = self.relu2(x)

        x = self.fc3(x, coeffs_t)
        
        raise NotImplementedError

        return x

class Dict_projector(nn.Module):
    def __init__(self, hidden_dim):
        super(Dict_projector, self).__init__()

        self.kernel = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(hidden_dim, 10)))
    
    def forward(self, x, t=1):
        x = x.div(torch.norm(x, dim=1, keepdim=True))
        norm_kernel = self.kernel.div(torch.norm(self.kernel, dim=0, keepdim=True))
        return torch.mm(x, norm_kernel) / t

class VGG16:
    def __init__(self):
        self.base = VGGBase
        self.curve = VGGCurve
        self.kwargs = {
            'depth': 16,
            'batch_norm': False
        }

class VGG16_phi_r:
    def __init__(self):
        self.base = VGGBase_phi_r
        self.curve = VGGCurve
        self.kwargs = {
            'depth': 16,
            'batch_norm': False
        }

class VGG16_dict:
    def __init__(self):
        self.base = VGGBase_dict
        self.curve = VGGCurve
        self.kwargs = {
            'depth': 16,
            'batch_norm': False,
            'big_dim' : False
        }

class VGG16_dict_big:
    def __init__(self):
        self.base = VGGBase_dict
        self.curve = VGGCurve
        self.kwargs = {
            'depth': 16,
            'batch_norm': False,
            'big_dim' : True
        }


class VGG16BN:
    def __init__(self):
        self.base = VGGBase
        self.curve = VGGCurve
        self.kwargs = {
            'depth': 16,
            'batch_norm': True
        }


class VGG19:
    def __init__(self):
        self.base = VGGBase
        self.curve = VGGCurve
        self.kwargs = {
            'depth': 19,
            'batch_norm': False
        }


class VGG19BN:
    def __init__(self):
        self.base = VGGBase
        self.curve = VGGCurve
        self.kwargs = {
            'depth': 19,
            'batch_norm': True
        }
