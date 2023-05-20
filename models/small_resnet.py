from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from types import MethodType
import itertools

__all__ = ['Dict_projector', 'ResNet_4_5_Base']


class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the input

def conv_block(in_channels, out_channels, kernel_size=3, pool=False):
    layers = [nn.BatchNorm2d(in_channels, affine = False), 
              nn.Conv2d(in_channels, out_channels, kernel_size, padding=1), 
              nn.BatchNorm2d(out_channels, affine = False), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class InstanceNorm1d_fixed(nn.InstanceNorm1d):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bs = input.size(0)
        x = super().forward(input.view(bs, 1, -1))
        return x.view(bs, -1)

class Tanh_fixed(nn.Tanh):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

def get_weight_norms_from_model(model, last_linear_weights):
    _sum = 0
    for p in model.parameters():
        _sum += p.detach().pow(2).sum()
    
    all_norm = _sum.sqrt()
    last_norm = last_linear_weights.detach().pow(2).sum().sqrt()
    all_except_last_norm = (_sum - last_linear_weights.detach().pow(2).sum()).sqrt()
    return {
        'all_norm': all_norm.item(), 
        'last_norm': last_norm.item(),
        'all_except_last_norm': all_except_last_norm.item()
    }

def get_info_for_last_linear_from_model(model, last_linear_weights):
    cos_sim = F.cosine_similarity(
        last_linear_weights.detach().data.view(1, -1), 
        model.lag_last_linear.detach().view(1, -1)
    )

    l2_dist = torch.norm(
        last_linear_weights.detach().data.view(1, -1) - 
        model.lag_last_linear.detach().view(1, -1)
    )

    model.lag_last_linear = last_linear_weights.data.clone()
    return {
        'correlation_linear': cos_sim.item(),
        'l2_diff_linear': l2_dist.item()
    }

class ResNet9Base(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=32, norm_gen_func_line_layer=None, norm_gen_func_logits=None, skip_connection=False, **kwargs):
        super().__init__()
        self.skip_connection = skip_connection
        # 3 x 32 x 32
        if img_size == 64:
            self.conv1 = nn.Sequential(conv_block(in_channels, 32),
                                       conv_block(32, 64, pool=True))
        else:
            assert img_size == 32
            self.conv1 = conv_block(in_channels, 64)         # 64 x 32 x 32
        self.conv2 = conv_block(64, 128, pool=True)      # 128 x 16 x 16
        self.res1 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128))  # 128 x 16 x 16
        
        self.conv3 = conv_block(128, 256, pool=True)    # 256 x 8 x 8        
        self.conv4 = conv_block(256, 512, pool=True)    # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(512, 512), 
                                  conv_block(512, 512))  # 512 x 4 x 4
        
        if self.skip_connection:
            self.res3 = nn.Sequential(conv_block(64, 64, pool=True), conv_block(64, 64, pool=True), conv_block(64, 512, pool=True))
            self.res4 = nn.Sequential(conv_block(128, 128, pool=True), conv_block(128, 512, pool=True))
           
            self.res5 = nn.Sequential(conv_block(256, 256, pool=True), conv_block(256, 256, pool=True), conv_block(256, 512, pool=True))
            
        fc = [nn.MaxPool2d(4), # 512 x 1 x 1
                           nn.Flatten(),     # 512
                           nn.Dropout(0.2),  
        ]
        
        if not norm_gen_func_line_layer is None:
            fc.append(norm_gen_func_line_layer(512, affine=False))
        
        fc.append(nn.Linear(512, num_classes, bias=False)) # 100
        
        self.last_linear_id = len(fc) - 1
        self.register_buffer('lag_last_linear', fc[self.last_linear_id].weight.data.clone(), persistent=False)

        if not norm_gen_func_logits is None:
            fc.append(norm_gen_func_logits(num_classes, affine=False))
            
        self.fc = nn.Sequential(*fc)

        
    def forward(self, xb):
        if not self.skip_connection: 
            out1 = self.conv1(xb)
            out2 = self.conv2(out1)
            out3 = self.res1(out2) + out2
            out4 = self.conv3(out3)
            out5 = self.conv4(out4)
            out6 = self.res2(out5) + out5
            out = self.fc(out6)
        else:
            out1 = self.conv1(xb)
            out2 = self.conv2(out1)
            out3 = self.res1(out2) + out2
            out4 = self.conv3(out3)
            out5 = self.conv4(out4)
            out6 = self.res2(out5) + out5 + self.res3(out1) + self.res4(out2) + self.res5(out4)
            out = self.fc(out6)
        return out
    
    def get_weight_norms(self):
        return get_weight_norms_from_model(self, self.fc[self.last_linear_id].weight)
    
    def get_info_for_last_linear(self):
        return get_info_for_last_linear_from_model(self, self.fc[self.last_linear_id].weight)


# class TinyResNet9Base(nn.Module):
#     def __init__(self, in_channels, num_classes, norm_logits=True, **kwargs):
#         super().__init__()
#         # 3 x 32 x 32
#         self.conv1 = conv_block(in_channels, 32, pool=True)         # 64 x 32 x 32
#         self.res1 = nn.Sequential(conv_block(32, 32))  # 128 x 16 x 16
        
#         self.conv3 = conv_block(32, 64, pool=True)    # 256 x 8 x 8
#         self.conv3.add_module('4', nn.MaxPool2d(2))    # 512 x 4 x 4
#         self.res2 = nn.Sequential(conv_block(64, 64))  # 512 x 4 x 4
        
#         self.fc = [nn.MaxPool2d(4), # 512 x 1 x 1
#                            nn.Flatten(),     # 512
#                            nn.Dropout(0.2),  
#         ]
#         if norm_logits:
#             self.fc.append(nn.Linear(64, num_classes, bias=False)) # 100
#             self.fc.append(nn.BatchNorm1d(num_classes, affine=False))
#         else:
#             self.fc.append(nn.Linear(64, num_classes, bias=False)) # 100
#             self.fc.append(nn.BatchNorm1d(num_classes, affine=False)) # 100

#         self.fc = nn.Sequential(*self.fc)
    
#     def forward(self, xb):
#         out1 = self.conv1(xb)
#         out3 = self.res1(out1) + out1
#         out4 = self.conv3(out3)
#         out6 = self.res2(out4) + out4
#         out = self.fc(out6)
#         return out

class ResNet9Base_freeze_last(ResNet9Base):
    def __init__(self, scale_last_weight, **args):
        super(ResNet9Base_freeze_last, self).__init__(**args)
        self.fc[self.last_linear_id].weight.requires_grad = False
        # self.last_linear.div_(self.last_linear.data.norm())
        self.fc[self.last_linear_id].weight.data.mul_(scale_last_weight)
        

class ResNet9Base_dict(ResNet9Base):
    def __init__(self, num_classes, emb_dim,
                 normalize_input=True, normalize_dict=True, **args):
        super(ResNet9Base_dict, self).__init__(num_classes = emb_dim, **args)
        
        self.fc.add_module('dict_projector', Dict_projector(emb_dim, num_classes, normalize_input, normalize_dict))


class Dict_projector(nn.Module):
    def __init__(self, hidden_dim, num_classes, normalize_input=True, normalize_dict=True):
        super(Dict_projector, self).__init__()

        self.normalize_input = normalize_input
        self.normalize_dict = normalize_dict
        self.kernel = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(hidden_dim, num_classes)))

        self.__return_embeddings = False
    
    def forward(self, x):
        if self.normalize_input:
            x = F.normalize(x)
        
        if self.__return_embeddings:
            return x
        
        if self.normalize_dict:
            kernel = F.normalize(self.kernel, dim=0)
        else:
            kernel = self.kernel
        return torch.mm(x, kernel)
    
    def return_embeddings(self, state):
        assert isinstance(state, bool)
        self.__return_embeddings = state
    
    @property
    def kernel_tensor(self):
        if self.normalize_dict:
            kernel = F.normalize(self.kernel, dim=0)
        else:
            kernel = self.kernel
        return kernel.data


class ResNet9:
    def __init__(self):
        self.base = ResNet9Base
        self.kwargs = {'in_channels':3}
__all__.append('ResNet9')

class ResNet9_CosFace:
    def __init__(self):
        self.base = ResNet9Base_dict
        self.kwargs = {
            'in_channels':3,
            'norm_gen_func_line_layer': nn.BatchNorm1d, 
            'norm_gen_func_logits': nn.BatchNorm1d,
            'normalize_input':True,
            'normalize_dict' :True,
            'emb_dim': 10, 
        }
__all__.append('ResNet9_CosFace')

code_to_name = {
    'BN': 'nn.BatchNorm1d', 
    'IN': 'InstanceNorm1d_fixed', 
    'tanh': 'Tanh_fixed', 
    '': 'None'
}
for pre in ['BN', 'IN', 'tanh', '']:
    for past in ['BN', 'IN', 'tanh', '']:
        exec(
f"""class ResNet9_{pre}_Linear_{past}:
    def __init__(self):
        self.base = ResNet9Base
        self.kwargs = {{'in_channels':3,
            'norm_gen_func_line_layer': {code_to_name[pre]}, 
            'norm_gen_func_logits': {code_to_name[past]} }}
""")
        __all__.append(f'ResNet9_{pre}_Linear_{past}')

class ResNet9_BN_line_layer_skip_connection:
    def __init__(self):
        self.base = ResNet9Base
        self.kwargs = {'in_channels':3,
                       'skip_connection':True,
                       'norm_gen_func_line_layer': nn.BatchNorm1d}
__all__.append('ResNet9_BN_line_layer_skip_connection')

for scale_float in itertools.chain(range(1, 11), np.logspace(-4, -1, num=4)):
    scale_str = str(scale_float).replace('.', '')
    exec(
f"""class ResNet9_freeze_last_{scale_str}norm:
    def __init__(self):
        self.base = ResNet9Base_freeze_last
        self.kwargs = {{'in_channels':3,
            'norm_gen_func_line_layer': nn.BatchNorm1d,
            'scale_last_weight': {str(scale_float)} }}
""")
    __all__.append(f'ResNet9_freeze_last_{scale_str}norm')


def create_torch_resnet(torch_model_generator, freeze_last=False, **kwargs):
    norm_gen_func_line_layer = kwargs.pop('norm_gen_func_line_layer', None)
    norm_gen_func_logits = kwargs.pop('norm_gen_func_logits', None)
    scale_last_weight = kwargs.pop('scale_last_weight', 1)
    num_classes = kwargs['num_classes']

    if 'emb_dim' in kwargs:
        print("'emb_dim' in architecture.kwargs is not used!")
        kwargs.pop('emb_dim')
    
    if 'img_size' in kwargs:
        print("'img_size' in architecture.kwargs is not used!")
        kwargs.pop('img_size')
    
    m = torch_model_generator(**kwargs)
    fc = []

    fc.append(nn.Linear(512, 512))
    fc.append(nn.ReLU())

    if not norm_gen_func_line_layer is None:
        fc.append(norm_gen_func_line_layer(512, affine=False))
    
    fc.append(m.fc)
    m.last_linear = fc[-1].weight 
    raise NotImplemented("edit last_linear to last_linear_id")
    
    if freeze_last:
        m.last_linear.requires_grad = False
        # m.last_linear.div_(m.last_linear.data.norm())
        m.last_linear.mul_(scale_last_weight)
    
    m.register_buffer('lag_last_linear', m.last_linear.data.clone(), persistent=False)

    if not norm_gen_func_logits is None:
        fc.append(norm_gen_func_logits(num_classes, affine=False))
    
    m.fc = nn.Sequential(*fc)

    m.get_weight_norms = MethodType(lambda self: get_weight_norms_from_model(self), m)
    m.get_info_for_last_linear = MethodType(lambda self: get_info_for_last_linear_from_model(self), m)

    return m


for pre in ['BN', 'IN', 'tanh', '']:
    for past in ['BN', 'IN', 'tanh', '']:
        exec(
f"""class ResNet18_{pre}_Linear_{past}:
    def __init__(self):
        self.base = lambda **kwargs: create_torch_resnet(torchvision.models.resnet18, freeze_last=False, **kwargs)
        self.kwargs = {{
            'norm_gen_func_line_layer': {code_to_name[pre]}, 
            'norm_gen_func_logits': {code_to_name[past]} }}
""")
        __all__.append(f'ResNet18_{pre}_Linear_{past}')

for scale_float in itertools.chain(range(1, 11), np.logspace(-4, -1, num=4)):
    scale_str = str(scale_float).replace('.', '')
    exec(
f"""class ResNet18_freeze_last_{scale_str}norm:
    def __init__(self):
        self.base = lambda **kwargs: create_torch_resnet(torchvision.models.resnet18, freeze_last=True, **kwargs)
        self.kwargs = {{
            'norm_gen_func_line_layer': nn.BatchNorm1d,
            'scale_last_weight': {str(scale_float)} }}
""")
    __all__.append(f'ResNet18_freeze_last_{scale_str}norm')

        
##########################################################################################
     
# class TinyResNet9_BN_Linear_(ResNet9_BN_Linear_):
#     def __init__(self):
#         super().__init__()
#         self.base = TinyResNet9Base




def conv_block_(in_channels, out_channels, kernel_size=3, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet_4_5_Base(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=32, norm_gen_func_logits=None, norm_gen_func_line_layer=None, skip_connection=False, **kwargs):
        super().__init__()
        self.skip_connection = skip_connection
        # 3 x 32 x 32
        if img_size == 64:
            self.conv1 = nn.Sequential(conv_block_(in_channels, 32),
                                       conv_block_(32, 64, pool=True))
        else:
            assert img_size == 32
            self.conv1 = conv_block_(in_channels, 64)         # 64 x 32 x 32
        self.conv2 = conv_block_(64, 128, pool=True)      # 128 x 16 x 16
        self.conv3 = conv_block_(64, 128, pool=True)      # 128 x 16 x 16
        self.conv4 = conv_block_(64, 128, pool=True)      # 128 x 16 x 16
        self.conv5 = conv_block_(64, 128, pool=True)      # 128 x 16 x 16
        
        
        self.res1 = nn.Sequential(conv_block_(128, 128), 
                                  conv_block_(128, 128))  # 128 x 16 x 16
        
      
        
        

        self.fc = [nn.MaxPool2d(16), # 512 x 1 x 1
                           nn.Flatten(),     # 512
                           nn.Dropout(0.2),  
        ]

        if not norm_gen_func_line_layer is None:
            self.fc.append(norm_gen_func_line_layer(128, affine=False))

        self.fc.append(nn.Linear(128, num_classes, bias=False)) # 100
        self.last_linear = self.fc[-1].weight
        self.register_buffer('lag_last_linear', self.last_linear.data.clone(), persistent=False)

        if not norm_gen_func_logits is None:
            self.fc.append(norm_gen_func_logits(num_classes, affine=False))

        self.fc = nn.Sequential(*self.fc)

    def forward(self, xb):       
        out1 = self.conv1(xb)
        out2 = self.conv2(out1)
        out4 = self.conv3(out1)
        out5 = self.conv4(out1)
        out6 = self.conv5(out1)
        
        out3 = self.res1(out2) + out2 + out4 + out5 + out6
        out = self.fc(out3)
        return out

class ResNet_4_5:
    def __init__(self):
        self.base = ResNet_4_5_Base
        self.kwargs = {'in_channels':3}

__all__.append('ResNet_4_5')

  
