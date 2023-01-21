from torch import nn
import torch
import torch.nn.functional as F

__all__ = ['ResNet9_BN_logits_and_line_layer', 'ResNet9_BN_line_layer', 'ResNet9', 'ResNet9_BN_logits', 'ResNet9_norm_emb_dict', 'ResNet9_dict', 'TinyResNet9_BN_logits', 'Dict_projector']


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

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9Base(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=32, norm_logits=False, norm_line_layer=False, **kwargs):
        super().__init__()
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
        
        self.fc = [nn.MaxPool2d(4), # 512 x 1 x 1
                           nn.Flatten(),     # 512
                           nn.Dropout(0.2),  
        ]
        
        if norm_line_layer:
            self.fc.append(nn.BatchNorm1d(512, affine=False))
        
        self.fc.append(nn.Linear(512, num_classes, bias=False)) # 100
        self.last_linear = self.fc[-1].weight
        self.register_buffer('lag_last_linear', self.last_linear.data.clone(), persistent=False)

        if norm_logits:
            self.fc.append(nn.BatchNorm1d(num_classes, affine=False))
            
        self.fc = nn.Sequential(*self.fc)

        
    def forward(self, xb):
        out1 = self.conv1(xb)
        out2 = self.conv2(out1)
        out3 = self.res1(out2) + out2
        out4 = self.conv3(out3)
        out5 = self.conv4(out4)
        out6 = self.res2(out5) + out5
        out = self.fc(out6)
        return out
    
    def get_weight_norms(self):
        _sum = 0
        for p in self.parameters():
            _sum += p.detach().pow(2).sum()
        
        all_norm = _sum.sqrt()
        last_norm = self.last_linear.detach().pow(2).sum().sqrt()
        all_except_last_norm = (_sum - self.last_linear.detach().pow(2).sum()).sqrt()
        return {
            'all_norm': all_norm.item(), 
            'last_norm': last_norm.item(),
            'all_except_last_norm': all_except_last_norm.item()
        }
    
    def get_cos_for_last_linear(self):
        cos_sim = F.cosine_similarity(
            self.last_linear.detach().data.view(1, -1), 
            self.lag_last_linear.detach().view(1, -1)
        )
        self.lag_last_linear = self.last_linear.data.clone()
        return cos_sim.item()


class TinyResNet9Base(nn.Module):
    def __init__(self, in_channels, num_classes, norm_logits=True, **kwargs):
        super().__init__()
        # 3 x 32 x 32
        self.conv1 = conv_block(in_channels, 32, pool=True)         # 64 x 32 x 32
        self.res1 = nn.Sequential(conv_block(32, 32))  # 128 x 16 x 16
        
        self.conv3 = conv_block(32, 64, pool=True)    # 256 x 8 x 8
        self.conv3.add_module('4', nn.MaxPool2d(2))    # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(64, 64))  # 512 x 4 x 4
        
        self.fc = [nn.MaxPool2d(4), # 512 x 1 x 1
                           nn.Flatten(),     # 512
                           nn.Dropout(0.2),  
        ]
        if norm_logits:
            self.fc.append(nn.Linear(64, num_classes, bias=False)) # 100
            self.fc.append(nn.BatchNorm1d(num_classes, affine=False))
        else:
            self.fc.append(nn.Linear(64, num_classes, bias=False)) # 100
            self.fc.append(nn.BatchNorm1d(num_classes, affine=False)) # 100

        self.fc = nn.Sequential(*self.fc)
    
    def forward(self, xb):
        out1 = self.conv1(xb)
        out3 = self.res1(out1) + out1
        out4 = self.conv3(out3)
        out6 = self.res2(out4) + out4
        out = self.fc(out6)
        return out

class ResNet9Base_dict(ResNet9Base):
    def __init__(self, in_channels, num_classes, img_size, emb_dim, 
                 norm_logits=False, normalize_input=True, normalize_dict=True, **args):
        super(ResNet9Base_dict, self).__init__(in_channels, emb_dim, img_size, norm_logits, **args)
        
        self.fc.add_module('dict_projector', Dict_projector(emb_dim, num_classes, normalize_input, normalize_dict))

        self.last_linear = self.fc['dict_projector'].kernel.data
        self.lag_last_linear = self.last_linear.clone()


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

class ResNet9_norm_emb_dict:
    def __init__(self):
        self.base = ResNet9Base_dict
        self.kwargs = {'in_channels':3,
                       'emb_dim': 10, 
                       'img_size': 32, 
                       'normalize_input':True,
                       'normalize_dict' :True}

class ResNet9_dict:
    def __init__(self):
        self.base = ResNet9Base_dict
        self.kwargs = {'in_channels':3,
                       'emb_dim': 10, 
                       'img_size': 32, 
                       'normalize_input':False,
                       'normalize_dict' :False}

class ResNet9_BN_logits:
    def __init__(self):
        self.base = ResNet9Base
        self.kwargs = {'in_channels':3,
                       'norm_logits': True}
        
class ResNet9_BN_line_layer:
    def __init__(self):
        self.base = ResNet9Base
        self.kwargs = {'in_channels':3,
                       'norm_line_layer': True}
        
class ResNet9_BN_logits_and_line_layer:
    def __init__(self):
        self.base = ResNet9Base
        self.kwargs = {'in_channels':3,
                       'norm_logits': True,
                       'norm_line_layer': True}

class TinyResNet9_BN_logits(ResNet9_BN_logits):
    def __init__(self):
        super().__init__()
        self.base = TinyResNet9Base