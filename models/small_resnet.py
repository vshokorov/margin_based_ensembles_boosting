from torch import nn


__all__ = ['ResNet9', 'ResNet9_BN_logits']



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
    def __init__(self, in_channels, num_classes, norm_logits=False, **kwargs):
        super().__init__()
        # 3 x 32 x 32
        self.conv1 = conv_block(in_channels, 64)         # 64 x 32 x 32
        self.conv2 = conv_block(64, 128, pool=True)      # 128 x 16 x 16
        self.res1 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128))  # 128 x 16 x 16
        
        self.conv3 = conv_block(128, 256, pool=True)    # 256 x 8 x 8
        self.conv4 = conv_block(256, 512, pool=True)    # 512 x 4 x 4
        self.res2 = nn.Sequential(conv_block(512, 512), 
                                  conv_block(512, 512))  # 512 x 4 x 4
        
        self.classifier = [nn.MaxPool2d(4), # 512 x 1 x 1
                           nn.Flatten(),     # 512
                           nn.Dropout(0.2),  
        ]
        if norm_logits:
            self.classifier.append(nn.Linear(512, num_classes, bias=False)) # 100
            self.classifier.append(nn.BatchNorm1d(num_classes))
        else:
            self.classifier.append(nn.Linear(512, num_classes)) # 100

        self.classifier = nn.Sequential(*self.classifier)

        
    def forward(self, xb):
        out1 = self.conv1(xb)
        out2 = self.conv2(out1)
        out3 = self.res1(out2) + out2
        out4 = self.conv3(out3)
        out5 = self.conv4(out4)
        out6 = self.res2(out5) + out5
        out = self.classifier(out6)
        return out


class ResNet9:
    def __init__(self):
        self.base = ResNet9Base
        self.kwargs = {'in_channels':3}

class ResNet9_BN_logits:
    def __init__(self):
        self.base = ResNet9Base
        self.kwargs = {'in_channels':3,
                       'norm_logits': True}