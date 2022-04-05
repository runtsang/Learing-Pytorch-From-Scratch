from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import OrderedDict
import numpy as np

def load_model_pth_yolov4(model, pth):
    print('Loading weights into state dict, name: %s'%(pth))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pth, map_location=device)
    matched_dict = {}
    for k,v in pretrained_dict.items():
        if np.shape(model_dict[k]) == np.shape(v):
            matched_dict[k] = v
        else:
            print('un matched layers: %s'%k)
    print(len(model_dict.keys()), len(pretrained_dict.keys()))
    print('%d layers matched,  %d layers miss'%(len(matched_dict.keys()), len(model_dict)-len(matched_dict.keys())))
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    print('Finished!')
    return model


def load_model_pth(model, pth):
    print('Loading weights into state dict, name: %s'%(pth))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pth, map_location=device)
    matched_dict = {}
    for k, v in model_dict.items():
        if k.find('backbone') == -1:
            key = 'backbone.'+k
            if np.shape(pretrained_dict[key]) == np.shape(v):
                matched_dict[k] = v

    
    for key in matched_dict:
         print('pretrained items:', key)
    print('%d layers matched,  %d layers miss'%(len(matched_dict.keys()), len(model_dict)-len(matched_dict.keys())))
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)
    print('Finished!')
    return model

class Mish(nn.Module):
    '''
    MISH activation function
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
class CBM(nn.Module):
    '''
    CBM
    CONV + BATCHNORM + MISH
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
class Resblock(nn.Module):
    '''
    Resblock
    CBM + CBM + SKIP CONNET
    '''
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = channels
            
        self.block = nn.Sequential(
            CBM(channels, hidden_channels, 1),
            CBM(hidden_channels, channels, 3),
        )
        
    def forward(self, x):
        return x + self.block(x)
    
class CSPX(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super().__init__()
        
        self.downsample_conv = CBM(in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = CBM(out_channels, out_channels, 1)
            self.split_conv1 = CBM(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                CBM(out_channels, out_channels, 1)
            )
            self.concat_conv = CBM(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = CBM(out_channels, out_channels//2, 1)
            self.split_conv1 = CBM(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(channels=out_channels//2) for _ in range(num_blocks)],
                CBM(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = CBM(out_channels, out_channels, 1)
            
    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        
        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)
        
        return x
    

class CSPDarknet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.inplanes = 32
        self.conv1 = CBM(in_channels=3, out_channels=self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]
        
        self.stages = nn.ModuleList([
            CSPX(self.inplanes, self.feature_channels[0], layers[0], first=True),
            CSPX(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            CSPX(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            CSPX(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            CSPX(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])
        
        self.num_features = 1
        # weight initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](x)
        out5 = self.stages[4](x)

        return out3, out4, out5
    
def darknet53(pretrained):
    model = CSPDarknet([1, 2, 8, 8, 4])
    if pretrained:
        load_model_pth(model, pretrained)
    return model

if __name__ == '__main__':
    backbone = darknet53('pth/yolo4_weights.pth')