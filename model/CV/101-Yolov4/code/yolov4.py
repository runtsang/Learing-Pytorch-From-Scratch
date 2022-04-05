from multiprocessing import pool
from re import M
from statistics import mode
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
from Darknet import *
import cv2


def CBL(filter_in, filter_out, kernel_size, stride=1):
    '''
    CBL
    Conv + BN + Leakyrelu
    '''
    
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))
    

class SPP(nn.Module):
    '''
    SPP
    Concat[ n * Maxpool + direct ]
    '''
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SPP, self).__init__()
        
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])
        
    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        
        return features
    
class CBL_UP(nn.Module):
    '''
    CBL + Upsample
    '''
    def __init__(self, in_channels, out_channels):
        super(CBL_UP, self).__init__()
        
        self.upsample = nn.Sequential(
            CBL(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
    def forward(self, x):
        x = self.upsample(x)
        return x
    
def make_five_conv(filters_list, in_filters):
    '''
    Five conv block
    '''
    m = nn.Sequential(
        CBL(in_filters, filters_list[0], 1),
        CBL(filters_list[0], filters_list[1], 3),
        CBL(filters_list[1], filters_list[0], 1),
        CBL(filters_list[0], filters_list[1], 3),
        CBL(filters_list[1], filters_list[0], 1),
    )
    return m

def make_three_conv(filters_list, in_filters):
    '''
    Three conv block
    '''
    m = nn.Sequential(
        CBL(in_filters, filters_list[0], 1),
        CBL(filters_list[0], filters_list[1], 3),
        CBL(filters_list[1], filters_list[0], 1),
    )
    return m

def yolo_head(filters_list, in_filters):
    '''
    Final to get the output
    '''
    m = nn.Sequential(
        CBL(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    
    return m


class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        
        self.backbone = darknet53(None)
        
        self.conv1 = make_three_conv([512,1024],1024)
        self.SPP = SPP()
        self.conv2 = make_three_conv([512,1024],2048)
        
        self.upsample1 = CBL_UP(512,256)
        self.conv_for_P4 = CBL(512,256,1)
        self.make_five_conv1 = make_five_conv([256, 512],512)

        self.upsample2 = CBL_UP(256,128)
        self.conv_for_P3 = CBL(256,128,1)
        self.make_five_conv2 = make_five_conv([128, 256],256)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 4+1+num_classes
        final_out_filter2 = num_anchors * (5 + num_classes)
        self.yolo_head3 = yolo_head([256, final_out_filter2],128)

        self.down_sample1 = CBL(128,256,3,stride=2)
        self.make_five_conv3 = make_five_conv([256, 512],512)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter1 =  num_anchors * (5 + num_classes)
        self.yolo_head2 = yolo_head([512, final_out_filter1],256)


        self.down_sample2 = CBL(256,512,3,stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024],1024)
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        final_out_filter0 =  num_anchors * (5 + num_classes)
        self.yolo_head1 = yolo_head([1024, final_out_filter0],512)
        
    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)
        
        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4,P5_upsample],axis=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3,P4_upsample],axis=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample,P4],axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample,P5],axis=1)
        P5 = self.make_five_conv4(P5)

        out2 = self.yolo_head3(P3)
        out1 = self.yolo_head2(P4)
        out0 = self.yolo_head1(P5)

        return out0, out1, out2
    
if __name__ == '__main__':
    model = YoloBody(3, 80)
    load_model_pth_yolov4(model, 'pth/yolo4_weights.pth')