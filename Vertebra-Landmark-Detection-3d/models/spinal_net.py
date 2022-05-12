#from models import SCNet
from .dec_net import DecNet
from .seg_net import SegNet
from . import resnet3d
import torch.nn as nn
import numpy as np
import torch

class SpineNet(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv,high_resolution = False,spatial = False,segmentation = False,net='resnet34'):
        super(SpineNet, self).__init__()
        assert down_ratio in [1, 2, 4, 8, 16]
        channels = [1,3,16,32, 64, 128, 256, 512] #channel 表示在最后一个上采样结束后的通道数
        self.net = net
        #down_ratio = 4；l1 = 2
        #self.l1 = int(np.log2(down_ratio))
        self.segmentation = segmentation
        if high_resolution:
            self.l1 = 2
        elif segmentation:
            if net == 'resnet50':
                self.l1 = 4
            else:self.l1 = 2
        else:
            self.l1 = 3
        #网络的第一部分，需要改造为3d
        if segmentation and net == 'resnet50':
            self.base_network = resnet3d.resnet50(sample_input_D = 1,sample_input_H = 512,sample_input_W = 512, pretrained=pretrained)
        else:
            self.base_network = resnet3d.resnet34(sample_input_D = 1,sample_input_H = 512,sample_input_W = 512, pretrained=pretrained)
        #网络的第二部分，需要改造为3d
        if segmentation and net == 'resnet50':
            self.seg_net = SegNet(heads, final_kernel, head_conv, channels[self.l1],spatial = spatial,segmentation=segmentation)
        self.dec_net = DecNet(heads, final_kernel, head_conv, channels[self.l1],spatial = spatial,segmentation=segmentation)
        #self.spatial_net = SCNet()

    def forward(self, x):
        x = self.base_network(x)
        if self.segmentation and self.net == 'resnet50':
            dec_dict = self.seg_net(x)
        else:
            dec_dict = self.dec_net(x)
        return dec_dict