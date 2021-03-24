from .dec_net import DecNet
from . import resnet3d
import torch.nn as nn
import numpy as np
import torch

class SpineNet(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(SpineNet, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        channels = [3, 64, 64, 128, 256, 512]
        #down_ratio = 4；l1 = 2
        self.l1 = int(np.log2(down_ratio))
        #网络的第一部分，需要改造为3d
        self.base_network = resnet3d.resnet34(sample_input_D = 3,sample_input_H = 1024,sample_input_W = 512, pretrained=pretrained)
        #网络的第二部分，需要改造为3d
        self.dec_net = DecNet(heads, final_kernel, head_conv, channels[self.l1])


    def forward(self, x):
        x = self.base_network(x)
        dec_dict = self.dec_net(x)
        return dec_dict