import torch
import torch.nn as nn
from torch import cat
import torch.nn.functional as F
from torch.autograd import Variable

from models.unet_base import UnetBase


class pub(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True,spatial=False):
        super(pub, self).__init__()
        if spatial:
            inter_channels = out_channels if in_channels > out_channels else out_channels//2
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            inter_channels = out_channels if in_channels > out_channels else out_channels//2
            self.relu = nn.ReLU(True)
        
        layers = [
                    nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
                    self.relu,
                    nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
                    self.relu
                 ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm3d(inter_channels))
            layers.insert(len(layers)-1, nn.BatchNorm3d(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)

class unet3dEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True,spatial = False):
        super(unet3dEncoder, self).__init__()
        self.pub = pub(in_channels, out_channels, batch_norm,spatial = spatial)
        if spatial:
            self.pool = nn.AvgPool3d(2, stride=2)
        else:
            self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pub(x)
        return x,self.pool(x)

class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2,batch_norm=True, sample=True,spatial = False):
        super(unet3dUp, self).__init__()
        self.spatial = spatial
        if spatial:
            self.pub = pub(in_channels, out_channels, batch_norm,spatial = spatial)
        else:
            self.pub = pub(in_channels//2+in_channels, out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False)
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        if self.spatial:
            x = F.interpolate(x, x1.shape[2:], mode='trilinear', align_corners=False)
        else: 
            x = self.sample(x)
        #c1 = (x1.size(2) - x.size(2)) // 2
        #c2 = (x1.size(3) - x.size(3)) // 2
        #x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        x = cat((x, x1), dim=1)
        x = self.pub(x)
        return x

#基本unet3d
class unet3d(nn.Module):
    def __init__(self,init_channels = 1,class_nums = 20,batch_norm = False,sample = True):
        super(unet3d, self).__init__()
        

        self.en1 = unet3dEncoder(init_channels, 64, batch_norm)
        self.en2 = unet3dEncoder(64, 128, batch_norm)
        self.en3 = unet3dEncoder(128, 256, batch_norm)
        self.en4 = unet3dEncoder(256, 512, batch_norm)
        self.en5 = unet3dEncoder(512, 1024, batch_norm)

        self.up4 = unet3dUp(1024, 512, batch_norm, sample)
        self.up3 = unet3dUp(512, 256, batch_norm, sample)
        self.up2 = unet3dUp(256, 128, batch_norm, sample)
        self.up1 = unet3dUp(128, 64, batch_norm, sample)
        self.con_last = nn.Sequential(nn.Conv3d(64, class_nums, 1),
                                       nn.ReLU(inplace=True))
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print(x.shape)
        x1,x = self.en1(x)
        #print(x1.shape,x.shape)
        x2,x= self.en2(x)
        #print(x2.shape,x.shape)
        x3,x= self.en3(x)
        #print(x3.shape,x.shape)
        x4,x = self.en4(x)
        x5,x = self.en5(x)

    
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        #print(x.shape)
        x = self.up2(x, x2)
        #print(x.shape)
        x = self.up1(x, x1)
        #print(x.shape)
        out = self.con_last(x)

        dec_dict = {}
        dec_dict['msk'] = out
        return dec_dict

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
#冠军论文的spatial net
class unet3d_spatial(nn.Module):
    def __init__(self,input_s,input_h,input_w,init_channels = 1,class_nums = 20,batch_norm = False,sample = True,num_filters_base = 64,spatial_downsample=8):
        super(unet3d_spatial, self).__init__()
        self.spatial_size = (input_s//spatial_downsample,input_h//spatial_downsample,input_w//spatial_downsample)
        self.en1 = unet3dEncoder(init_channels, 64, batch_norm,spatial=True)
        self.en2 = unet3dEncoder(64, 128, batch_norm,spatial=True)
        self.en3 = unet3dEncoder(128, 256, batch_norm,spatial=True)
        self.en4 = unet3dEncoder(256, 512, batch_norm,spatial=True)
        self.en5 = unet3dEncoder(64, 64, batch_norm,spatial=True)

        self.spatial_pub = pub(class_nums,64,spatial=True)
        self.spatial_down = nn.AdaptiveAvgPool3d(self.spatial_size)
        self.spatial_conv = nn.Sequential(nn.Conv3d(num_filters_base,num_filters_base,7,padding='same'),
                                          nn.LeakyReLU(True))
        self.spatial_up = nn.Sequential(nn.Upsample(size=(input_s//2,input_h//2,input_w//2), mode='trilinear',align_corners=False),
                                        nn.Sigmoid())
        self.con_last = nn.Sequential(nn.Conv3d(128, class_nums, 1),
                                       )
        
        self.con_last_spatial = nn.Sequential(nn.Conv3d(64, class_nums, 1),
                                       )

        self.up4 = unet3dUp(512, 256, batch_norm, sample,spatial=False)
        self.up3 = unet3dUp(256, 128, batch_norm, sample,spatial=False)
        self.up2 = unet3dUp(128, 64, batch_norm, sample,spatial=False)
        self.up1 = unet3dUp(64, 32, batch_norm, sample,spatial=False)
        
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print(x.shape)
        x1,x = self.en1(x)
        #print(x1.shape,x.shape)
        x2,x= self.en2(x)
        #print(x2.shape,x.shape)
        x3,x= self.en3(x)
        #print(x3.shape,x.shape)
        x4,x = self.en4(x)
        #x5,_ = self.en5(x)
        #print(x4.shape)

        #print(x5.shape,x4.shape)
        #x = self.up4(x5, x4)
        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        #print(x.shape)
        #x = self.up2(x, x1)
        #print(x.shape)
        #考虑不上采样到原大小，因为爆显存了
        #x = self.up1(x, x1)
        #print(x.shape)
        local_heatmaps = node = self.con_last(x)
        #(local_heatmaps.shape)
        node = self.spatial_pub(node)
        node = self.spatial_down(node)
        for i in range(4):
            node = self.spatial_conv(node)
        node = self.con_last_spatial(node)
        spatial_heatmaps = self.spatial_up(node)
        #print(spatial_heatmaps.shape)
        result = spatial_heatmaps * local_heatmaps
        dec_dict = {}
        dec_dict['hm'] = result
        return dec_dict

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()