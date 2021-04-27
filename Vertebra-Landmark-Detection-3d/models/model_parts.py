import torch.nn as nn
import torch.nn.functional as F
import torch



class CombinationModule(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False, group_norm=False, instance_norm=False):
        super(CombinationModule, self).__init__()
        if batch_norm:
            self.up =  nn.Sequential(nn.Conv3d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm3d(c_up),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv3d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.BatchNorm3d(c_up),
                                           nn.ReLU(inplace=True))
        elif group_norm:
            self.up = nn.Sequential(nn.Conv3d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=32, num_channels=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv3d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.GroupNorm(num_groups=32, num_channels=c_up),
                                          nn.ReLU(inplace=True))
        elif instance_norm:
            self.up = nn.Sequential(nn.Conv3d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.InstanceNorm3d(num_features=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv3d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.InstanceNorm3d(num_features=c_up),
                                          nn.ReLU(inplace=True))
        else:
            self.up =  nn.Sequential(nn.Conv3d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv3d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.ReLU(inplace=True))

    def forward(self, x_low, x_up):
        # 双线性插值“bilinear”,interpolate()函数将输入上 /下采样至指定size
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='trilinear', align_corners=False))
        # 按维数0拼接（竖着拼）
        # C = torch.cat((A, B), 0)
        # 按维数1拼接（横着拼）
        # C = torch.cat((A, B), 1)
        #按第1维合并
        #import torch
# x = torch.rand([1, 512, 1, 32, 16])
# y = torch.rand([1, 512, 1, 32, 16])
# print(torch.cat((x,y),0).shape) = torch.Size([2, 512, 1, 32, 16])
        return self.cat_conv(torch.cat((x_up, x_low), 1))