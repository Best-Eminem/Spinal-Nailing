import torch.nn as nn
import torch
import torch.nn.functional as F
from .model_parts import CombinationModule

class DecNet(nn.Module):
    def __init__(self, heads, final_kernel, head_conv, channel):
        # channel = 64 head_conv = 256 final_kernel = 1
        super(DecNet, self).__init__()
        self.dec_c0 = CombinationModule(17, 16, batch_norm=True)
        self.dec_c1 = CombinationModule(96, 32, batch_norm=True)
        self.dec_c2 = CombinationModule(128, 64, batch_norm=True)
        self.dec_c3 = CombinationModule(256, 128, batch_norm=True)
        self.dec_c4 = CombinationModule(512, 256, batch_norm=True)
        self.ConvTranspose = nn.Sequential(nn.ConvTranspose3d(64, 64, (4, 4, 4), (2, 2, 2), (1, 1, 1)),
                                      nn.ReLU(inplace=True))
        # heads = {'hm': args.num_classes = 1,(Heatmap)
        #          'reg': 2 * args.num_classes,(Center offset)
        #          'wh': 2 * 4, (corner offset)}
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            # 不需要计算corner offset
            # if head == 'wh':
            #     fc = nn.Sequential(nn.Conv3d(channel, head_conv, kernel_size=7, padding=7//2, bias=True),
            #                        nn.ReLU(inplace=True),
            #                        nn.Conv3d(head_conv, classes, kernel_size=7, padding=7 // 2, bias=True))
            # else:
            #     fc = nn.Sequential(nn.Conv3d(channel, head_conv, kernel_size=3, padding=1, bias=True),
            #                        nn.ReLU(inplace=True),
            #                        nn.Conv3d(head_conv, classes, kernel_size=final_kernel, stride=1,
            #                                  padding=final_kernel // 2, bias=True))
            fc = nn.Sequential(nn.Conv3d(channel, head_conv, kernel_size=3, padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv3d(head_conv, classes, kernel_size=final_kernel, stride=1,
                                         padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)
            # 重载方法 每次调用相当于dict[head] = fc，类似于一个字典，head是key，fc是value
            self.__setattr__(head, fc)


    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        #E5经过上采样与E4合并
        c4_combine = self.dec_c4(x[-1], x[-2])
        # E4经过上采样与E3合并
        c3_combine = self.dec_c3(c4_combine, x[-3])
        # E3经过上采样与E2合并
        c2_combine = self.dec_c2(c3_combine, x[-4])

        c1_combine = self.dec_c1(c2_combine, x[-5])

        #c0_combine = self.dec_c0(c1_combine, x[-6])

        # temp = nn.Sequential(nn.ReLU(inplace=True))
        # c2_combine = F.interpolate(c2_combine, [60, 100, 100], mode='trilinear', align_corners=False)
        # c2_combine = temp(c2_combine)


        # c2_combine = F.interpolate(c2_combine, [60,100,100], mode='trilinear', align_corners=False)
        # c2_combine = ConvTranspose(c2_combine)
        # c2_combine = F.interpolate(c2_combine, [120, 200, 200], mode='trilinear', align_corners=False)
        #c2_combine = self.ConvTranspose(c2_combine)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c1_combine)
            if 'hm' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])
        return dec_dict