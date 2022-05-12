import torch.nn as nn
import torch
import torch.nn.functional as F
from .model_parts import CombinationModule

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

class DecNet(nn.Module):
    def __init__(self, heads, final_kernel, head_conv, channel,spatial = False,segmentation = False):
        #channel 表示在最后一个上采样结束后的通道数
        # channel = 32 head_conv = 256 final_kernel = 1
        super(DecNet, self).__init__()
        self.segmentation = segmentation
        self.spatial = spatial
        self.spatial_size = (15,25,25)
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
            fc1 = nn.Sequential(nn.Conv3d(channel, head_conv, kernel_size=3, padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv3d(head_conv, classes, kernel_size=final_kernel, stride=1,
                                         padding=final_kernel // 2, bias=True)
                               )
            self.fc2 = nn.Sequential(nn.Conv3d(head_conv, classes, kernel_size=final_kernel, stride=1,
                                         padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc1[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc1)
            # # 重载方法 每次调用相当于dict[head] = fc，类似于一个字典，head是key，fc是value
            self.__setattr__(head, fc1)

            if spatial:
                self.spatial_pub = pub(classes,64)
                self.spatial_down = nn.AdaptiveMaxPool3d(self.spatial_size)
                self.spatial_conv = nn.Sequential(nn.Conv3d(64,64,3,stride=1, padding=1),
                                                nn.ReLU(True))
                self.spatial_up = nn.Sequential(nn.Upsample(size=(50,64,64), mode='trilinear',align_corners=False),
                                                )
                self.con_last = nn.Sequential(nn.Conv3d(64, classes, 1),
                                            nn.Sigmoid())


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

        c0_combine = self.dec_c0(c1_combine, x[-6]) if self.segmentation else None

        # temp = nn.Sequential(nn.ReLU(inplace=True))
        # c2_combine = F.interpolate(c2_combine, [60, 100, 100], mode='trilinear', align_corners=False)
        # c2_combine = temp(c2_combine)


        # c2_combine = F.interpolate(c2_combine, [60,100,100], mode='trilinear', align_corners=False)
        # c2_combine = ConvTranspose(c2_combine)
        # c2_combine = F.interpolate(c2_combine, [120, 200, 200], mode='trilinear', align_corners=False)
        #c2_combine = self.ConvTranspose(c2_combine)

        dec_dict = {}
        for head in self.heads:
            #print(c0_combine.shape)
            node = self.__getattr__(head)(c0_combine) if self.segmentation else self.__getattr__(head)(c1_combine)

            if 'hm' in head:
                local_heatmaps = node = torch.sigmoid(node)
            
            #print(dec_dict[head].shape)
            if self.spatial:
                node = self.spatial_pub(node)
                node = self.spatial_down(node)
                for i in range(4):
                    node = self.spatial_conv(node)
                node = self.spatial_up(node)
                spatial_heatmaps = self.con_last(node)                
                #print(spatial_heatmaps.shape)
                result = spatial_heatmaps * local_heatmaps
                dec_dict[head] = result
            else:
                dec_dict[head] = node
            
        return dec_dict