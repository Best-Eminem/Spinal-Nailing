import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
import torch
import torch.nn as nn
import os
import numpy as np
from models import spinal_net
from models.SCNet import unet3d,unet3d_spatial
from models import DenseUNet3d
from torchsummary import summary
from models import unet_transformer
if __name__ == '__main__':
    heads = {'msk': 1,
                    }
    model = DenseUNet3d.DenseUNet3d()
    model = spinal_net.SpineNet(heads=heads,
                                pretrained=False,
                                down_ratio=2,
                                final_kernel=1,
                                head_conv=256,
                                spatial=False,
                                segmentation=True) 
    # model = unet3d_spatial(120,200,200)  
    # model = unet_transformer.UNETR(in_channels=1, out_channels=1, img_size=(64,80,160), feature_size=32, norm_name='batch')
    a = torch.rand((1,1,64,80,160))
    # x = model(a)
    # max_pool = nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
    # b = max_pool(a)
    # print(b.size())
    # # output = model(a)
    # # print(output['hm'].size())
    type_size = 4
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # # 模型参数占用大小
    # print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
    # mods = list(model.modules())
    # for i in mods:
    #     print(i)
    #model = unet3d_spatial()
    # 模型参数占用大小
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
    result = model(a)
    print(result['msk'].shape)
    # print(result.shape)
    # summary(model)
