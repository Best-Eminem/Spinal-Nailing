import torch
import torch.nn as nn
import os
import numpy as np
from models import spinal_net
import decoder
import loss
from dataset import BaseDataset
heads = {'hm': 1,
                 'reg': 3*1
                 # 不需要计算corner offset
                 #'wh': 3*4
                 }
model = spinal_net.SpineNet(heads=heads,
                            pretrained=True,
                            down_ratio=4,
                            final_kernel=1,
                            head_conv=256)
a = torch.rand((1,1,240,512,512))
max_pool = nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
b = max_pool(a)
print(b.size())
# output = model(a)
# print(output['hm'].size())
type_size = 4
para = sum([np.prod(list(p.size())) for p in model.parameters()])
# 模型参数占用大小
print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
mods = list(model.modules())
for i in mods:
    print(i)
#spinalNet = mods[]
# input_ = a.clone()
# # 确保不需要计算梯度，因为我们的目的只是为了计算中间变量而已
# input_.requires_grad_(requires_grad=False)
#
# mods = list(model.modules())
# out_sizes = []
#
# for i in range(1, len(mods)):
#     m = mods[i]
#     res_mods = list(m.modules())
#     for j in range(1, len(res_mods)):
#         n = res_mods[j]
#         # 注意这里，如果relu激活函数是inplace则不用计算
#         if isinstance(n, nn.ReLU):
#             if n.inplace:
#                 continue
#         out = n(input_)
#         out_sizes.append(np.array(out.size()))
#         input_ = out
#
# total_nums = 0
# for i in range(len(out_sizes)):
#     s = out_sizes[i]
#     nums = np.prod(np.array(s))
#     total_nums += nums
#
# # 打印两种，只有 forward 和 foreward、backward的情况
# print('Model {} : intermedite variables: {:3f} M (without backward)'
#         .format(model._get_name(), total_nums * type_size / 1000 / 1000))
# print('Model {} : intermedite variables: {:3f} M (with backward)'
#         .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))