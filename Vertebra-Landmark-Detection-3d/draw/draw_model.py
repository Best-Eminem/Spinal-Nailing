import cv2
import joblib
import torch
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
import SimpleITK as sitk
from dataset import BaseDataset
from models.spinal_net import SpineNet
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable

heads = {'hm': 1,
                 'reg': 3*1
                 # 不需要计算corner offset
                 #'wh': 3*4
                 }
model = SpineNet(heads=heads,
                 pretrained=True,
                 down_ratio=1,
                 final_kernel=1,
                 head_conv=256)


input = Variable(torch.rand(1, 1,120, 200,200))
output = model(input)
print(1)
# dummy_input = Variable(torch.rand(1, 1,120, 200,200))
# with SummaryWriter(comment='SpineNet') as w:
#     w.add_graph(model, (dummy_input, ))
