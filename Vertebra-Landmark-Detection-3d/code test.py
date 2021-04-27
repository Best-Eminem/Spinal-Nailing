import cv2
import joblib
import torch
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
import SimpleITK as sitk
from dataset import BaseDataset
from models.spinal_net import SpineNet

'''
def processing_test(image, input_h, input_w):

    image = cv2.resize(image, (input_w, input_h))
    out_image = image.astype(np.float32) / 255.
    plt.imshow(out_image,cmap="gray")
    plt.show()
    out_image = out_image - 0.5
    out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
    out_image = torch.from_numpy(out_image)
    return out_image
# image = cv2.imread('dataPath/data/train/sunhl-1th-11-Jan-2017-254 I AP.jpg',1)
# print(image.shape)
# image = np.array(image)
# processing_test(image,1024,512)
def load_gt_pts(annopath):
    pts = loadmat(annopath)['p2']   # num x 2 (x,y)
    pts = rearrange_pts(pts)
    print(pts.shape)
    return pts
#load_gt_pts('dataPath/labels/train/sunhl-1th-02-Jan-2017-162 A AP.jpg.mat')
# h = np.array(range(9))
# h = h.reshape(1,3,3)
# print(h[0,:,:],h[0,:,:].shape)

'''
# import torch
# x = torch.rand([1, 512, 1, 32, 16])
# y = torch.rand([1, 512, 1, 32, 16])
# print(torch.cat((x,y),0).shape)
# **************************切片为350爆显存了**********************
# X = torch.rand((1, 1, 64, 512, 512))
# heads = {'hm': 1,'reg': 2*1}
# model = SpineNet(heads=heads,pretrained=True,down_ratio=4,final_kernel=1,head_conv=256)
# print(model(X)['hm'].shape,model(X)['reg'].shape)
# torch.Size([1, 1, 16, 128, 128]) torch.Size([1, 2, 16, 128, 128])
##测试读取txt
# def load_gt_pts(landmark_path):
#     # 取出 txt文件中的三位坐标点信息
#     pts = []
#     with open(landmark_path, "r") as f:
#         i = 1
#         for line in f.readlines():
#             line = line.strip('\n')  # 去掉列表中每一个元素的换行符
#             if line != '' and i >= 13:
#                 _,__,x,y,z = line.split()
#                 pts.append((x,y,z))
#             #
#             i+=1
#     # pts = rearrange_pts(pts)
#     return pts
# print(load_gt_pts('E://ZN-CT-nii//labels//train//1.txt'))

##测试basedataset
# dataset = BaseDataset(data_dir='E:\\ZN-CT-nii',
#                                    phase='train',
#                                    input_h=512,
#                                    input_w=512,
#                                    input_s=350,
#                                    down_ratio=4)
# data = dataset.__getitem__(0)
# print(data)

# h = sitk.ReadImage('E://sunhl-1th-01-Mar-2017-312 D AP.jpg')
# h = sitk.GetArrayFromImage(h)
# print(h.shape)
# data_dict = joblib.load('E:\\ZN-CT-nii\\groundtruth\\'+ '6'+'.gt')
# print(1)