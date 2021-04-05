import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat

from dataset import BaseDataset
from models.spinal_net import SpineNet

'''
def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k+4,:]
        #返回排序后的下标
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:,1])
        y_inds_r = np.argsort(pt_r[:,1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
    return np.asarray(boxes, np.float32)

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
# import torch
# x = torch.rand([1, 512, 1, 32, 16])
# y = torch.rand([1, 512, 1, 32, 16])
# print(torch.cat((x,y),1).shape)
'''
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
# dataset.__getitem__(1)
