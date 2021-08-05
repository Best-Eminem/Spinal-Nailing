import cv2
import joblib
import torch
import os
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
def define_area(point1, point2, point3):
    """
    法向量    ：n={A,B,C}
    :return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    AB = np.asmatrix(point2 - point1)
    AC = np.asmatrix(point3 - point1)
    N = np.cross(AB, AC)  # 向量叉乘，求法向量
    # Ax+By+Cz
    Ax = N[0, 0]
    By = N[0, 1]
    Cz = N[0, 2]
    D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
    return Ax, By, Cz, D


def point2area_distance(point1, point2, point3, point4):
    """
    :param point1:数据框的行切片，三维
    :param point2:
    :param point3:
    :param point4:
    :return:点到面的距离
    """
    Ax, By, Cz, D = define_area(point1, point2, point3)
    mod_d = Ax * point4[0] + By * point4[1] + Cz * point4[2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(mod_d) / mod_area
    return d


if __name__ == '__main__':
    # 初始化数据
    # point1 = [24, 10, 3]
    # point2 = [19, 17, 4]
    # point3 = [26, 15, 4]
    # point4 = [8, -4, -5]
    # #法向量[[  2   7 -39]]
    # # 计算点到面的距离
    #
    # d1 = point2area_distance(point1, point2, point3, point4)  # s=8.647058823529413
    # print("点到面的距离s: " + str(d1))
    # # list = [[1,2,3],[4,5,6]]
    # # list = np.asarray(list)
    # # list = torch.from_numpy(list)
    # # list = list[:2,:2]*3
    # # list = list.data
    # # print('1')
    # # a = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
    # # a = a.view(1,-1,1)
    # # print(a)
    # # b = torch.randint(0,4,(1,5))
    # # b = b.unsqueeze(2)
    # # b = b.expand(1,5,3)
    # # # print(b)
    # # tp = a.gather(1,b)
    # # # print(tp)
    # # b.requires_grad = True
    # # tp.backward()
    # # print(b.grad,a.grad)
    # z= []
    # for i in range(1,28):
    #     path = os.path.join("E:\\ZN-CT-nii\\data\\gt", str(i)+'.nii.gz')
    #     itk_img = sitk.ReadImage(path)
    #     img = sitk.GetArrayFromImage(itk_img)
    #     size = img.shape
    #     z.append(size[0])
    # for i in range(29,51):
    #     path = os.path.join("E:\\ZN-CT-nii\\data\\gt", str(i)+'.nii.gz')
    #     itk_img = sitk.ReadImage(path)
    #     img = sitk.GetArrayFromImage(itk_img)
    #     size = img.shape
    #     z.append(size[0])
 #    z = [356,594 ,438 ,438 ,332, 401, 424, 394, 433, 517, 363, 397, 554, 351, 279, 563, 388, 376,
 # 389, 351, 440, 338, 357, 338, 371, 372, 336, 357, 291, 544, 413, 382, 419, 394, 457, 376,
 # 313, 555, 413, 344, 369, 363, 313, 451, 351, 351, 401, 426, 407]
 #    z = np.asarray(z)
 #    print(z.mean())