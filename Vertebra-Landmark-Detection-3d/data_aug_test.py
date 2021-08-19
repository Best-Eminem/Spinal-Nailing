import SimpleITK as sitk
import cv2
import joblib
import torch
# from draw_gaussian import *
from draw_3dgaussian import *
import transform
import math
from transform import resize_image_itk
from xyz2irc_irc2xyz import xyz2irc
import torch.nn as nn
from torchvision import transforms as transforms
from matplotlib import pyplot as plt
from numpy import random

# data_aug = {'train': transform.Compose([transform.ConvertImgFloat(),  # 转为float32格式
#                                             transform.PhotometricDistort(), #对比度，噪声，亮度
#                                             transform.Expand(max_scale=1.5, mean=(0, 0, 0)),
#                                             transform.RandomMirror_w(),
#                                             transform.Resize(s=400, h=512, w=512), #Resize了点坐标和img
#                                             #transform.RandomTranslation(dim = 3,offset=[30] * 3)
#                                             ]),  # resize
#             }

# itk_img = sitk.ReadImage("E:\\ZN-CT-nii\\data\\gt\\10.nii.gz")
# spacing = itk_img.GetSpacing()
# spacing = [spacing[0],spacing[1],spacing[2]]
# origin = itk_img.GetOrigin()
# size = itk_img.GetSize()
# itk_img.SetMetaData()
data_dict = joblib.load('E:\\ZN-CT-nii\\groundtruth\\spine_localisation\\' + "50.gt")
input = data_dict['input'].copy()[0]
itk_image = sitk.GetImageFromArray(input)

itk_information = data_dict['itk_information']
spacing = itk_information['spacing']
origin = np.array(itk_information['origin'])
size = itk_information['size']
direction = itk_information['direction']

hm = data_dict['hm'].copy()
hm1 = data_dict['hm'].copy()
hm[hm == 0] = 1
hm1[hm1 == 0] = 1
itk_image.SetDirection(direction)
itk_image.SetOrigin(tuple((origin/4).tolist()))
itk_image.SetSpacing(spacing)

hm_itk = []
for i in range(hm.shape[0]):
    hm_tp = sitk.GetImageFromArray(hm[i])
    hm_tp.SetOrigin(tuple((origin/8).tolist()))
    hm_tp.SetSpacing(spacing)
    hm_tp.SetDirection(direction)
    hm_itk.append(hm_tp)
#itk_hm = sitk.GetImageFromArray(hm)

center = origin + np.array(size)*np.array(spacing)*0.5
now_shape = np.array(list(input.shape)[-1::-1])
center = center / size * now_shape
aug = {'train': transform.Compose([transform.RandomTranslation(dim=3, offset=[5] * 3, spacing=spacing),
                                   #transform.RandomRotation(dim=3, angles=[15] * 3, center=center),
                                   # 随机旋转
                                   #transform.RandomScale(dim=3, scale_factor=0.15, center=center),
                                   # 随机放缩
                                   transform.PhotometricDistort()
                                   ])}
img_array,hm_array = aug['train'](itk_image,hm_itk)
#img_array = sitk.GetArrayFromImage(img_array)
# hm_array = sitk.GetArrayFromImage(hm)
data_dict['hm'] = np.asarray(hm_array)
data_dict['input'] = img_array.reshape(1,img_array.shape[0],img_array.shape[1],img_array.shape[2])
for i in range(0, hm_array[0].shape[0] + 1, 1):
    plt.imshow(img_array[i*2])
    #plt.imshow(hm1[0][i])
    plt.show()
    plt.imshow(hm_array[1][i])
    plt.show()
