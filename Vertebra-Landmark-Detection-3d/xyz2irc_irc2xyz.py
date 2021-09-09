import collections
import numpy as np
import SimpleITK as sitk
file = sitk.ReadImage('F:\\ZN-CT-nii\\data\\train\\7.nii.gz')
# # # 图像大小
# # file.GetSize()
# # # 坐标原点
# # origin_xyz = file.GetOrigin()
# # print(origin_xyz)
# # # 像素间距
# vxSize_xyz = file.GetSpacing()
# #print(vxSize_xyz)
# # # 方向
# direction_a = file.GetDirection()
# print(file.GetSize())
# print(np.array(direction_a).reshape(3,3))
# # 获取影像元数据(返回DICOM tags元组)
# file.GetMetaDataKeys()
#
# # 像素矩阵
# pixel_array = sitk.GetArrayFromImage(file)


# IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
# XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(origin_a,vxSize_a,direction_a,coord_irc):
    direction_a = np.array(direction_a).reshape(3, 3)
    cri_a = np.array(coord_irc)[::-1]
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return coords_xyz
    #return [float(coords_xyz[2]), float(coords_xyz[1]), float(coords_xyz[0])]

def xyz2irc(itkImage,coord_xyz):
    origin_a = np.array(itkImage.GetOrigin())
    vxSize_a = np.array(itkImage.GetSpacing())
    direction_a = np.array(itkImage.GetDirection())
    coord_a = np.array(coord_xyz)
    coord_a = coord_a.astype('float32')
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(np.array(direction_a).reshape(3,3))) / vxSize_a
    #cri_a = np.round(cri_a)
    #return cri_a
    return [float(cri_a[2]), float(cri_a[1]), float(cri_a[0])]

# irc = [79,232,216]
# xyz = irc2xyz(file,irc)
# irc = xyz2irc(file,xyz)
# print(1)