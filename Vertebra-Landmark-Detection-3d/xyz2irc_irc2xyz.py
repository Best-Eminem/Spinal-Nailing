import collections
import numpy as np
import SimpleITK as sitk
# file = sitk.ReadImage('D:\\CT\\1.nii.gz')
# # 图像大小
# file.GetSize()
# # 坐标原点
# origin_xyz = file.GetOrigin()
# print(origin_xyz)
# # 像素间距
# vxSize_xyz = file.GetSpacing()
# # 方向
# direction_a = file.GetDirection()
# print(file.GetSize())
# print(np.array(direction_a).reshape(3,3))
# # 获取影像元数据(返回DICOM tags元组)
# file.GetMetaDataKeys()
#
# # 像素矩阵
# pixel_array = sitk.GetArrayFromImage(file)


IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])
def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(itkImage,coord_xyz):
    origin_a = np.array(itkImage.GetOrigin())
    vxSize_a = np.array(itkImage.GetSpacing())
    direction_a = np.array(itkImage.GetDirection())
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(np.array(direction_a).reshape(3,3))) / vxSize_a
    cri_a = np.round(cri_a)
    #return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))
    return [int(cri_a[2]), int(cri_a[1]), int(cri_a[0])]