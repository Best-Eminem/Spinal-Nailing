import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
import SimpleITK as sitk
import itk
import numpy as np
import transform
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def smooth(image, sigma):
    """
    Smooth image with Gaussian smoothing.
    :param image: ITK image.
    :param sigma: Sigma for smoothing.
    :return: Smoothed image.
    """
    ImageType = itk.Image[itk.SS, 3]
    filter = itk.SmoothingRecursiveGaussianImageFilter[ImageType, ImageType].New()
    filter.SetInput(image)
    filter.SetSigma(sigma)
    filter.Update()
    smoothed = filter.GetOutput()
    return smoothed
def elastic_transform(image, alpha, sigma,
                      alpha_affine, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:3]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    # pts1 = np.float32([center_square + square_size,
    #                    [center_square[0] + square_size,
    #                     center_square[1] - square_size],
    #                    center_square - square_size])
    # # pts2: 仿射变换后的点
    # pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
    #                                    size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    # M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    # imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    # imageB = image
    # cv2.imshow("imageB", imageB)
    # cv2.waitKey(0)
    # image = sitk.GetImageFromArray(image)
    # spacing = image.GetSpacing()  # 获取体素的大小
    # origin = image.GetOrigin()  # 获取CT的起点位置
    # size = image.GetSize()  # 获取CT的大小
    # direction = image.GetDirection()  # 获取C的方向
    # itk_information = {}
    # itk_information['spacing'] = spacing
    # itk_information['origin'] = origin
    # itk_information['size'] = size
    # itk_information['direction'] = direction
    # dim = 3
    # #print("sitk.Version: ",sitk.Version_MajorVersion())
    # transformation_list = []
    # transformation_list.append(transform.InputCenterToOrigin.get(dim = 3,itk_information=itk_information))
    # transformation_list.extend([
    #     # transform.RandomTranslation.get(dim=3, offset=[20,20,20]),
    #                             # transform.RandomRotation.get(dim=3, random_angles=[15] * 3),
    #                             # # 随机旋转
    #                             # transform.RandomScale.get(dim=3, random_scale=0.15),
    #                             transform.OriginToOutputCenter.get(dim = 3, itk_information=itk_information)])
    # #合并transformation
    # #compos = sitk.Transform(dim, sitk.sitkIdentity)

    # transformation_comp = sitk.CompositeTransform(dim)
    # for transformation in transformation_list:
    #     transformation_comp.AddTransform(transformation)
    # sitk_transformation = transformation_comp
    # #sitk_transformation = sitk.DisplacementFieldTransform(sitk.TransformToDisplacementField(transformation_comp, sitk.sitkVectorFloat64, size=size, outputSpacing=spacing))

    # imageB = transation_test.resample(image,
    #                  sitk_transformation,
    #                  output_size = size,
    #                  output_spacing = spacing,
    #                  interpolator=sitk.sitkLinear,
    #                  output_pixel_type=None,
    #                  default_pixel_value=-1024)

    # imageB = sitk.GetArrayFromImage(imageB)
    imageB = image
    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid，meshgrid用于生成坐标矩阵，输入是Z,Y,X的坐标范围，输出是坐标矩阵，大小均为（Z*Y*X）
    z, y, x = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]),np.arange(shape[2])) 

    indices = np.reshape(y + dy, (-1, 1, 1)), np.reshape(z + dz, (-1, 1,1)),np.reshape(x + dx, (-1, 1,1))
    # bilinear interpolation
    imageC = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)

    return imageC


if __name__ == '__main__':
    img_path = 'D:/sub-verse506_dir-iso_L2_ct.nii.gz'
    imageA = sitk.ReadImage(img_path)
    imageA = sitk.GetArrayFromImage(imageA)
    img_show = imageA.copy()
    # imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # Apply elastic transform on image
    imageC = elastic_transform(imageA, imageA.shape[1] * 3,
                                   imageA.shape[1] * 0.08,
                                   imageA.shape[1] * 0.08)
    imageC = sitk.GetImageFromArray(imageC)
    sitk.WriteImage(imageC, 'D:/test.nii.gz')

    # cv2.namedWindow("img_a", 0)
    # cv2.imshow("img_a", img_show)
    # cv2.namedWindow("img_c", 0)
    # cv2.imshow("img_c", imageC)
    # cv2.waitKey(0)