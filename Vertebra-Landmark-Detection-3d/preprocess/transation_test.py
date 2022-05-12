import copy
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
import torch
from torch._C import dtype
import torch.utils.data as data
from draw.draw_3dgaussian import *
from draw import draw
import numpy as np
import SimpleITK as sitk
from preprocess import transform
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from preprocess.xyz2irc_irc2xyz import xyz2irc, irc2xyz

def elastic_transform(image,mask, alpha, sigma,
                      alpha_affine, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:3]

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
    image = map_coordinates(image, indices, order=1, mode='constant').reshape(shape)
    mask = map_coordinates(mask, indices, order=1, mode='constant').reshape(shape)

    return image,mask
def resample(input_image,
             transform,
             output_size,
             output_spacing=None,
             output_origin=None,
             output_direction=None,
             interpolator=None,
             output_pixel_type=None,
             default_pixel_value=None):
    """
    Resample a given input image according to a transform.
    :param input_image: The input sitk image.
    :param transform: The sitk transformation to apply to the resample filter
    :param output_size: The image size in pixels of the output image.
    :param output_spacing: The spacing in mm of the output image.
    :param output_direction: The direction matrix of the output image.
    :param default_pixel_value: The pixel value of pixels outside the image region.
    :param output_origin: The output origin.
    :param interpolator: The interpolation function. See get_sitk_interpolator() for possible values.
    :param output_pixel_type: The output pixel type.
    :return: The resampled image.
    """
    image_dim = input_image.GetDimension()
    transform_dim = transform.GetDimension()
    assert image_dim == transform_dim, 'image and transform dim must be equal, are ' + str(image_dim) + ' and ' + str(transform_dim)
    output_spacing = output_spacing or [1] * image_dim
    output_origin = output_origin or [0] * image_dim
    output_direction = output_direction or np.eye(image_dim).flatten().tolist()
    #interpolator = interpolator or 'linear'

    sitk_interpolator =interpolator

    # resample the image
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(output_size)
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetOutputOrigin(output_origin)
    resample_filter.SetOutputDirection(output_direction)
    resample_filter.SetTransform(transform)
    if default_pixel_value is not None:
        resample_filter.SetDefaultPixelValue(default_pixel_value)
    if output_pixel_type is None:
        resample_filter.SetOutputPixelType(input_image.GetPixelID())
    else:
        resample_filter.SetOutputPixelType(output_pixel_type)

    # perform resampling
    output_image = resample_filter.Execute(input_image)

    return output_image
class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, input_s=None, down_ratio=4,down_size = 2,mode = None):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.input_s = input_s
        self.down_ratio = down_ratio
        self.class_name = ['__background__', 'cell']
        self.num_classes = 40 #原始表示68个特征点
        self.img_dir = os.path.join(data_dir, 'data', self.phase)
        self.img_ids = sorted(os.listdir(self.img_dir))
        self.down_size = down_size
        self.mode = mode

    def load_image(self, index):
        path = os.path.join(self.img_dir, self.img_ids[index])
        itk_img = sitk.ReadImage(path)
        # image = sitk.GetArrayFromImage(itk_img)
        # image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        return itk_img

    def load_gt_pts(self, landmark_path):
        # 取出 txt文件中的三位坐标点信息
        pts = []
        with open(landmark_path, "r") as f:
            i = 1
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                if line != '' and i >= 13:
                    _, __, x, y, z = line.split()
                    pts.append((x, y, z))
                #
                i += 1
        # pts = rearrange_pts(pts)
        return pts

    def get_landmark_path(self, img_id):
        # eg.  F://ZN-CT-nii//labels//train//xxxx.txt
        index,_,_ = img_id.split('.')
        return os.path.join(self.data_dir, 'labels', self.phase, str(index)+'.txt')

    def load_landmarks(self, index):
        img_id = self.img_ids[index]
        landmark_Folder_path = self.get_landmark_path(img_id)
        pts = self.load_gt_pts(landmark_Folder_path)
        return pts

    def preprocess(self,index, points_num, full, mode):
        # 此函数用来生成groundtruth

        img_id = self.img_ids[index]
        # print(img_id)
        img_id_num = img_id[0:-7]
        image = self.load_image(index)  # image是 itk image格式，不是np格式

        aug_label = True
        # 返回shape为(35，3)的labels列表,排列方式为(z,y,x)
        pts = self.load_landmarks(index)  # x,y,z
        # data_series = \

        data_series = spine_localisation_processing_train(image=image,
                                            pts=pts,
                                            points_num=points_num,
                                            image_s=self.input_s,  # 400
                                            image_h=self.input_h,  # 512
                                            image_w=self.input_w,  # 512
                                            aug_label=aug_label,
                                            img_id=img_id_num,
                                            full=full
                                            )
        data_dict_series = []
        for out_image, pts_2, img_id_num in data_series:
            data_dict = spine_localisation_generate_ground_truth(out_image=out_image,
                                                                          points_num=points_num,
                                                                          pts_2=pts_2,
                                                                          image_s=self.input_s // self.down_ratio,
                                                                          image_h=self.input_h // self.down_ratio,
                                                                          image_w=self.input_w // self.down_ratio,
                                                                          img_id=img_id_num,
                                                                          full=full,
                                                                          down_size=self.down_size,
                                                                          down_ratio=self.down_ratio)
            data_dict_series.append(data_dict)

        return data_dict_series


def transform_landmarks(landmarks, transformation):
    """
    Transforms a list of landmarks for a given sitk transformation.
    :param landmarks: List of landmarks.
    :param transformation: The sitk transformation.
    :return: The list of transformed landmarks.
    """
    transformed_landmarks = []
    for landmark in landmarks:
        transformed_landmark = copy.deepcopy(landmark)
        transformed_landmark = np.array(transformation.TransformPoint(transformed_landmark.astype(np.float64)), np.float32)
        transformed_landmarks.append(transformed_landmark)
    return transformed_landmarks

def sitk_to_np(image_sitk, type=None):
    if type is None:
        return sitk.GetArrayFromImage(image_sitk)
    else:
        return sitk.GetArrayViewFromImage(image_sitk).astype(type)

def find_maximum_coord_in_image(image):
    """
    Return the max coordinate from an image.
    :param image: The np image.
    :return: The coordinate as np array.
    """
    max_index = np.argmax(image)
    coord = np.array(np.unravel_index(max_index, image.shape), np.int32)
    return coord


def refine_coordinate_subpixel(image, coord):
    """
    Refine a local maximum coordinate to the subpixel maximum.
    :param image: The np image.
    :param coord: The coordinate to refine
    :return: The refined coordinate as np array.
    """
    refined_coord = coord.astype(np.float32)
    dim = coord.size
    for i in range(dim):
        if int(coord[i]) - 1 < 0 or int(coord[i]) + 1 >= image.shape[i]:
            continue
        before_coord = coord.copy()
        before_coord[i] -= 1
        after_coord = coord.copy()
        after_coord[i] += 1
        pa = image[tuple(before_coord)]
        pb = image[tuple(coord)]
        pc = image[tuple(after_coord)]
        diff = 0.5 * (pa - pc) / (pa - 2 * pb + pc)
        refined_coord[i] += diff
    return refined_coord


def find_quadratic_subpixel_maximum_in_image(image):
    """
    Return the max value and the subpixel refined coordinate from an image.
    Refine a local maximum coordinate to the subpixel maximum.
    :param image: The np image.
    :return: A tuple of the max value and the refined coordinate as np array.
    """
    coord = find_maximum_coord_in_image(image)
    max_value = image[tuple(coord)]
    refined_coord = refine_coordinate_subpixel(image, coord)
    return max_value, refined_coord

def transform_landmarks_inverse_with_resampling(landmarks, transformation, size, spacing, max_min_distance=None):
    """
    Transforms a list of landmarks by calculating the inverse of a given sitk transformation by resampling from a displacement field.
    :param landmarks: The list of landmark objects.
    :param transformation: The sitk transformation.
    :param size: The size of the output image, on which the landmark should exist.
    :param spacing: The spacing of the output image, on which the landmark should exist.
    :param max_min_distance: The maximum distance of the coordinate calculated by resampling. If the calculated distance is larger than this value, the landmark will be set to being invalid.
    :return: The landmark object with transformed coords.
    """
    transformed_landmarks = copy.deepcopy(landmarks)
    dim = len(size)
    displacement_field = sitk.TransformToDisplacementField(transformation, sitk.sitkVectorFloat32, size=size, outputSpacing=spacing)
    if dim == 2:
        displacement_field = np.transpose(sitk_to_np(displacement_field), [1, 0, 2])
        mesh = np.meshgrid(np.array(range(size[0]), np.float32),
                           np.array(range(size[1]), np.float32),
                           indexing='ij')
        # add meshgrid to every displacement value, as the displacement field is relative to the pixel coordinate
        displacement_field += np.stack(mesh, axis=2) * np.expand_dims(np.expand_dims(np.array(spacing, np.float32), axis=0), axis=0)

        for i in range(len(transformed_landmarks)):
            if transformed_landmarks[i] is None:
                continue
            coords = transformed_landmarks[i]
            # calculate distances to current landmark coordinates
            vec = displacement_field - coords
            distances = np.linalg.norm(vec, axis=2)
            invert_min_distance, transformed_coords = find_quadratic_subpixel_maximum_in_image(-distances)
            min_distance = -invert_min_distance
            if max_min_distance is not None and min_distance > max_min_distance:
                transformed_landmarks[i].is_valid = False
                transformed_landmarks[i] = None
            else:
                transformed_landmarks[i] = transformed_coords
    elif dim == 3:
        displacement_field = np.transpose(sitk_to_np(displacement_field), [2, 1, 0, 3])

        mesh = np.meshgrid(np.array(range(size[0]), np.float32),
                           np.array(range(size[1]), np.float32),
                           np.array(range(size[2]), np.float32),
                           indexing='ij')
        # add meshgrid to every displacement value, as the displacement field is relative to the pixel coordinate
        displacement_field += np.stack(mesh, axis=3) * np.expand_dims(np.expand_dims(np.expand_dims(np.array(spacing, np.float32), axis=0), axis=0), axis=0)

        for i in range(len(transformed_landmarks)):
            if transformed_landmarks[i] is None:
                continue
            coords = transformed_landmarks[i]
            # calculate distances to current landmark coordinates
            vec = displacement_field - coords
            distances = np.linalg.norm(vec, axis=3)
            invert_min_distance, transformed_coords = find_quadratic_subpixel_maximum_in_image(-distances)
            min_distance = -invert_min_distance
            if max_min_distance is not None and min_distance > max_min_distance:
                transformed_landmarks[i].is_valid = False
                transformed_landmarks[i] = None
            else:
                transformed_landmarks[i] = transformed_coords
    return transformed_landmarks


def transform_landmarks_inverse(landmarks, transformation, size, spacing, max_min_distance=None):
    """
    Transforms a landmark object with the inverse of a given sitk transformation. If the transformation
    is not invertible, calculates the inverse by resampling from a dispacement field.
    :param landmarks: The landmark objects.
    :param transformation: The sitk transformation.
    :param size: The size of the output image, on which the landmark should exist.
    :param spacing: The spacing of the output image, on which the landmark should exist.
    :param max_min_distance: The maximum distance of the coordinate calculated by resampling. If the calculated distance is larger than this value, the landmark will be set to being invalid.
                             If this parameter is None, np.max(spacing) * 2 will be used.
    :return: The landmark object with transformed coords.
    """
    try:
        inverse = transformation.GetInverse()
        transformed_landmarks = transform_landmarks(landmarks, inverse)
        for transformed_landmark in transformed_landmarks:
            #if transformed_landmark.is_valid:
            transformed_landmark /= np.array(spacing)
        return transformed_landmarks
    except:
        # consider a distance of 2 pixels as a maximum allowed distance
        # for calculating the inverse with a transformation field
        max_min_distance = max_min_distance or np.max(spacing) * 2
        return transform_landmarks_inverse_with_resampling(landmarks, transformation, size, spacing, max_min_distance)


def preprocess_landmarks(landmarks, transformation, output_size, output_spacing):
    """
    Flip, filter and transform landmarks
    :param landmarks: list of landmarks to flip, filter and transform
    :param transformation: transformation to perform
    :param output_size: The output size.
    :param output_spacing: The output spacing.
    :return: list of flipped, filtered and transformed landmarks
    """
    return transform_landmarks_inverse(landmarks, transformation, output_size, output_spacing, None)



def generate_ground_truth(output_img,
                          pts_2,
                          points_num,
                          image_s,
                          image_h,
                          image_w,
                          img_id,
                          full,
                          down_size,down_ratio,bottom_z,heatmap_sigmas=None):
    #因为要down_size为1/2，所以要将参数大小除2取整数
    pts_2 = transform.rescale_pts(pts_2, down_ratio=down_ratio) #特征点除以down_ratio
    #hm = np.zeros((20,image_s//down_size, image_h//down_size, image_w//down_size),dtype=np.float32)
    reg = np.zeros((points_num, 3), dtype=np.float32)  #reg表示ct_int和ct的差值
    ind = np.zeros((points_num), dtype=np.int64) # 每个landmark坐标值与ct长宽高形成约束？
    reg_mask = np.zeros((points_num), dtype=np.uint8)

    if pts_2[:,0].max()>image_s:
        print('s is big', pts_2[:,0].max())
    if pts_2[:,1].max()>image_h:
        print('h is big', pts_2[:,1].max())
    if pts_2[:,2].max()>image_w:
        print('w is big', pts_2[:,2].max())

    if pts_2.shape[0]!=40:
        print('ATTENTION!! image {} pts does not equal to 40!!! '.format(img_id))

    pts = pts_2
    # 计算同一侧椎弓更上两点在x和y轴上的间距
    ct_landmark = pts
    min_diameter = 99
    for i in range(5):
        zhuigonggeng_left_landmarks_distance = np.sqrt(np.sum((pts[8*i+4, :] - pts[8*i+5, :]) ** 2))
        zhuigonggeng_right_landmarks_distance = np.sqrt(np.sum((pts[8*i+6, :] - pts[8*i+7, :]) ** 2))
        # cen_z1, cen_y1, cen_x1 = np.mean(pts[8*i+4:8*i+6], axis=0)
        # cen_z2, cen_y2, cen_x2 = np.mean(pts[8*i+6:8*i+8], axis=0)
        # ct_landmark.append(pts[7*i+1])
        # for j in range(4):
        #     ct_landmark.append(pts[8 * i + j])

        # ct_landmark.append([cen_z1, cen_y1, cen_x1])
        # ct_landmark.append([cen_z2, cen_y2, cen_x2])
        diameter = min(zhuigonggeng_left_landmarks_distance, zhuigonggeng_right_landmarks_distance)
        min_diameter = min(min_diameter,diameter)
    # 椎弓更的landmark
    #要down_size，所以要除2
    ct_landmark = np.asarray(ct_landmark, dtype=np.float32) / down_size
    ct_landmark_int = np.floor(ct_landmark).astype(np.int32)
    # radius = gaussian_radius((math.ceil(distance_y), math.ceil(distance_x)))
    radius = max(0, int(min_diameter)) // 2
    # 生成脊锥中心点的热图
    # for i in range(points_num//8):
    #     #只预测椎弓根的点
    #     for k in range(4,8):
    #         #hm[i * 4 + k - 4] = draw_umich_gaussian(hm[i * 4 + k - 4], ct_landmark_int[i * 8 + k], radius=heatmap_sigmas[i * 4 + k - 4])
    #         hm[i * 4 + k - 4] = draw_umich_gaussian(hm[i * 4 + k - 4], ct_landmark_int[i * 8 + k], radius=radius)
        #预测40个点
        # for k in range(8):
        #         hm[i * 8 + k] = draw_umich_gaussian(hm[i * 8 + k], ct_landmark_int[i * 8 + k], radius=4)

        #只预测矢状面的点
        # for k in range(4):
        #     hm[i * 4 + k] = draw_umich_gaussian(hm[i * 4 + k], ct_landmark_int[i * 8 + k], radius=4)

    ct_landmark_pedical = []
    ct_landmark_sagittal_plane = []
    for i in range(5):
        ct_landmark_sagittal_plane.append(ct_landmark_int[8 * i + 0])
        ct_landmark_sagittal_plane.append(ct_landmark_int[8 * i + 1])
        ct_landmark_sagittal_plane.append(ct_landmark_int[8 * i + 2])
        ct_landmark_sagittal_plane.append(ct_landmark_int[8 * i + 3])

        ct_landmark_pedical.append(ct_landmark_int[8 * i + 4])
        ct_landmark_pedical.append(ct_landmark_int[8 * i + 5])
        ct_landmark_pedical.append(ct_landmark_int[8 * i + 6])
        ct_landmark_pedical.append(ct_landmark_int[8 * i + 7])
    #只预测椎弓根的所需的点使用ct_landmark_pedical，矢状面的点使用ct_landmark_sagittal_plane，全部点使用ct_landmark_int
    ct_landmark_pedical = np.asarray(ct_landmark_pedical)
    ct_landmark_sagittal_plane = np.asarray(ct_landmark_sagittal_plane)

    max_pool_down_size = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
    avg_pool_down_size = nn.AdaptiveAvgPool3d((int(image_s/down_size/down_ratio),int(image_h/down_size/down_ratio),int(image_w/down_size/down_ratio)))
    intense_image = output_img.copy()
    intense_image = torch.from_numpy(intense_image)
    intense_image = max_pool_down_size(intense_image).numpy()
    intense_image = intense_image.astype(np.float32)
    # draw.draw_by_matplotlib(intense_image[0], ct_landmark_pedical)
    ret = {'input': intense_image,
           'origin_image': output_img,
           #'hm': torch.tensor(hm),
           'landmarks': ct_landmark_int,
           'img_id': img_id,
           'bottom_z': np.ndarray(bottom_z)
           }
    return ret


def processing_train(image, pts, points_num,image_h, image_w, image_s,
                     aug_label, img_id,full,spine_localisation_eval_dict):
    spine_localisation_eval_pts = spine_localisation_eval_dict['pts']
    spine_localisation_eval_center = spine_localisation_eval_dict['pts_center']
    ct_landmarks = np.asarray(pts, dtype='float32')
    if aug_label:
        spacing = image.GetSpacing()  # 获取体素的大小
        origin = image.GetOrigin()  # 获取CT的起点位置
        size = image.GetSize()  # 获取CT的大小
        direction = image.GetDirection()  # 获取C的方向
        itk_information = {}
        itk_information['spacing'] = spacing
        itk_information['origin'] = origin
        itk_information['size'] = size
        itk_information['direction'] = direction
        dim = 3
        data_series = []
        #print("sitk.Version: ",sitk.Version_MajorVersion())
        transformation_list = []
        transformation_list.append(transform.InputCenterToOrigin.get(dim = 3,itk_information=itk_information))
        transformation_list.extend([transform.RandomTranslation.get(dim=3, offset=[30,30,30]),
                                    transform.RandomRotation.get(dim=3, random_angles=[15] * 3),
                                    # 随机旋转
                                    transform.RandomScale.get(dim=3, random_scale=0.15),
                                    transform.OriginToOutputCenter.get(dim = 3, itk_information=itk_information)])
        #合并transformation
        #compos = sitk.Transform(dim, sitk.sitkIdentity)

        transformation_comp = sitk.CompositeTransform(dim)
        for transformation in transformation_list:
            transformation_comp.AddTransform(transformation)
        sitk_transformation = transformation_comp
        #sitk_transformation = sitk.DisplacementFieldTransform(sitk.TransformToDisplacementField(transformation_comp, sitk.sitkVectorFloat64, size=size, outputSpacing=spacing))
        output_image_sitk = resample(image,
                                    sitk_transformation,
                                    output_size = size,
                                    output_spacing = spacing,
                                    interpolator=sitk.sitkLinear,
                                    output_pixel_type=None,
                                    default_pixel_value=-1024)
        output_image = sitk.GetArrayFromImage(output_image_sitk)
    #     Transforms an image by first shifting and scaling, and then optionally clamps the values.
    # Order of operations:
    #     image += shift
    #     image *= scale
    #     image += random.float_uniform(-random_shift, random_shift)
    #     image *= 1 + random.float_uniform(-random_scale, random_scale)
    #     image = np.clip(image, clamp_min, clamp_max)
        output_image = output_image/2048
        output_image += float(np.random.uniform(-0.25,1.25))
        output_image *= float(np.random.uniform(0.75,1.25))
        output_image = np.clip(output_image,-1,1)

        preprocessed_landmarks = preprocess_landmarks(ct_landmarks.copy(), sitk_transformation, size, spacing)
        preprocessed_landmarks = np.array(np.round(np.array(preprocessed_landmarks)[:, [2, 1, 0]]), dtype='int32')

        spine_localisation_eval_pts_preprocessed = []
        for i in range(len(spine_localisation_eval_pts)):
            spine_localisation_eval_pts_preprocessed.append(irc2xyz(origin,spacing,direction,spine_localisation_eval_pts[i].copy()))
        spine_localisation_eval_pts_preprocessed = preprocess_landmarks(spine_localisation_eval_pts_preprocessed, sitk_transformation, size, spacing)
        spine_localisation_eval_pts_preprocessed = np.array(np.round(np.array(spine_localisation_eval_pts_preprocessed)[:, [2, 1, 0]]), dtype='int32')

        spine_localisation_eval_center_preprocessed = [irc2xyz(origin,spacing,direction,spine_localisation_eval_center.copy())]
        spine_localisation_eval_center_preprocessed = preprocess_landmarks(spine_localisation_eval_center_preprocessed, sitk_transformation, size, spacing)
        spine_localisation_eval_center_preprocessed = np.array(np.round(np.array(spine_localisation_eval_center_preprocessed)[:, [2, 1, 0]]), dtype='int32')
    else:
        preprocessed_landmarks = []
        for i in range(len(ct_landmarks)):
            preprocessed_landmarks.append(xyz2irc(image, ct_landmarks[i]))
        preprocessed_landmarks = np.asarray(preprocessed_landmarks)
        spine_localisation_eval_pts_preprocessed = spine_localisation_eval_pts
        spine_localisation_eval_center_preprocessed = [spine_localisation_eval_center]
        output_image = sitk.GetArrayFromImage(image)
        output_image = output_image / 2048
        output_image = np.clip(output_image,-1,1)

    # draw.draw_points_test(output_image, preprocessed_landmarks)
    #print('done')

    # spine_localisation_bottom_z = spine_localisation_eval_dict['spine_localisation_bottom_z'][0]
    # spine_localisation_eval_center_preprocessed[0] += spine_localisation_bottom_z
    bo = preprocessed_landmarks[:, 0].min()
    # spine_localisation_eval_pts_preprocessed[:,0] += spine_localisation_bottom_z
    bottom_z = (spine_localisation_eval_pts_preprocessed[0][0] - 30) if spine_localisation_eval_pts_preprocessed[0][
                                                                            0] - 30 >= 0 else 0
    bottom_z = np.floor(bottom_z).astype(np.int32)
    top_z = spine_localisation_eval_pts_preprocessed[4][0] + 30
    top_z = np.ceil(top_z).astype(np.int32)
    # 所有点的z轴坐标要改变

    preprocessed_landmarks[:, 0] = preprocessed_landmarks[:, 0] - bottom_z
    # 根据预测的整个ct的中心点的x,y坐标取出原ct的一部分
    spine_localisation_eval_center_preprocessed = spine_localisation_eval_center_preprocessed[0]
    pts_center_y = spine_localisation_eval_center_preprocessed[1]
    pts_center_x = spine_localisation_eval_center_preprocessed[2]
    # 所有点的x，y轴坐标要减小
    preprocessed_landmarks[:, 1] = preprocessed_landmarks[:, 1] - ((pts_center_y - 200) if pts_center_y - 200 >= 0 else 0)
    preprocessed_landmarks[:, 2] = preprocessed_landmarks[:, 2] - ((pts_center_x - 200) if pts_center_x - 200 >= 0 else 0)
    # 截取包含5段脊椎的部分
    image_array_section = output_image[bottom_z:top_z,
                          (pts_center_y - 200) if pts_center_y - 200 >= 0 else 0:(
                                      pts_center_y + 200) if pts_center_y + 200 <= 512 else 512,
                          (pts_center_x - 200) if pts_center_x - 200 >= 0 else 0:(
                                      pts_center_x + 200) if pts_center_x + 200 <= 512 else 512]

    output_image,preprocessed_landmarks = transform.Resize.resize(img = image_array_section,pts = preprocessed_landmarks,input_s=image_s, input_h=image_h, input_w=image_w)
    #print(preprocessed_landmarks)
    # if aug_label == False:
    # draw.draw_by_matplotlib(output_image, preprocessed_landmarks)
    output_image = np.reshape(output_image, (1, image_s, image_h, image_w))
    data_series = []
    data_series.append((output_image, preprocessed_landmarks,img_id,bottom_z)) #这里的bottom_z是resize之前的，是原始CT中的点或原始CT数据增强后的点
                                                                               #若想要将第二部预测的点恢复到原始CT的话，需要先resize点，再将预测点z坐标加上bottom_z




    #data_series.append((out_image,intense_image, pts2,img_id,bottom_z))

    #return np.asarray(out_image, np.float32), pts2
    return data_series

def spine_localisation_processing_train(image, pts, points_num,image_h, image_w, image_s, aug_label, img_id, full):
    pts = np.array(pts, dtype='float32')
    ct_landmarks = []
    for i in range(points_num):
        temp1 = np.mean(pts[8 * i + 0:8 * i + 2], axis=0)
        temp2 = np.mean(pts[8 * i + 2:8 * i + 4], axis=0)
        cen_z, cen_y, cen_x = np.mean([temp1, temp2], axis=0)
        ct_landmarks.append([cen_z, cen_y, cen_x])
    ct_landmarks = np.asarray(ct_landmarks, dtype='float32')
    if aug_label:
        spacing = image.GetSpacing()  # 获取体素的大小
        origin = image.GetOrigin()  # 获取CT的起点位置
        size = image.GetSize()  # 获取CT的大小
        direction = image.GetDirection()  # 获取C的方向
        itk_information = {}
        itk_information['spacing'] = spacing
        itk_information['origin'] = origin
        itk_information['size'] = size
        itk_information['direction'] = direction
        dim = 3
        data_series = []
        #print("sitk.Version: ",sitk.Version_MajorVersion())
        transformation_list = []
        transformation_list.append(transform.InputCenterToOrigin.get(dim = 3,itk_information=itk_information))
        transformation_list.extend([
                                    transform.RandomTranslation.get(dim=3, offset=[30] * 3),
                                    transform.RandomRotation.get(dim=3, random_angles=[15] * 3),
                                    # 随机旋转
                                    transform.RandomScale.get(dim=3, random_scale=0.15),
                                    transform.OriginToOutputCenter.get(dim = 3, itk_information=itk_information)])
        #合并transformation
        #compos = sitk.Transform(dim, sitk.sitkIdentity)

        transformation_comp = sitk.CompositeTransform(dim)
        for transformation in transformation_list:
            transformation_comp.AddTransform(transformation)
        sitk_transformation = transformation_comp
        #sitk_transformation = sitk.DisplacementFieldTransform(sitk.TransformToDisplacementField(transformation_comp, sitk.sitkVectorFloat64, size=size, outputSpacing=spacing))
        output_image_sitk = resample(image,
                                    sitk_transformation,
                                    output_size = size,
                                    output_spacing = spacing,
                                    interpolator=sitk.sitkLinear,
                                    output_pixel_type=None,
                                    default_pixel_value=-1024)
        output_image = sitk.GetArrayFromImage(output_image_sitk)
        #     Transforms an image by first shifting and scaling, and then optionally clamps the values.
    # Order of operations:
    #     image += shift
    #     image *= scale
    #     image += random.float_uniform(-random_shift, random_shift)
    #     image *= 1 + random.float_uniform(-random_scale, random_scale)
    #     image = np.clip(image, clamp_min, clamp_max)
        output_image = output_image/2048
        #改变像素值不利于特征点识别
        # output_image += float(np.random.uniform(-0.25,1.25))
        # output_image *= float(np.random.uniform(0.75,1.25))
        output_image = np.clip(output_image,-1,1)
        if len(transformation_list) == 0:
            preprocessed_landmarks = []
            for i in range(len(ct_landmarks)):
                preprocessed_landmarks.append(xyz2irc(image, ct_landmarks[i]))
            preprocessed_landmarks = np.asarray(preprocessed_landmarks)
        else:
            preprocessed_landmarks = preprocess_landmarks(ct_landmarks.copy(), sitk_transformation, size, spacing)
            preprocessed_landmarks = np.array(np.round(np.array(preprocessed_landmarks)[:, [2, 1, 0]]), dtype='int32')
    else:
        preprocessed_landmarks = []
        for i in range(len(ct_landmarks)):
            preprocessed_landmarks.append(xyz2irc(image, ct_landmarks[i]))
        preprocessed_landmarks = np.asarray(preprocessed_landmarks)
        output_image = sitk.GetArrayFromImage(image)
        output_image = output_image / 2048
        output_image = np.clip(output_image,-1,1)

    # draw.draw_points_test(output_image, preprocessed_landmarks)
    #print('done')
    origin_size = output_image.shape
    output_image,preprocessed_landmarks = transform.Resize.resize(img = output_image,pts = preprocessed_landmarks,input_s=image_s, input_h=image_h, input_w=image_w)
    # if aug_label == False:
    #     draw.draw_by_matplotlib(output_image, preprocessed_landmarks)
    output_image = np.reshape(output_image, (1, image_s, image_h, image_w))
    data_series = []
    data_series.append((output_image, preprocessed_landmarks,img_id,np.asarray(origin_size)))

    return data_series

def spine_localisation_generate_ground_truth(
                          output_img,
                          pts_2,
                          points_num,
                          image_s,
                          image_h,
                          image_w,
                          img_id,
                          full,
                          down_size,down_ratio,origin_size):
    #因为要down_size为1/2，所以要将参数大小除2取整数
    pts_2 = transform.rescale_pts(pts_2, down_ratio=down_ratio)
    #hm = np.zeros((5,image_s//down_size, image_h//down_size, image_w//down_size), dtype=np.float32)

    if pts_2[:,0].max()>image_s:
        print('s is big', pts_2[:,0].max())
    if pts_2[:,1].max()>image_h:
        print('h is big', pts_2[:,1].max())
    if pts_2[:,2].max()>image_w:
        print('w is big', pts_2[:,2].max())

    if pts_2.shape[0]!=5:
        print('ATTENTION!! image {} pts does not equal to 5!!! '.format(img_id))

    pts = pts_2
    # 计算同一侧椎弓更上两点在x和y轴上的间距
    ct_landmark = pts
    #min_diameter = 12
    # 椎弓更的landmark
    #要down_size，所以要除
    ct_landmark = np.asarray(ct_landmark, dtype=np.float32)/down_size
    ct_landmark_int = np.floor(ct_landmark).astype(np.int32)
    #radius = gaussian_radius((math.ceil(distance_y), math.ceil(distance_x)))
    # radius = 6
    # #生成脊锥中心点的热图
    # for i in range(points_num):
    #     hm[i] = draw_umich_gaussian(hm[i], ct_landmark_int[i], radius=radius, scale = 2)

    max_pool_down_size = nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
    intense_image = output_img.copy()
    intense_image = torch.from_numpy(intense_image)
    intense_image = max_pool_down_size(intense_image)
    intense_image = max_pool_down_size(intense_image).numpy()
    intense_image = intense_image.astype(np.float32)

    ret = {'input': intense_image,
           'origin_image':output_img,
           #'hm': hm,
           'landmarks':ct_landmark_int,
           'img_id':img_id,
           'origin_size':origin_size
           }
    return ret


def lumbar_segmentation_process(image,msk,image_h, image_w, image_s, aug_label, img_id):
    # aug_label = False
    if aug_label:
        spacing = image.GetSpacing()  # 获取体素的大小
        origin = image.GetOrigin()  # 获取CT的起点位置
        size = image.GetSize()  # 获取CT的大小
        direction = image.GetDirection()  # 获取CT的方向
        itk_information = {}
        itk_information['spacing'] = spacing
        itk_information['origin'] = origin
        itk_information['size'] = size
        itk_information['direction'] = direction
        dim = 3
        data_series = []
        #print("sitk.Version: ",sitk.Version_MajorVersion())
        transformation_list = []
        # transformation_list.append(transform.InputCenterToOrigin.get(dim = 3,itk_information=itk_information))
        # transformation_list.extend([transform.RandomTranslation.get(dim=3, offset=[10,1,10]),
        #                             transform.RandomRotation.get(dim=3, random_angles=[15] * 3),
        #                             # 随机旋转
        #                             transform.RandomScale.get(dim=3, random_scale=0.15),
        #                             transform.OriginToOutputCenter.get(dim = 3, itk_information=itk_information)])
        #合并transformation
        #compos = sitk.Transform(dim, sitk.sitkIdentity)

        transformation_comp = sitk.CompositeTransform(dim)
        for transformation in transformation_list:
            transformation_comp.AddTransform(transformation)
        sitk_transformation = transformation_comp
        #sitk_transformation = sitk.DisplacementFieldTransform(sitk.TransformToDisplacementField(transformation_comp, sitk.sitkVectorFloat64, size=size, outputSpacing=spacing))
        output_image_sitk = resample(image,
                                    sitk_transformation,
                                    output_size = size,
                                    output_spacing = spacing,
                                    interpolator=sitk.sitkLinear,
                                    output_pixel_type=None,
                                    default_pixel_value=-1024)
        output_image = sitk.GetArrayFromImage(output_image_sitk)
        msk_image_sitk = resample(msk,
                                     sitk_transformation,
                                     output_size=size,
                                     output_spacing=spacing,
                                     interpolator=sitk.sitkLinear,
                                     output_pixel_type=None,
                                     default_pixel_value=0)
        msk_image = sitk.GetArrayFromImage(msk_image_sitk)
        # 数据增强之弹性变形
        # distort_intensity = np.random.uniform(low=float(2.5), high=float(4), size=None)
        # output_image, msk_image = elastic_transform(output_image, msk_image,output_image.shape[1] * distort_intensity,
        #                            output_image.shape[1] * 0.08,
        #                            output_image.shape[1] * 0.08)
        #     Transforms an image by first shifting and scaling, and then optionally clamps the values.
    # Order of operations:
    #     image += shift
    #     image *= scale
    #     image += random.float_uniform(-random_shift, random_shift)
    #     image *= 1 + random.float_uniform(-random_scale, random_scale)
    #     image = np.clip(image, clamp_min, clamp_max)
        output_image = output_image/2048
        #做体素值的增强
        # output_image += float(np.random.uniform(-0.25,0.25))
        # output_image *= float(np.random.uniform(0.75,1.25))
        output_image = np.clip(output_image,-1,1)

        msk_image = np.clip(msk_image, 0, 1)
        msk_image[msk_image != 0] = 1

    else:
        output_image = sitk.GetArrayFromImage(image)
        output_image = output_image / 2048
        output_image = np.clip(output_image,-1,1)
        if msk != None:
            msk_image = sitk.GetArrayFromImage(msk)
            msk_image = np.clip(msk_image, 0, 1)
            msk_image[msk_image != 0] = 1
        else:msk_image = msk

    # draw.draw_points_test(output_image, preprocessed_landmarks)
    #print('done')
    origin_size = output_image.shape
    output_image,_ = transform.Resize.resize(img = output_image,pts = np.asarray([[1,1,1]]),input_s=image_s, input_h=image_h, input_w=image_w)
    # if aug_label == False:
    #     draw.draw_points_test(output_image, preprocessed_landmarks)
    output_image = np.reshape(output_image, (1, image_s, image_h, image_w))
    output_image = np.asarray(output_image,dtype='float32')

    if msk_image.shape[0]!=0:
        msk_image, _ = transform.Resize.resize(img=msk_image, pts=np.asarray([[1,1,1]]), input_s=image_s, input_h=image_h,input_w=image_w)
        # msk_image = np.reshape(msk_image, (1, image_s, image_h, image_w))
        msk_image = np.expand_dims(msk_image,axis=0)
        msk_image = np.asarray(msk_image,dtype='float32')
        msk_image[msk_image != 0] = 1.0
    else:
        msk_image = np.asarray([[[1.2,2.5,3.1],[1.2,2.5,3.1]]])

    ret = {'input': output_image,
            #'img_id': img_id,
           'msk':msk_image,
           'origin_size': np.asarray(origin_size)
           }

    return ret


if __name__ == '__main__':
    dataset = BaseDataset(data_dir='F:\\ZN-CT-nii',
                          phase='gt',
                          input_h=512,
                          input_w=512,
                          input_s=400,
                          down_ratio=2, down_size=4)
    data_dict_series = dataset.preprocess(index=7,points_num=5,full=True,mode='1')