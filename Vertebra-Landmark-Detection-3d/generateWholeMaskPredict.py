from glob import glob
import random
import shutil
import joblib
import json
import torch
import numpy as np
import sys
import os

from torch._C import dtype
from preprocess.transation_test import resample
o_path = os.getcwd()
sys.path.append(o_path)
from draw.draw import draw_points_test, draw_by_matplotlib
from draw.draw_distribute_points import get_points_from_point
from draw.draw_distribute_points import structureRotateLineByAngle
from models import spinal_net
import SimpleITK as sitk
from utils import decoder
from preprocess import transform
import torch.nn as nn
from dataset.dataset import BaseDataset
import loss
from matplotlib import pyplot as plt

class Network(object):
    def __init__(self):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # heads表示的是最后一层输出的通道数
        heads = {'hm': 5,
                 }

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=False,
                                         down_ratio=2,
                                         final_kernel=1,
                                         head_conv=256,
                                         #spatial=True
                                         )

        self.dataset = {'spinal': BaseDataset}
        self.criterion = loss.LossAll()
        self.points_num = 1 #K为特征点的最大个数
        self.decoder = decoder.DecDecoder(K=self.points_num, conf_thresh=0.2)  # K为特征点的最大个数
        self.down_ratio = 2
        self.down_size = 4
        self.mode = "spine_localisation"
        self.output_channel = 20

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def plain_angle(self,points_plane):
        """
        法向量    ：n={A,B,C}
        :return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
        """
        point1, point2, point3, point4 = points_plane
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        point3 = np.asarray(point3)
        point4 = np.asarray(point4)
        AB = np.asmatrix(point2 - point1)
        CD = np.asmatrix(point4 - point3)
        N = np.cross(AB, CD)  # 向量叉乘，求法向量
        # Ax+By+Cz
        Ax = N[0, 0]
        By = N[0, 1]
        Cz = N[0, 2]
        # D = -(Ax * point3[0] + By * point3[1] + Cz * point3[2])

        normal1 = np.asarray([Ax,By,Cz])
        normal2 = np.asarray([0,0,1])
        data_M = np.sqrt(np.sum(normal1 * normal1))
        data_N = np.sqrt(np.sum(normal2 * normal2))
        cos_theta = np.sum(normal1 * normal2) / (data_M * data_N)
        theta = np.degrees(np.arccos(cos_theta))  # 角点b的夹角值
        if theta>90:
            theta = 180-theta
        if theta>15:
            theta = 15
        return theta
    def rotate_points_by_line(self,points,z_distance,centerPoint,pedicalCenterPointsY,angle):
        # 依据矢状面预测点中上侧远离追弓根的那个点，框体长度120，高度40，宽度50，进行倾斜并绘制
        box_length = pedicalCenterPointsY+45  #15为小边界框，20为大边界框，35为完整边界框
        box_height = z_distance if 5<z_distance<70 else 40
        box_width = 80
        sign_bit = 1
        point_a, point_b = points
        if point_a[1] > point_b[1]:
            temp = point_b.copy()
            point_b = point_a
            point_a = temp
        point_a[1] -= 5
        #point_a[2] += (2 if point_a[2] >= point_b[2] else 5)
        point_b = point_a.copy();point_b[1] = box_length;
        point_c = point_a.copy();point_c[2] -= box_height
        point_d = point_b.copy();point_d[2] -= box_height

        # if point_c[1] < point_d[1]:
        #     point_d = get_point_from_line(point_c, point_d)
        # else:
        #     point_c = get_point_from_line(point_c, point_d)

        point_a_left, point_a_right = get_points_from_point(point_a, box_width)
        point_b_left, point_b_right = get_points_from_point(point_b, box_width)
        point_c_left, point_c_right = get_points_from_point(point_c, box_width)
        point_d_left, point_d_right = get_points_from_point(point_d, box_width)

        box_center_point = centerPoint
        box_center_point_left,box_center_point_right = get_points_from_point(box_center_point, box_width)
        #让八个顶点旋转
        vertex_points = [point_a_left, point_a_right,
                        point_b_left, point_b_right,
                        point_c_left, point_c_right,
                        point_d_left, point_d_right]

        point_a_left, point_a_right,point_b_left, point_b_right,point_c_left, point_c_right,point_d_left, point_d_right = structureRotateLineByAngle(vertex_points,
                                                box_center_point_left,
                                                box_center_point_right,
                                                sign_bit * angle)
        return [point_a_left, point_a_right,point_b_left, point_b_right,point_c_left, point_c_right,point_d_left, point_d_right]
                                    
    def generateWholeMask(self,CT_path):
        #CT_path 是需要预测mask并恢复的nii
        
        output_paths = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/wholeMaskRecover'
        
        # 第二步骤预测

        heads = {'hm': self.output_channel}
        self.points_num = 1
        self.down_size = 2
        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=False,
                                         down_ratio=2,
                                         final_kernel=1,
                                         head_conv=256)
        save_path = '/home/gpu/Spinal-Nailing/weights_spinal'
        # 不启用 Batch Normalization 和 Dropout。
        self.model.eval()
        #将每个CT切成5块
            

        itk_img = sitk.ReadImage(CT_path)
        # msk_img = sitk.ReadImage(mask_path)
        # msk_name = os.path.basename(mask_path)
        source_name = os.path.basename(CT_path)
        # print(itk_img.GetSize())
        print(source_name)
        image_array = sitk.GetArrayFromImage(itk_img)
        # msk_array = sitk.GetArrayFromImage(msk_img)
        # print(image_array.shape)

        img = sitk.GetArrayFromImage(itk_img)
        origin_size = img.shape #腰椎区域的原始大小
        mask_recover_groundtruth_array = np.zeros_like(img,dtype=np.float32)
        mask_recover_pre_array = np.zeros_like(img,dtype=np.float32)

        image_array_section = image_array.copy()
        #print(image_array_section.shape)
        output_image,_ = transform.Resize.resize(img = image_array_section.copy(),pts = np.asarray([[1,1,1]]),input_s=240, input_h=400, input_w=400)
        temp = 2048
        output_image = output_image / temp
        output_image = np.clip(output_image, -1, 1)
        output_image = np.asarray(output_image, np.float32)
        intense_image = output_image.copy()
        intense_image = np.reshape(intense_image, (1, 1,240, 400, 400))

        max_pool_down_size = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        intense_image = torch.from_numpy(intense_image)
        intense_image = max_pool_down_size(intense_image)

        landmark_detection_input = intense_image



        print("************Second step fine predict*************")
        steps = ['landmark_detection/sagittal.pth','landmark_detection/pedical.pth']
        sagittal_landmarks = []
        pedical_landmarks = []
        temp = []
        # 用以填充旋转之后的空隙
        fill_pixel = -1024
        #预测特征点
        for step in steps:
            self.model = self.load_model(self.model,os.path.join(save_path, step))
            self.model = self.model.to(self.device)
            # 不启用 Batch Normalization 和 Dropout。
            self.model.eval()
            landmark_detection_input = landmark_detection_input.to(self.device)

            landmark_detection_predict = self.model(landmark_detection_input)
            landmark_detection_hm = landmark_detection_predict['hm']

            reg = 0
            self.decoder = decoder.DecDecoder(K=self.points_num, conf_thresh=0.2)

            landmark_detection_hm = landmark_detection_hm[0]
            pts_last = []
            for i in range(self.output_channel):
                pts2 = self.decoder.ctdet_decode(landmark_detection_hm[i].reshape((1, 1, 60,100,100)), reg,
                                                True, down_ratio=self.down_ratio, down_size=self.down_size)
                pts0 = pts2.copy()
                pts0[:self.points_num, :3] *= (self.down_ratio * self.down_size)
                pts_now = pts0[:self.points_num, :3].tolist()
                pts_last.extend(pts_now)

            #pts_last is the coards of the landmarks in size of [240,400,400]

            pts_last = np.asarray(pts_last, 'float32')
            pts_last = np.asarray(np.round(pts_last), 'int32')
            print('pts_last(相对坐标): ', pts_last.tolist())
            # resize the coards into the size of the origin size 将坐标还原为在原始CT中的坐标
            img_size = [240, 400, 400]
            pts_predict = pts_last.copy()

            pts_predict[:, 0] = pts_predict[:, 0] / img_size[0] * origin_size[0]
            pts_predict[:, 1] = pts_predict[:, 1] / img_size[1] * origin_size[1]
            pts_predict[:, 2] = pts_predict[:, 2] / img_size[2] * origin_size[2]

            print('pts_predict(在原始CT中的坐标): ', pts_predict.tolist())
            print('origin size: ',image_array_section.shape)
            temp.append(pts_predict)
        sagittal_landmarks = temp[0]
        pedical_landmarks = temp[1]
        for i in range(5):
            single_lumbar_mask_groundtruth = sitk.ReadImage(os.path.join('/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/wholeMaskRecover/lumbarSections','sub-verse621_L'+str(5-i)+'_seg-vert_msk.nii.gz'))
            single_lumbar_mask_pre = sitk.ReadImage(os.path.join('/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/wholeMaskRecover/lumbarSections','sub-verse621_L'+str(5-i)+'_ALL_msk.nii.gz'))
            single_lumbar_mask_pre = transform.resize_image_itk(single_lumbar_mask_pre, newSize=single_lumbar_mask_groundtruth.GetSize(), resamplemethod=sitk.sitkLinear)

            single_lumbar_mask_groundtruth = sitk.GetArrayFromImage(single_lumbar_mask_groundtruth)
            single_lumbar_mask_pre = sitk.GetArrayFromImage(single_lumbar_mask_pre)
            
            sagittal_tp = sagittal_landmarks[i*4:i*4+4].tolist()
            sagittal_tp = sorted(sagittal_tp,key=lambda x: x[0]) #根据z轴从小到大排序
            z_distance = sagittal_tp[-1][0] - sagittal_tp[0][0]
            pedical_tp = pedical_landmarks[i*4:i*4+4].tolist()
            pedicalCenterPoints = np.mean(pedical_tp, axis=0)
        
            sign_bit = 1
            if i >= 2:
                sign_bit = -1
            spacing = itk_img.GetSpacing()  # 获取体素的大小
            origin = itk_img.GetOrigin()  # 获取CT的起点位置
            size = itk_img.GetSize()  # 获取CT的大小
            direction = itk_img.GetDirection()  # 获取CT的方向
            #求CT中心点(x,y,z)
            dim = 3
            size_half = [(size[i] - 1) * 0.5 for i in range(dim)]
            CTcenterPoint =  np.asarray(np.array(origin) + np.matmul(np.matmul(np.array(direction).reshape([dim, dim]), np.diag(spacing)), np.array(size_half)),dtype=np.int32)

            # 求得椎体倾斜角度
            angle_points = []
            angle_points.append(sagittal_tp[-1])
            angle_points.append(sagittal_tp[-2])
            angle_points.append(np.mean(pedical_tp[0:2], axis=0))
            angle_points.append(np.mean(pedical_tp[2:], axis=0))
            angle_points = np.asarray(angle_points)
            ttp = angle_points[:,0].copy()
            angle_points[:, 0] = angle_points[:, 2]
            angle_points[:, 2] = ttp
            angle = self.plain_angle(angle_points) * sign_bit

            itk_information = {}
            itk_information['spacing'] = spacing
            itk_information['origin'] = origin
            itk_information['size'] = size
            itk_information['direction'] = direction
            dim = 3
            transformation_list = []
            transformation_list.append(transform.InputCenterToOrigin.get(dim=3, itk_information=itk_information))
            transformation_list.extend([transform.RotationSpine.get(dim=3, angle=angle),
                                        transform.OriginToOutputCenter.get(dim=3, itk_information=itk_information)])
            transformation_comp = sitk.CompositeTransform(dim)
            for transformation in transformation_list:
                transformation_comp.AddTransform(transformation)
            sitk_transformation = transformation_comp

            mask_recover_pre_section = resample(sitk.GetImageFromArray(mask_recover_pre_array),
                                    sitk_transformation,
                                    output_size=size,
                                    output_spacing=spacing,
                                    interpolator=sitk.sitkLinear,
                                    output_pixel_type=None,
                                    default_pixel_value=0)

            mask_recover_pre_array = sitk.GetArrayFromImage(mask_recover_pre_section)

            mask_recover_groundtruth_section = resample(sitk.GetImageFromArray(mask_recover_groundtruth_array),
                                    sitk_transformation,
                                    output_size=size,
                                    output_spacing=spacing,
                                    interpolator=sitk.sitkLinear,
                                    output_pixel_type=None,
                                    default_pixel_value=0)

            mask_recover_groundtruth_array = sitk.GetArrayFromImage(mask_recover_groundtruth_section)
            # plt.imshow(spine_tp_section_array[:,:,spine_tp_section_array.shape[2]//2],cmap='gray')
            # plt.show()
            

            #绕直线旋转坐标点
            rotatedPointsList = np.asarray(sagittal_tp[-1:-3:-1].copy())
            ttp = rotatedPointsList[:, 0].copy()
            rotatedPointsList[:, 0] = rotatedPointsList[:, 2]
            rotatedPointsList[:, 2] = ttp
            # pedicalCenterPointsY = pedicalCenterPoints[1] if i!=0 else pedicalCenterPoints[1]+10
            #得到旋转之后的点
            rotatedPointsList = self.rotate_points_by_line(rotatedPointsList,z_distance,CTcenterPoint,pedicalCenterPoints[1],angle=angle)
            z_extent = 15 if i==0 else 5
            section_min_z = int(sorted(rotatedPointsList,key=lambda x: x[2])[0][2]) if int(sorted(rotatedPointsList,key=lambda x: x[2])[0][2])>=0 else 0
            section_max_z = int(sorted(rotatedPointsList,key=lambda x: x[2])[-1][2])+z_extent if int(sorted(rotatedPointsList,key=lambda x: x[2])[-1][2])+z_extent< image_array.shape[0] else image_array.shape[0]
            section_min_y = int(sorted(rotatedPointsList,key=lambda x: x[1])[0][1]) if int(sorted(rotatedPointsList,key=lambda x: x[1])[0][1])>=0 else 0
            section_max_y = int(sorted(rotatedPointsList,key=lambda x: x[1])[-1][1]) if int(sorted(rotatedPointsList,key=lambda x: x[1])[-1][1])< image_array.shape[1] else image_array.shape[1]
            section_min_x = int(sorted(rotatedPointsList,key=lambda x: x[0])[0][0]) if int(sorted(rotatedPointsList,key=lambda x: x[0])[0][0])>=0 else 0
            section_max_x = int(sorted(rotatedPointsList,key=lambda x: x[0])[-1][0]) if int(sorted(rotatedPointsList,key=lambda x: x[0])[-1][0])< image_array.shape[2] else image_array.shape[2]
            
            mask_recover_groundtruth_array[section_min_z:section_max_z,section_min_y:section_max_y,section_min_x:section_max_x] = single_lumbar_mask_groundtruth
            mask_recover_pre_array[section_min_z:section_max_z,section_min_y:section_max_y,section_min_x:section_max_x] = single_lumbar_mask_pre
            

            transformation_list = []
            transformation_list.append(transform.InputCenterToOrigin.get(dim=3, itk_information=itk_information))
            transformation_list.extend([transform.RotationSpine.get(dim=3, angle=-angle),
                                        transform.OriginToOutputCenter.get(dim=3, itk_information=itk_information)])
            transformation_comp = sitk.CompositeTransform(dim)
            for transformation in transformation_list:
                transformation_comp.AddTransform(transformation)
            sitk_transformation = transformation_comp
            mask_recover_pre_section = resample(sitk.GetImageFromArray(mask_recover_pre_array),
                                    sitk_transformation,
                                    output_size=size,
                                    output_spacing=spacing,
                                    interpolator=sitk.sitkLinear,
                                    output_pixel_type=None,
                                    default_pixel_value=0)

            mask_recover_pre_array = sitk.GetArrayFromImage(mask_recover_pre_section)

            mask_recover_groundtruth_section = resample(sitk.GetImageFromArray(mask_recover_groundtruth_array),
                                    sitk_transformation,
                                    output_size=size,
                                    output_spacing=spacing,
                                    interpolator=sitk.sitkLinear,
                                    output_pixel_type=None,
                                    default_pixel_value=0)

            mask_recover_groundtruth_array = sitk.GetArrayFromImage(mask_recover_groundtruth_section)
            # plt.imshow(seg_section[:,:,seg_section.shape[2]//2],cmap='gray')
            # plt.show()
            # plt.imshow(seg_section[:, seg_section.shape[1] // 2,:], cmap='gray')
            # plt.show()
            # plt.imshow(seg_section[seg_section.shape[0] // 3 * 2, :, :], cmap='gray')
            # plt.show()

            #显示旋转前的原始图像
            # plt.imshow(msk_seg_section[:, :, msk_seg_section.shape[2] // 2], cmap='gray')
            # plt.show()
            # plt.close()


        sitk.WriteImage(sitk.GetImageFromArray(mask_recover_pre_array),os.path.join(output_paths,'mask_recover_pre.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(mask_recover_groundtruth_array), os.path.join(output_paths,'mask_recover_groundtruth.nii.gz'))


if __name__ == '__main__':
    is_object = Network()
    CT_path = '/home/gpu/Spinal-Nailing/VerseCT/2020origin_data/train/wholeMaskRecover/lumbarSections/sub-verse621_ct.nii.gz'
    is_object.generateWholeMask(CT_path = CT_path)

