import joblib
import json
import torch
import numpy as np
import sys
import os

from torch._C import dtype
o_path = os.getcwd()
sys.path.append(o_path)
from draw.draw import draw_points_test, draw_by_matplotlib
#import draw_distribute_points
from models import spinal_net
import SimpleITK as sitk
from utils import decoder
from preprocess import transform
import torch.nn as nn
from dataset.dataset import BaseDataset
import loss
from matplotlib import pyplot as plt

class Network(object):
    def __init__(self, args,generateZNSeg = False):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # heads表示的是最后一层输出的通道数
        heads = {'hm': 5,
                 # 'hm1': args.num_classes,
                 # 'hm2': args.num_classes,
                 # 'hm3': args.num_classes,
                 # 'hm4': args.num_classes,
                 # 'hm5': args.num_classes,
                 # 'reg': 3*args.num_classes,
                 # 'normal_vector': 3 * args.num_classes,
                 #'wh': 3*4,
                 }
        self.generateZNSeg = generateZNSeg
        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=False,
                                         down_ratio=2,
                                         final_kernel=1,
                                         head_conv=256,
                                         #spatial=True
                                         )
        self.num_classes = args.num_classes

        self.dataset = {'spinal': BaseDataset}
        self.criterion = loss.LossAll()
        self.points_num = 1 #K为特征点的最大个数
        self.decoder = decoder.DecDecoder(K=self.points_num, conf_thresh=args.conf_thresh)  # K为特征点的最大个数
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

    def pts_sort(self,pts):
        pts.sort(key=lambda x: (x[0]))
        landmarks = []
        for i in range(5):
            temp = pts[8 * i:8 * (i + 1)]
            temp.sort(key=lambda x: x[2])
            pedical_point = temp[0:2]
            pedical_point.extend(temp[-2:])
            plain_point = temp[2:6]
            plain_point.sort(key=lambda x: x[0])
            bottom_point = plain_point[0:2]
            top_point = plain_point[2:4]
            bottom_point.sort(key=lambda x: x[1])
            top_point.sort(key=lambda x: x[1])
            landmarks.extend(top_point)
            landmarks.extend(bottom_point)
            landmarks.extend(pedical_point)
        return landmarks
    def eval(self, args, save,CT_path,label_path):
        save_path = args.model_dir
        print("************First step coarse predict************")
        self.model = self.load_model(self.model, os.path.join(save_path, 'spine_localisation/model_last_origin.pth'))
        self.model = self.model.to(self.device)
        #不启用 Batch Normalization 和 Dropout。
        self.model.eval()
            #将heatmap还原为坐标
        itk_img = sitk.ReadImage(CT_path)
        # print(itk_img.GetSize())
        image_array = sitk.GetArrayFromImage(itk_img)
        # print(image_array.shape)
        image_array_section = image_array
        # for i in range(image_array_section.shape[0]):
        #     image_array_section[i] = np.flip(image_array_section[i,:,:], axis=0)
        #image_array_section = image_array_section[160:380]
        # image_array_section = image_array_section[:,:,190:350]
        # image_array_section = image_array_section[:,300:420,:]
        # print(image_array_section.shape)
        # plt.imshow(image_array_section[120,:,:], cmap = 'gray')
        # plt.show()
        # plt.imshow(image_array_section[:, :, 100], cmap='gray')
        # plt.show()
        bottom_z = 0
        out_image, _ = transform.Resize.resize(img = image_array_section.copy(),pts = np.asarray([[1,1,1]]),input_s=400, input_h=512, input_w=512)
        temp = 2048
        out_image = out_image / temp
        out_image = np.clip(out_image, -1, 1)
        out_image = np.asarray(out_image, np.float32)
        intense_image = out_image
        intense_image = np.reshape(intense_image, (1, 400, 512, 512))
        #缩小输入大小
        max_pool_down_size = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        intense_image = torch.from_numpy(intense_image)
        intense_image = max_pool_down_size(intense_image)
        intense_image = max_pool_down_size(intense_image)
        intense_image = intense_image.to(device=self.device)

        output = self.model(intense_image.reshape((1,1,100,128,128)))
        hm = output['hm']
        # hm1 = output['hm1']
        # hm2 = output['hm2']
        # hm3 = output['hm3']
        # hm4 = output['hm4']
        # hm5 = output['hm5']
        # hm = [hm1,hm2,hm3,hm4,hm5]
        hm = hm[0]
        pts_predict = []
        reg = 0
        for i in range(5):
            pts2 = self.decoder.ctdet_decode(hm[i].reshape((1, 1, 50, 64, 64)), reg, True,down_ratio=2,down_size=4)
            pts0 = pts2.copy()
            pts0[:self.points_num, :3] *= (2 * 4)
            pts_now = pts0[:self.points_num, :3].tolist()
            pts_predict.extend(pts_now)

        print('totol pts num is {}'.format(len(pts_predict)))
        pts_predict.sort(key = lambda x:(x[0],x[1],x[2]))
        pts_predict = np.asarray(pts_predict,'float32')
        pts_predict = np.asarray(np.round(pts_predict), 'int32')
        print('pts_predict: ', pts_predict.tolist())
        #draw_points_test(out_image.reshape((400,512,512)), pts_predict) #draw 2D image to show the position of the landmarks
        # draw_by_matplotlib(out_image.reshape((400,512,512)), pts_predict)


        img = sitk.GetArrayFromImage(itk_img)
        img_size = [400,512,512]
        origin_size = img.shape
        #将第一步的预测坐标复原到原始CT的位置
        pts_predict[:, 0] = pts_predict[:, 0] / img_size[0] * origin_size[0]
        pts_predict[:, 1] = pts_predict[:, 1] / img_size[1] * origin_size[1]
        pts_predict[:, 2] = pts_predict[:, 2] / img_size[2] * origin_size[2]
        pts_predict = np.asarray(pts_predict)

        image_array = img.copy()


        spine_localisation_eval_pts = pts_predict

        #pts_label = [[187,355,268],[226,345,268],[272,347,268],[315,350,268],[356,355,268]]
        # with open(label_path, 'r') as f:
        # # load json file
        #     json_data = json.load(f)
        #     for dict in json_data:
        #         if 'direction' in dict:
        #             continue
        #         if 'label' in dict:
        #             if dict['label'] == 20 or dict['label'] == 21 or dict['label'] == 22 or dict['label'] == 23 or dict['label'] == 24:
        #                 pts_label.append([dict['Z'],dict['Y'],dict['X']])

        # if len(pts_label) != 5:
        #     print(label_path,"doesn't have a lumbar section!!" )
        #     return None
        # pts_label = np.asarray(pts_label)
        #spine_localisation_eval_pts = pts_label #将5个质心点设置为label的
        spine_localisation_eval_center = np.asarray(np.mean(pts_predict,axis=0),'int32')

        bottom_z = (spine_localisation_eval_pts[0][0] - 30) if spine_localisation_eval_pts[0][0] - 30 >= 0 else 0
        bottom_z = np.floor(bottom_z).astype(np.int32)
        top_z = spine_localisation_eval_pts[4][0] + 30
        top_z = np.ceil(top_z).astype(np.int32)

        # 根据预测的整个ct的中心点的x,y坐标取出原ct的一部分
        pts_center_y = spine_localisation_eval_center[1]
        pts_center_x = spine_localisation_eval_center[2]

        # 截取包含5段脊椎的部分
        image_array_section = image_array[bottom_z:top_z,
                              (pts_center_y - 200) if pts_center_y - 200 >= 0 else 0:(
                                          pts_center_y + 200) if pts_center_y + 200 <= 512 else 512,
                              (pts_center_x - 200) if pts_center_x - 200 >= 0 else 0:(
                                          pts_center_x + 200) if pts_center_x + 200 <= 512 else 512]
        if self.generateZNSeg:
            save_spine_section = sitk.GetImageFromArray(image_array_section)
            sitk.WriteImage(save_spine_section, os.path.join('/home/gpu/Spinal-Nailing/ZN-CT-nii/spine_eval_sections', os.path.basename(CT_path)))
            return
        # image_array_section = image_array #测试Verse CT的时候用
        print(image_array_section.shape)
        output_image,_ = transform.Resize.resize(img = image_array_section.copy(),pts = np.asarray([[1,1,1]]),input_s=240, input_h=400, input_w=400)
        temp = 2048
        output_image = output_image / temp
        output_image = np.clip(output_image, -1, 1)
        output_image = np.asarray(output_image, np.float32)
        #joblib.dump(output_image, 'F:\\ZN-CT-nii\\my_eval\\'+CT)
        intense_image = output_image.copy()
        intense_image = np.reshape(intense_image, (1, 1,240, 400, 400))

        max_pool_down_size = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        intense_image = torch.from_numpy(intense_image)
        intense_image = max_pool_down_size(intense_image)

        landmark_detection_input = intense_image


        #第二步骤预测
        heads = {'hm': self.output_channel}
        self.points_num = 1
        self.down_size = 2
        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=2,
                                         final_kernel=1,
                                         head_conv=256)
        print("************Second step fine predict*************")
        self.model = self.load_model(self.model,os.path.join(save_path, 'landmark_detection/model_last_origin.pth'))
        self.model = self.model.to(self.device)
        # 不启用 Batch Normalization 和 Dropout。
        self.model.eval()
        landmark_detection_input = landmark_detection_input.to(self.device)

        landmark_detection_predict = self.model(landmark_detection_input)
        landmark_detection_hm = landmark_detection_predict['hm']

        reg = 0
        self.decoder = decoder.DecDecoder(K=self.points_num, conf_thresh=args.conf_thresh)

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
        #pts_last.sort(key=lambda x: (x[0]))
        pts_last = np.asarray(pts_last, 'float32')
        pts_last = np.asarray(np.round(pts_last), 'int32')
        print('pts_last(相对坐标): ', pts_last.tolist())
        # resize the coards into the size of the origin size 将坐标还原为在原始CT中的坐标
        img_size = [240, 400, 400]
        pts_origin_last = pts_last.copy()

        pts_origin_last[:, 0] = pts_origin_last[:, 0]/img_size[0] * img.shape[0]

        pts_origin_last[:, 0] = pts_origin_last[:, 0] + bottom_z
        pts_origin_last[:, 1] = pts_origin_last[:, 1] + pts_center_y - 200
        pts_origin_last[:, 2] = pts_origin_last[:, 2] + pts_center_x - 200


        pts_last = pts_last.tolist()

        pts_origin_last = pts_origin_last.tolist()
        # pts_origin_last = self.pts_sort(pts_origin_last)
        print('pts_origin_last(在原始CT中的坐标): ', pts_origin_last)
        #draw_3D = draw_distribute_points.draw(output_image.reshape((240,400,400)),pts_last)
        #draw_points_test(output_image.reshape((240,400,400)), pts_last) #draw 2D image to show the position of the landmarks
        draw_by_matplotlib(output_image.reshape((240,400,400)), pts_last)

        # img = np.transpose(img, (2, 1, 0))
        # img = img/2048
        # print("ok")

        # img[img < 0.05] = -1
        # img[img >= 0.1] = 1
        # pts_origin_last = np.asarray(pts_origin_last)
        # temp = pts_origin_last[:, 0].copy()
        # pts_origin_last[:, 0] = pts_origin_last[:, 2]
        # pts_origin_last[:, 2] = temp
        #draw_distribute_points.draw_plane(output_image.reshape((240,400,400)),pts_last) #画片段CT置钉示意图
       # draw_distribute_points.draw_plane(img,pts_origin_last) #画原始CT置钉示意图





