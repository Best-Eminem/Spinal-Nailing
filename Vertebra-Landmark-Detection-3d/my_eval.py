import joblib
import torch
import numpy as np

from draw import draw_points_test
from models import spinal_net
import SimpleITK as sitk
import decoder
import transform
import os
import torch.nn as nn
from dataset import BaseDataset
import loss
from matplotlib import pyplot as plt

class Network(object):
    def __init__(self, args):
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

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=2,
                                         final_kernel=1,
                                         head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=1, conf_thresh=args.conf_thresh) #K为特征点的最大个数
        self.dataset = {'spinal': BaseDataset}
        self.criterion = loss.LossAll()
        self.points_num = 1 #K为特征点的最大个数
        self.down_ratio = 2
        self.downsize = 4
        self.mode = "spine_localisation"

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def eval(self, args, save,CT_path):
        save_path = args.model_dir
        self.model = self.load_model(self.model, os.path.join(save_path, 'spine_localisation/one output/clean/model_200.pth'))
        self.model = self.model.to(self.device)
        #不启用 Batch Normalization 和 Dropout。
        self.model.eval()
            #将heatmap还原为坐标
        itk_img = sitk.ReadImage(CT_path)
        image_array = sitk.GetArrayFromImage(itk_img)
        image_array_section = image_array
        bottom_z = 0
        out_image, _ = transform.Resize.resize(img = image_array_section.copy(),pts = np.asarray([[1,1,1]]),input_s=400, input_h=512, input_w=512)
        temp = 2048
        out_image = out_image / temp
        out_image = np.asarray(out_image, np.float32)
        intense_image = out_image
        intense_image = np.reshape(intense_image, (1, 400, 512, 512))
        #缩小输入大小
        max_pool_downsize = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        intense_image = torch.from_numpy(intense_image)
        intense_image = max_pool_downsize(intense_image)
        intense_image = max_pool_downsize(intense_image)
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
            pts2 = self.decoder.ctdet_decode(hm[i].reshape((1, 1, 50, 64, 64)), reg, True,down_ratio=2,downsize=4)
            pts0 = pts2.copy()
            pts0[:self.points_num, :3] *= (2 * 4)
            pts_now = pts0[:self.points_num, :3].tolist()[0]
            pts_predict.append(pts_now)

        print('totol pts num is {}'.format(len(pts_predict)))
        pts_predict.sort(key = lambda x:(x[0],x[1],x[2]))
        pts_predict = np.asarray(pts_predict,'float32')
        pts_predict = np.asarray(np.round(pts_predict), 'int32')
        print('pts_predict: ', pts_predict.tolist())
        #pts_predict = pts_predict.tolist()

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
        output_image,_ = transform.Resize.resize(img = image_array_section,pts = np.asarray([[1,1,1]]),input_s=240, input_h=400, input_w=400)
        temp = 2048
        output_image = output_image / temp
        output_image = np.asarray(output_image, np.float32)
        #joblib.dump(output_image, 'F:\\ZN-CT-nii\\my_eval\\'+CT)
        intense_image = output_image.copy()
        intense_image = np.reshape(intense_image, (1, 1,240, 400, 400))

        max_pool_downsize = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        intense_image = torch.from_numpy(intense_image)
        intense_image = max_pool_downsize(intense_image)

        landmark_detection_input = intense_image


        #第二步骤预测
        heads = {'hm': 5}
        self.points_num = 8
        self.downsize = 2
        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=2,
                                         final_kernel=1,
                                         head_conv=256)
        self.model = self.load_model(self.model,os.path.join(save_path, 'landmark_detection/clean/model_200.pth'))
        self.model = self.model.to(self.device)
        # 不启用 Batch Normalization 和 Dropout。
        self.model.eval()
        landmark_detection_input = landmark_detection_input.to(self.device)

        landmark_detection_predict = self.model(landmark_detection_input)
        landmark_detection_hm = landmark_detection_predict['hm']

        reg = 0
        self.decoder = decoder.DecDecoder(K=8, conf_thresh=args.conf_thresh)

        landmark_detection_hm = landmark_detection_hm[0]
        pts_last = []
        for i in range(5):
            pts2 = self.decoder.ctdet_decode(landmark_detection_hm[i].reshape((1, 1, 60,100,100)), reg,
                                             True, down_ratio=self.down_ratio, downsize=self.downsize)
            pts0 = pts2.copy()
            pts0[:self.points_num, :3] *= (self.down_ratio * self.downsize)
            pts_now = pts0[:self.points_num, :3].tolist()
            pts_last.extend(pts_now)

        pts_last.sort(key=lambda x: (x[0], x[1], x[2]))
        pts_last = np.asarray(pts_last, 'float32')
        pts_last = np.asarray(np.round(pts_last), 'int32')
        print('pts_last: ', pts_last.tolist())
        pts_last = pts_last.tolist()
        draw_points_test(output_image.reshape((240,400,400)), pts_last)




