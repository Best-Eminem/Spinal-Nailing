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
        heads = {#'hm': 5,
                 'hm1': args.num_classes,
                 'hm2': args.num_classes,
                 'hm3': args.num_classes,
                 'hm4': args.num_classes,
                 'hm5': args.num_classes,
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

    def eval(self, args, save,CT):
        save_path = 'weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, 'spine_localisation//five output//model_150.pth'))
        self.model = self.model.to(self.device)
        #不启用 Batch Normalization 和 Dropout。
        self.model.eval()
            #将heatmap还原为坐标
        itk_img = sitk.ReadImage(os.path.join("E:\\ZN-CT-nii\\data\\gt", CT))
        image_array = sitk.GetArrayFromImage(itk_img)
        data_aug = {'eval': transform.Compose([transform.ConvertImgFloat(),  # 转为float32格式
                                                # transform.PhotometricDistort(), #对比度，噪声，亮度
                                                # transform.Expand(max_scale=1.5, mean=(0, 0, 0)),
                                                # transform.RandomMirror_w(),
                                                transform.Resize(s=400, h=512, w=512)  # Resize了点坐标和img
                                                ])
                    }
        image_array_section = image_array
        bottom_z = 0
        out_image, _ = data_aug['eval'](image_array_section.copy(), np.asarray([[1,1,1]]))
        temp = 2048

        # out_image = out_image / np.max(abs(out_image))  某些值过大，导致椎体区域归一化值很小
        out_image = out_image / temp
        out_image = np.asarray(out_image, np.float32)
        intense_image = out_image.copy()
        # intense_image[intense_image >= 0.1] += 0.2
        intense_image[intense_image > 1] = 1
        intense_image = np.reshape(intense_image, (1, 400, 512, 512))
        out_image = np.reshape(out_image, (1, 400, 512, 512))

        max_pool_downsize = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        # hm = max_pool_downsize(hm).numpy()
        # hm = hm.reshape(hm.shape[2:])
        # 增强hm中的landmark的像素值
        # hm[hm>=1] +=1

        # image = torch.from_numpy(image)
        # image = max_pool_downsize(image).numpy()
        intense_image = torch.from_numpy(intense_image)
        intense_image = max_pool_downsize(intense_image)
        intense_image = max_pool_downsize(intense_image)
        intense_image = intense_image.to(device=self.device)

        output = self.model(intense_image.reshape((1,1,100,128,128)))
        hm1 = output['hm1']
        hm2 = output['hm2']
        hm3 = output['hm3']
        hm4 = output['hm4']
        hm5 = output['hm5']
        hm = [hm1,hm2,hm3,hm4,hm5]
        #hm = hm[0]
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
        origin_size = [400,512,512]
        img_size = img.shape
        #将第一步的预测坐标复原到原始CT的位置
        pts_predict[:, 0] = pts_predict[:, 0] / origin_size[0] * img_size[0]
        pts_predict[:, 1] = pts_predict[:, 1] / origin_size[1] * img_size[1]
        pts_predict[:, 2] = pts_predict[:, 2] / origin_size[2] * img_size[2]
        pts_predict = np.asarray(pts_predict)

        image_array = img.copy()

        data_aug = {'test': transform.Compose([transform.ConvertImgFloat(),  # 转为float32格式
                                                transform.Resize(s=240, h=400, w=400)  # Resize了点坐标和img
                                                ]),} # resize
        spine_localisation_eval_pts = pts_predict
        spine_localisation_eval_center = np.asarray(np.mean(pts_predict,axis=0),'int32')

        bottom_z = (spine_localisation_eval_pts[0][0] - 25) if spine_localisation_eval_pts[0][0] - 25 >= 0 else 0
        bottom_z = np.floor(bottom_z).astype(np.int32)
        top_z = spine_localisation_eval_pts[4][0] + 25
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
        out_image, _ = data_aug['test'](image_array_section.copy(), pts_predict)

        min = np.min(out_image)
        max = np.max(out_image)
        temp = 2048
        # out_image = np.clip(out_image, a_min=0., a_max=255.)
        # 不需要调换顺序
        # out_image = out_image / np.max(abs(out_image))
        out_image = out_image / temp
        out_image = np.asarray(out_image, np.float32)
        joblib.dump(out_image, 'E:\\ZN-CT-nii\\my_eval\\'+CT)
        intense_image = out_image.copy()
        intense_image[intense_image >= 0.1] += 0.2
        intense_image[intense_image > 1] = 1
        intense_image = np.reshape(intense_image, (1, 1,240, 400, 400))

        max_pool_downsize = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        intense_image = torch.from_numpy(intense_image)
        intense_image = max_pool_downsize(intense_image)

        landmark_detection_input = intense_image


        #第二步骤预测
        heads = {'hm': 1}

        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=2,
                                         final_kernel=1,
                                         head_conv=256)
        self.model = self.load_model(self.model,os.path.join(save_path, 'landmark_detection.pth'))
        self.model = self.model.to(self.device)
        # 不启用 Batch Normalization 和 Dropout。
        self.model.eval()
        landmark_detection_input = landmark_detection_input.to(self.device)

        landmark_detection_predict = self.model(landmark_detection_input)
        landmark_detection_hm = landmark_detection_predict['hm']

        reg = 0
        self.decoder = decoder.DecDecoder(K=30, conf_thresh=args.conf_thresh)
        pts2 = self.decoder.ctdet_decode(landmark_detection_hm, reg, True, down_ratio=2, downsize=2)  # 17, 11
        pts0 = pts2.copy()

        pts0[:30, :3] *= (2 * 2)
        pts_last = pts0[:30, :3].tolist()
        pts_last.sort(key=lambda x: (x[0], x[1], x[2]))
        pts_last = np.asarray(pts_last, 'float32')
        pts_last = np.asarray(np.round(pts_last), 'int32')
        print('pts_last: ', pts_last.tolist())
        pts_last = pts_last.tolist()
        draw_points_test(out_image.reshape((1,1,240,400,400)), pts_last)




