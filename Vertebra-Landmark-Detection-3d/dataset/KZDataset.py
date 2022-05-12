import os

import torch
import torch.utils.data as data
import random
import joblib
from preprocess.transation_test import *
import numpy as np
import SimpleITK as sitk

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class KZDataset(data.Dataset):
    def __init__(self, data_dir, phase, ki=0, K=5,input_h=None, input_w=None, input_s=None, down_ratio=4,down_size = 2,mode = None,sigmas = None):
        '''
        ki：当前是第几折, 从0开始，范围为[0, K)
        K：总的折数
        '''
        super(KZDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.input_s = input_s
        self.down_ratio = down_ratio
        self.class_name = ['__background__', 'cell']
        # self.num_classes = 40 #原始表示68个特征点
        #self.img_dir = os.path.join(data_dir, 'data', self.phase)
        self.img_dir = os.path.join(data_dir, 'data', 'k_fold_smooth')
        self.img_ids = sorted(os.listdir(self.img_dir))
        self.down_size = down_size
        self.mode = mode
        self.sigmas = sigmas

        if 1:
            #mix the data
            random.seed(ki + 5)
            random.shuffle(self.img_ids)
        leng = len(self.img_ids)
        every_z_len = leng // K
        if phase == 'val':
            self.data_info = self.img_ids[every_z_len * ki: every_z_len * (ki + 1)]
        elif phase == 'train':
            self.data_info = self.img_ids[: every_z_len * ki] + self.img_ids[every_z_len * (ki + 1):]
        else:
            self.data_info = []
            if self.mode == 'landmark_detection':
                f = open(r'/home/gpu/Spinal-Nailing/weights_spinal/landmark_detection/pedicle_points_k_fold_'+str(ki)+'/val_set.txt','r')
            else:
                f = open(r'/home/gpu/Spinal-Nailing/weights_spinal/spine_localisation/fold_'+str(ki)+'/val_set.txt','r')
            data = list(f)[0]
            f.close
            data = data.replace('[','');data = data.replace(']','')
            data = data.replace('\'','');data = data.replace(' ','')
            data = data.replace('\n','')
            data_list = data.split(',')
            self.data_info = data_list
            

        if self.phase == 'train':
            self.aug_label = True
        else: self.aug_label = False

    def load_image(self, index):
        path = os.path.join(self.img_dir, self.data_info[index])
        itk_img = sitk.ReadImage(path)
        # image = sitk.GetArrayFromImage(itk_img)
        # image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        return itk_img

    def load_gt_pts(self, landmark_path):
        # 取出 txt文件中的三位坐标点信息
        pts = []
        #print(landmark_path)
        with open(landmark_path, "r") as f:
            i = 1
            for line in f.readlines():
                line = line.strip('\n')  # 去掉列表中每一个元素的换行符
                if line != '' and i >= 13 and i<=52:
                    _, __, x, y, z = line.split()
                    pts.append((x, y, z))
                #
                i += 1
        #print(landmark_path)
        # pts = rearrange_pts(pts)
        return pts

    def get_landmark_path(self, img_id):
        # eg.  E://ZN-CT-nii//labels//train//xxxx.txt
        index,_,_ = img_id.split('.')
        return os.path.join(self.data_dir, 'labels', 'k_fold', str(index)+'.txt')

    def load_landmarks(self, index):
        img_id = self.data_info[index]
        #print(img_id)
        landmark_Folder_path = self.get_landmark_path(img_id)
        pts = self.load_gt_pts(landmark_Folder_path)
        #print(img_id)
        return pts

    def preprocess(self,index,points_num,full,mode):
        img_id = self.data_info[index]
        #print("preprocess img_id: ",img_id)
        img_id_num = np.array(int(img_id[0:img_id.find('.')]),dtype='int32')
        image = self.load_image(index)  # image是 itk image格式，不是np格式
        aug_label = self.aug_label
        # 返回shape为(35，3)的labels列表,排列方式为(z,y,x)
        pts = self.load_landmarks(index)  # x,y,z
        # data_series = \
        if mode == 'spine_localisation':
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
            #data_dict_series = []
            for out_image, pts_2, img_id_num,origin_size in data_series:
                data_dict = spine_localisation_generate_ground_truth(output_img=out_image,
                                                                     points_num=points_num,
                                                                     pts_2=pts_2,
                                                                     image_s=self.input_s // self.down_ratio,
                                                                     image_h=self.input_h // self.down_ratio,
                                                                     image_w=self.input_w // self.down_ratio,
                                                                     img_id=img_id_num,
                                                                     full=full,
                                                                     down_size=self.down_size,
                                                                     down_ratio=self.down_ratio,
                                                                     origin_size=origin_size)
                #data_dict_series.append(data_dict)


        elif mode == 'landmark_detection':
            eval_store_path = 'eval/spine_localisation_eval/' + img_id[0:-7] + '.eval'

            spine_localisation_eval_dict = joblib.load(os.path.join(self.data_dir, eval_store_path))
            data_series = processing_train(image=image,
                                           pts=pts,
                                           points_num=points_num,
                                           image_s=self.input_s,  # 400
                                           image_h=self.input_h,  # 512
                                           image_w=self.input_w,  # 512
                                           aug_label=aug_label,
                                           img_id=img_id_num,
                                           full=full,
                                           spine_localisation_eval_dict= spine_localisation_eval_dict
                                           )
            # data_dict_series = []
            for out_image, pts_2, img_id_num,bottom_z in data_series:
                data_dict = generate_ground_truth(output_img=out_image,
                                                  points_num=points_num,
                                                  pts_2=pts_2,
                                                  image_s=self.input_s // self.down_ratio,
                                                  image_h=self.input_h // self.down_ratio,
                                                  image_w=self.input_w // self.down_ratio,
                                                  img_id=img_id_num,
                                                  full=full,
                                                  down_size=self.down_size,
                                                  down_ratio=self.down_ratio,
                                                  bottom_z = bottom_z,
                                                  )
                # data_dict_series.append(data_dict)
        else:
            data_dict = None
        return data_dict


    def __getitem__(self, index):
        # index记得改回来，不要-1,检查一下这里取到的img_id是正确的吗？
        img_id = self.data_info[index]
        # img_num = img_id.split('.')[0]
        if self.phase == 'test':
            # images 为经过预处理后的tensor
            if self.mode == 'spine_localisation':
                data_dict = self.preprocess(index=index,points_num=5,full=True,mode=self.mode)
            elif self.mode == 'landmark_detection':
                data_dict = self.preprocess(index=index,points_num=40,full=True,mode=self.mode)
            else:
                data_dict = self.preprocess(index=index,points_num=0,full=True,mode=self.mode)
            #images = pre_proc.processing_test(image=self.load_image(index), input_h=self.input_h, input_w=self.input_w, input_s=self.input_s)
            input = data_dict['input']
            input = input.reshape((1, 1, self.input_s//self.down_size,self.input_h//self.down_size, self.input_w//self.down_size))
            #input_images = torch.from_numpy(input)
            origin_images = data_dict['origin_image'].reshape((1, 1, self.input_s, self.input_h, self.input_w))
            origin_images = torch.from_numpy(origin_images)
            return {'origin_image': origin_images, 'img_id': img_id,#'hm':data_dict['hm'],
                    'input':input,
                    #'reg_mask':data_dict['reg_mask'],'ind':data_dict['ind'],'reg':data_dict['reg'],
                    'landmarks':data_dict['landmarks']
                    }
        elif self.phase =='train':
            if self.mode == 'spine_localisation':
                data_dict = self.preprocess(index=index,points_num=5,full=True,mode=self.mode)
            elif self.mode == 'landmark_detection':
                data_dict = self.preprocess(index=index,points_num=40,full=True,mode=self.mode)
            else:
                data_dict = self.preprocess(index=index,points_num=0,full=True,mode=self.mode)
            return data_dict
        else:
            if self.mode == 'spine_localisation':
                data_dict = self.preprocess(index=index, points_num=5, full=True, mode=self.mode)
            elif self.mode == 'landmark_detection':
                data_dict = self.preprocess(index=index, points_num=40, full=True, mode=self.mode)
            else:
                data_dict = self.preprocess(index=index, points_num=0, full=True, mode=self.mode)
            return data_dict

    def __len__(self):
        return len(self.data_info)


