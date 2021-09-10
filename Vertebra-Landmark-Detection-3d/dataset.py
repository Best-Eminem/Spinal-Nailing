import os

import torch
import torch.utils.data as data
import joblib
from transation_test import *
import numpy as np
import SimpleITK as sitk

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 重新排列64个点的顺序，按此规则每次重排4个点：上左，上右，下左，下右
def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k+4,:]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:,1])
        y_inds_r = np.argsort(pt_r[:,1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
    return np.asarray(boxes, np.float32)


class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, input_s=None, down_ratio=4,downsize = 2,mode = None):
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
        self.downsize = downsize
        self.mode = mode
        if self.phase == 'train':
            self.aug_label = True
        else: self.aug_label = False

    def load_image(self, index):
        path = os.path.join(self.img_dir, self.img_ids[index])
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
        return os.path.join(self.data_dir, 'labels', self.phase, str(index)+'.txt')

    def load_landmarks(self, index):
        img_id = self.img_ids[index]
        #print(img_id)
        landmark_Folder_path = self.get_landmark_path(img_id)
        pts = self.load_gt_pts(landmark_Folder_path)
        #print(img_id)
        return pts

    def preprocess(self,index,points_num,full,mode):
        img_id = self.img_ids[index]
        #print("preprocess img_id: ",img_id)
        img_id_num = np.array(int(img_id[0:-7]),dtype='int32')
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
                                                                     downsize=self.downsize,
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
                                                  downsize=self.downsize,
                                                  down_ratio=self.down_ratio,
                                                  bottom_z = bottom_z,
                                                  )
                # data_dict_series.append(data_dict)
        else:
            data_dict = None
        return data_dict


    def __getitem__(self, index):
        # index记得改回来，不要-1,检查一下这里取到的img_id是正确的吗？
        img_id = self.img_ids[index]
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
            input = input.reshape((1, 1, self.input_s//self.downsize,self.input_h//self.downsize, self.input_w//self.downsize))
            #input_images = torch.from_numpy(input)
            origin_images = data_dict['origin_image'].reshape((1, 1, self.input_s, self.input_h, self.input_w))
            origin_images = torch.from_numpy(origin_images)
            return {'origin_image': origin_images, 'img_id': img_id,'hm':data_dict['hm'],
                    'input':input,
                    #'reg_mask':data_dict['reg_mask'],'ind':data_dict['ind'],'reg':data_dict['reg'],
                    'landmarks':data_dict['landmarks']
                    }
        elif self.phase == 'spine_localisation_eval':
            data_dict = self.preprocess(index=index,points_num=5,full=True,mode=self.mode)
            input = data_dict['input']
            input = input.reshape(
                (1, 1, self.input_s // self.downsize, self.input_h // self.downsize, self.input_w // self.downsize))
            # input_images = torch.from_numpy(input)
            origin_images = data_dict['origin_image'].reshape((1, 1, self.input_s, self.input_h, self.input_w))
            origin_images = torch.from_numpy(origin_images)
            return {'origin_image': origin_images, 'img_id': img_id, 'hm': data_dict['hm'],
                    'input': input,
                    # 'reg_mask':data_dict['reg_mask'],'ind':data_dict['ind'],'reg':data_dict['reg'],
                    'landmarks': data_dict['landmarks'],
                    'origin_size':data_dict['origin_size']
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
                data_dict = self.preprocess(index=index,points_num=5,full=True,mode=self.mode)
            elif self.mode == 'landmark_detection':
                data_dict = self.preprocess(index=index,points_num=40,full=True,mode=self.mode)
            else:
                data_dict = self.preprocess(index=index,points_num=0,full=True,mode=self.mode)
            #print(1111)
            return data_dict

    def __len__(self):
        return len(self.img_ids)


