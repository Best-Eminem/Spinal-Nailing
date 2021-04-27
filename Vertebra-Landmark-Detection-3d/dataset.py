import os

import torch
import torch.utils.data as data
import pre_proc
import joblib
import cv2
from scipy.io import loadmat
import numpy as np
import SimpleITK as sitk
from xyz2irc_irc2xyz import xyz2irc
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
    def __init__(self, data_dir, phase, input_h=None, input_w=None, input_s=None, down_ratio=4):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.input_s = input_s
        self.down_ratio = down_ratio
        self.class_name = ['__background__', 'cell']
        self.num_classes = 35 #原始表示68个特征点
        self.img_dir = os.path.join(data_dir, 'data', self.phase)
        self.img_ids = sorted(os.listdir(self.img_dir))

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
        # eg.  E://ZN-CT-nii//labels//train//xxxx.txt
        index,_,_ = img_id.split('.')
        return os.path.join(self.data_dir, 'labels', self.phase, str(index)+'.txt')

    def load_landmarks(self, index):
        img_id = self.img_ids[index]
        landmark_Folder_path = self.get_landmark_path(img_id)
        pts = self.load_gt_pts(landmark_Folder_path)
        return pts

    def origin_getitem(self,index,points_num,full):
        #此函数用来生成groundtruth
        img_id = self.img_ids[index]
        image = self.load_image(index)  # image是 itk image格式，不是np格式
        if self.phase == 'gt':
            aug_label = False
            if self.phase == 'gt':
                aug_label = True
            # 返回shape为(35，3)的labels列表,排列方式为(z,y,x)
            pts = self.load_landmarks(index)  # x,y,z
            # 欧式坐标转化为体素坐标
            # pts_irc = []  # ['index', 'row', 'col']
            # for i in range(len(pts)):
            #     pts_irc.append(xyz2irc(image, pts[i]))
            # out_image为numpy格式，pts_2为欧式坐标（x,y,z）
            data_series = pre_proc.processing_train(image=image,
                                                         pts=pts,
                                                         points_num = points_num,
                                                         image_s=self.input_s,  # 240 or 36
                                                         image_h=self.input_h,  # 512
                                                         image_w=self.input_w,  # 512
                                                         down_ratio=self.down_ratio,
                                                         aug_label=aug_label,
                                                         img_id=img_id,
                                                         full = full
                                                         )
            data_dict_series = []
            for out_image,intense_image,pts_2 in data_series:
                data_dict = pre_proc.generate_ground_truth(image=out_image,
                                                           intense_image= intense_image,
                                                           points_num=points_num,
                                                           pts_2=pts_2,
                                                           image_s=self.input_s // self.down_ratio,
                                                           image_h=self.input_h // self.down_ratio,
                                                           image_w=self.input_w // self.down_ratio,
                                                           img_id=img_id,
                                                           full = full)
                data_dict_series.append(data_dict)

            return data_dict_series

    def __getitem__(self, index):
        # index记得改回来，不要-1,检查一下这里取到的img_id是正确的吗？
        img_id = self.img_ids[index]
        # img_num = img_id.split('.')[0]
        if self.phase == 'test':
            # images 为经过预处理后的tensor
            data_dict = joblib.load('E:\\ZN-CT-nii\\groundtruth\\' + img_id)
            #images = pre_proc.processing_test(image=self.load_image(index), input_h=self.input_h, input_w=self.input_w, input_s=self.input_s)
            images = data_dict['input'].reshape((1, 1, self.input_s,self.input_h, self.input_w))
            images = torch.from_numpy(images)
            return {'images': images, 'img_id': img_id,'hm':data_dict['hm'],
                    'reg_mask':data_dict['reg_mask'],'ind':data_dict['ind'],'reg':data_dict['reg'],
                    'landmarks':data_dict['landmarks']}
        else:
            data_dict = joblib.load('E:\\ZN-CT-nii\\groundtruth\\'+ img_id)
            return data_dict

    def __len__(self):
        return len(self.img_ids)


