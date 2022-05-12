import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
from torch._C import dtype
from dataset.KZDataset import KZDataset
import torch
import numpy as np
from models import spinal_net
from models.SCNet import unet3d,unet3d_spatial
import SimpleITK as sitk
from utils import decoder
from dataset.dataset import BaseDataset
from draw import draw_points
from draw import draw
import loss
from matplotlib import pyplot as plt
class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # heads表示的是最后一层输出的通道数
        self.KZdataset = args.KZdataset
        self.output_channel = args.output_channel
        self.heatmap_sigmas = torch.nn.Parameter(torch.FloatTensor(self.output_channel * [12]))
        heads = {'hm': args.num_classes * self.output_channel,

                 # 若第一步输出5个hm的话，使用下面这一部分
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
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256,
                                         spatial=False
                                         )
        #self.model = unet3d_spatial(120,200,200)
        # self.model = unet3d()
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh) #K为特征点的最大个数
        self.dataset = {'spinal': BaseDataset} if not self.KZdataset else {'spinal': KZDataset}
        #self.dataset = {'spinal': KZDataset}
        self.criterion = loss.LossAll()
        self.points_num = args.K #K为特征点的最大个数
        self.down_ratio = args.down_ratio
        self.down_size = args.down_size
        self.mode = args.mode

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model
    def pts_sort(self,pts,mode):
        pts.sort(key=lambda x: (x[0]))
        landmarks = []
        # 排序矢状面4个点
        # for i in range(10):
        #     temp = pts[2 * i:2 *(i+1)]
        #     temp.sort(key=lambda x: x[1])
        #     landmarks.extend(temp)
        # # 排序椎弓根4个点
        # for i in range(5):
        #     temp = pts[4 * i:4 *(i+1)]
        #     temp.sort(key=lambda x: x[2])
        #     landmarks.extend(temp)
        #排序8个点
        if mode == 'landmark_detection':
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
        else:
            return landmarks
        #return pts
        #return landmarks
    def test(self, args, save):
        save_path = args.model_dir
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        #不启用 Batch Normalization 和 Dropout。
        self.model.train()

        #就是dataset_module = BaseDataset
        dataset_module = self.dataset[args.dataset]
        if not self.KZdataset:
            dsets = dataset_module(data_dir=args.data_dir,
                                       phase='test',
                                       input_h=args.input_h,
                                       input_w=args.input_w,
                                       input_s=args.input_s,
                                       down_ratio=args.down_ratio,
                                       down_size=args.down_size,
                                       mode=self.mode,
                                       sigmas=None
                                       )
        else:
            dsets = dataset_module(data_dir=args.data_dir,
                                        phase='test',
                                        ki=args.ki,
                                        K=args.k_fold,
                                        input_h=args.input_h,
                                        input_w=args.input_w,
                                        input_s=args.input_s,
                                        down_ratio=args.down_ratio,
                                        down_size=args.down_size,
                                        mode=self.mode,
                                        sigmas=None)
        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True)

        mean_pe_error = 0 # means the mean point to point distance of all the test CTs
        ID_correct = 0  # means the number of correct localzed landmarks of all landmarks in test CTs
        standard_deviation = [] #标准差
        for cnt, data_dict in enumerate(data_loader):
            for name in data_dict:
                # 将数据放入显存中
                if name!='img_id':
                    data_dict[name] = data_dict[name].to(device=self.device)
            images = data_dict['input'][0]
            origin_images = data_dict['origin_image'][0]
            img_id = data_dict['img_id'][0]
            #hm_gt = data_dict['hm']
            #reg_gt = data_dict['reg'].cpu().numpy()[0]
            pts_gt = data_dict['landmarks'].cpu().numpy()[0]
            pts_gt = np.asarray(pts_gt,dtype=np.float32)

            pts_gt *= (self.down_ratio*self.down_size)
            #print('reg_gt: ' , reg_gt)
            #pts_gt += reg_gt
            pts_gt = pts_gt.tolist()
            #pts_gt = self.pts_sort(pts_gt,args.mode)
            #pts_gt.sort(key=lambda x: (x[0], x[1], x[2]))
            pts_gt = np.asarray(pts_gt,dtype=np.int32)

            images = images.to('cuda')
            print('processing {}/{} image ... {}'.format(cnt+1, len(data_loader), img_id))
            with torch.no_grad():
                output = self.model(images)
                hm = output['hm']
                # 若第一步输出5个hm的话，使用下面这一部分
                # hm1 = output['hm1']
                # hm2 = output['hm2']
                # hm3 = output['hm3']
                # hm4 = output['hm4']
                # hm5 = output['hm5']
                # hm = [hm1,hm2,hm3,hm4,hm5]

                #wh = output['wh']
                #reg = output['reg']
                reg = 1
                #loss = self.criterion(output, data_dict)
            # 等待当前设备上所有流中的所有核心完成。
            torch.cuda.synchronize(self.device)
            #将heatmap还原为坐标

            if self.mode == 'spine_localisation' or self.mode == 'landmark_detection':
                hm = hm[0]
                pts_predict = []
                for i in range(self.output_channel):
                    pts2 = self.decoder.ctdet_decode(hm[i].reshape((1, 1, int(args.input_s/self.down_ratio/self.down_size), int(args.input_h/self.down_ratio/self.down_size), int(args.input_w/self.down_ratio/self.down_size))), reg, True,down_ratio=self.down_ratio,down_size=self.down_size)
                    pts0 = pts2.copy()
                    pts0[:self.points_num, :3] *= (self.down_ratio * self.down_size)
                    pts_now = pts0[:self.points_num, :3].tolist()
                    pts_predict.extend(pts_now)
                    # pts_now = pts0[:self.points_num, :3].tolist()[1]
                    # pts_predict.append(pts_now)
            else:
                pts2 = self.decoder.ctdet_decode(hm, reg, True,down_ratio=self.down_ratio,down_size=self.down_size)   # 17, 11
                pts0 = pts2.copy()

                pts0[:self.points_num,:3] *= (self.down_ratio * self.down_size)
                pts_predict = pts0[:self.points_num,:3].tolist()

            print('totol pts num is {}'.format(len(pts_predict)))
            images = images.to('cpu')
            images = images.numpy()
            origin_images = origin_images.to('cpu')
            origin_images = origin_images.numpy()
            pts_predict = np.asarray(pts_predict,'float32')
            pts_predict = np.asarray(np.round(pts_predict), 'int32')
            draw.draw_by_matplotlib(origin_images[0][0], pts_gt[:10])
            draw.draw_by_matplotlib(origin_images[0][0], pts_gt[10:])
            pts_predict = pts_predict.tolist()
            #pts_predict = self.pts_sort(pts_predict,args.mode)
            pts_predict = np.asarray(pts_predict,'int32')
            print('pts_predict: ', pts_predict.tolist())
            print('pts_gt:      ',pts_gt.tolist())
            if self.KZdataset:
                sitk_image = sitk.ReadImage(os.path.join(args.data_dir, 'data', 'k_fold_smooth',img_id))
            else:
                sitk_image = sitk.ReadImage(os.path.join(args.data_dir, 'data', 'test',img_id))
            sitk_spacing = np.asarray(sitk_image.GetSpacing(), 'float32')
            temp = sitk_spacing[0].copy()
            sitk_spacing[0] = sitk_spacing[2]
            sitk_spacing[2] = temp

            # if args.mode == "landmark_detection":
            
            #     predict_diameters = []
            #     gt_diameters = []
            #     for i in range(5):
            #         predict_diameters.append(pts_predict[4 * i + 0] - pts_predict[4 * i + 1])
            #         gt_diameters.append(pts_gt[4 * i + 0] - pts_gt[4 * i + 1])
            
            #         predict_diameters.append(pts_predict[4 * i + 2] - pts_predict[4 * i + 3])
            #         gt_diameters.append(pts_gt[4 * i + 2] - pts_gt[4 * i + 3])
            
            #     predict_diameters = np.asarray(predict_diameters,'float32')
            #     gt_diameters = np.asarray(gt_diameters,'float32')
            #     predict_diameters = predict_diameters * sitk_spacing
            #     gt_diameters = gt_diameters * sitk_spacing
            #     for i in range(10):
            #         print("groundtruth pedical diameter: ",np.sqrt(np.sum(np.square(gt_diameters[i]))))
            #         print("predict pedical diameter: ",np.sqrt(np.sum(np.square(predict_diameters[i]))))

            #draw.draw_points(origin_images,np.asarray(pts_predict,'int32'),pts_gt,self.mode)

            # draw.draw_by_matplotlib(origin_images[0][0],np.asarray(pts_predict,'int32'))

            gap_predict_truth = pts_predict - pts_gt
            gap_predict_truth = gap_predict_truth * sitk_spacing
            print("sitk_spacing: ",sitk_spacing)
            mean_oushi_distance = 0
            #print(gap_predict_truth)
            for i in range(len(gap_predict_truth)):
                point_distance = np.sqrt(np.sum(np.square(gap_predict_truth[i])))
                standard_deviation.append(point_distance)
                mean_oushi_distance = mean_oushi_distance + point_distance
                if point_distance < 5:
                    ID_correct += 1
            mean_oushi_distance = mean_oushi_distance / len(gap_predict_truth)
            print("mean oushi distance: ",mean_oushi_distance)
            mean_pe_error += mean_oushi_distance
        ID_rate = ID_correct / (len(data_loader) * args.K * self.output_channel)
        mean_pe_error = mean_pe_error/len(data_loader)
        standard_deviation = np.asarray(standard_deviation,dtype='float32')
        standard_deviation -= mean_pe_error
        standard_deviation = np.sqrt(np.sum(np.square(standard_deviation))/(40 * len(data_loader)))
        print('In step of {}, test {} CTs, the mean point to point error is {} mm, the standard_deviation is {} mm, ID rate is {} %'.format(args.mode, len(data_loader), mean_pe_error,standard_deviation,ID_rate * 100))