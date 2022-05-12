import sys
import os
from typing_extensions import final
from SimpleITK.SimpleITK import FastApproximateRank


o_path = os.getcwd()
sys.path.append(o_path)
from torch._C import dtype
import GeodisTK
from scipy import ndimage
from hausdorff import hausdorff_distance
from dataset.KZDataset import KZDataset
from preprocess.transform import resize_image_itk
import torch
import numpy as np
from models import spinal_net, unet_transformer
from models.SCNet import unet3d,unet3d_spatial
from models import DenseUNet3d
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
        self.output_channel = args.output_channel
        heads = {'msk': self.output_channel,
                 }

        # self.model = spinal_net.SpineNet(heads=heads,
        #                                  pretrained=False,
        #                                  down_ratio=args.down_ratio,
        #                                  final_kernel=1,
        #                                  head_conv=256,
        #                                  spatial=False,
        #                                  segmentation=True,
        #                                  net=args.net
        #                                  )
        self.model = unet_transformer.UNETR(in_channels=1, out_channels=1, img_size=(64,80,160), feature_size=32, norm_name='batch')

        # self.model = DenseUNet3d.DenseUNet3d()
        #self.model = unet3d_spatial(120,200,200)
        # self.model = unet3d(class_nums = 1)
        self.num_classes = args.num_classes
        self.dataset = {'spinal': BaseDataset}
        #self.dataset = {'spinal': KZDataset}
        self.criterion = loss.Seg_BceLoss()
        self.down_ratio = args.down_ratio
        self.down_size = args.down_size
        self.mode = args.mode
        self.lumbar_number = args.lumbar_number

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def dice_coef(self,output, target):#output为预测结果 target为真实结果
        smooth = 1e-5 #防止0除

        if torch.is_tensor(output):
            output = torch.sigmoid(output).data.cpu().numpy()
        if torch.is_tensor(target):
            target = target.data.cpu().numpy()

        intersection = (output * target).sum()

        return (2. * intersection + smooth) / \
            (output.sum() + target.sum() + smooth)
    def get_edge_points(self,img):
        """
        get edge points of a binary segmentation result
        """
        dim = len(img.shape)
        if (dim == 2):
            strt = ndimage.generate_binary_structure(2, 1)
        else:
            strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
        ero = ndimage.morphology.binary_erosion(img, strt)
        edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
        return edge
    def binary_hausdorff95(self,s, g, spacing=None):
        """
        get the hausdorff distance between a binary segmentation and the ground truth
        inputs:
            s: a 3D or 2D binary image for segmentation
            g: a 2D or 2D binary image for ground truth
            spacing: a list for image spacing, length should be 3 or 2
        """
        s_edge = self.get_edge_points(s)
        g_edge = self.get_edge_points(g)
        image_dim = len(s.shape)
        assert (image_dim == len(g.shape))
        if (spacing == None):
            spacing = [1.0] * image_dim
        else:
            assert (image_dim == len(spacing))
        img = np.zeros_like(s)
        if (image_dim == 2):
            s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
            g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
        elif (image_dim == 3):
            s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
            g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

        dist_list1 = s_dis[g_edge > 0]
        dist_list1 = sorted(dist_list1)
        dist1 = dist_list1[int(len(dist_list1)*0.95) -1]
        dist_list2 = g_dis[s_edge > 0]
        dist_list2 = sorted(dist_list2)
        dist2 = dist_list2[int(len(dist_list2)*0.95) -1]
        return max(dist1, dist2)
    def test(self, args, save,ZNSeg = False):
        save_path = args.model_dir
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        #不启用 Batch Normalization 和 Dropout。
        self.model.train()

        #就是dataset_module = BaseDataset
        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_s=args.input_s,
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio,down_size=args.down_size,
                               mode=self.mode)
        # dsets = dataset_module(data_dir=args.data_dir,
        #                             phase='test',
        #                             ki=args.ki,
        #                             K=args.k_fold,
        #                             input_h=args.input_h,
        #                             input_w=args.input_w,
        #                             input_s=args.input_s,
        #                             down_ratio=args.down_ratio,
        #                             down_size=args.down_size,
        #                             mode=self.mode)
        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True)

        dice_score = 0 # means the mean point to point distance of all the test CTs
        dice_standard_deviation = [] #标准差
        Hausdorff_standard_deviation = []
        my_Hausdorff_distance = 0
        for cnt, data_dict in enumerate(data_loader):
            for name in data_dict:
                # 将数据放入显存中
                if name =='input':
                    data_dict[name] = data_dict[name].to(device=self.device)
            images = data_dict['input'][0]
            msk_gt = data_dict['msk'][0][0].numpy()
            img_id = data_dict['img_id'][0]

            #images = images.to('cuda')

            with torch.no_grad():
                output = self.model(images)
                msk_pred = output['msk']
            # 等待当前设备上所有流中的所有核心完成。
            torch.cuda.synchronize(self.device)

            images = images.to('cpu')
            images = images.numpy()[0][0]
            msk_pred = torch.sigmoid(msk_pred).data.cpu().numpy()[0][0]
            # img = resize_image_itk(sitk.GetImageFromArray(msk_pred), newSize=[msk_gt.shape[2],msk_gt.shape[1],msk_gt.shape[0]], resamplemethod=sitk.sitkLinear)
            # msk_pred = sitk.GetArrayFromImage(img)
            if img_id.find('L5')!=-1:
                sp_threshold = 0.5
            else:sp_threshold = 0.5
            # msk_pred_cp = msk_pred.copy()
            # msk_pred_cp[msk_pred_cp>=sp_threshold] = 1
            # msk_pred_cp[msk_pred_cp<sp_threshold] = 0
            
            

            # draw.draw_by_matplotlib(origin_images[0][0],np.asarray(pts_predict,'int32'))
            if ZNSeg == False:
                
                tp_dice_score = 0
                final_threshold = 0.
                # 测试每个数据的最佳阈值
                for sp_threshold in range(1,90):
                    sp_threshold *=0.01
                    msk_pred_cp = msk_pred.copy()
                    msk_pred_cp[msk_pred_cp>=sp_threshold] = 1
                    msk_pred_cp[msk_pred_cp<sp_threshold] = 0
                    final_threshold = sp_threshold if tp_dice_score<=self.dice_coef(msk_pred_cp,msk_gt) else final_threshold
                    tp_dice_score = self.dice_coef(msk_pred_cp,msk_gt) if tp_dice_score<self.dice_coef(msk_pred_cp,msk_gt) else tp_dice_score
                
                msk_pred_cp = msk_pred.copy()
                msk_pred_cp[msk_pred_cp>=final_threshold] = 1
                msk_pred_cp[msk_pred_cp<final_threshold] = 0

                # msk_pred_cp[msk_pred_cp>=0.5] = 1
                # msk_pred_cp[msk_pred_cp<0.5] = 0

                # tp_dice_score = self.dice_coef(msk_pred_cp,msk_gt)
                tp_Hausdorff_distance = self.binary_hausdorff95(msk_pred_cp,msk_gt, spacing=None)
                dice_standard_deviation.append(tp_dice_score)
                Hausdorff_standard_deviation.append(tp_Hausdorff_distance)
                dice_score+=tp_dice_score
                my_Hausdorff_distance+=tp_Hausdorff_distance
                print('processing {}/{} image ... {},dice score is {},Hausdorff_distance is {},threshold is {}'.format(cnt + 1, len(data_loader), img_id,
                                                                              tp_dice_score,tp_Hausdorff_distance,final_threshold))

                # 存储重采样后的输入和输出
                output_save_path = os.path.join(args.data_dir, 'output')
                if not os.path.exists(output_save_path):
                    os.mkdir(output_save_path)
                # 修改msk为对应椎体的标签值20-24
                msk_pred_cp[msk_pred_cp==1] = 19+int(args.lumbar_number[-1]) if args.lumbar_number !='ALL' else 1
                msk_pred_cp[msk_pred_cp<0.5] = 0
                msk_itk = sitk.GetImageFromArray(msk_pred_cp)
                images_itk = sitk.GetImageFromArray(images)
                sitk.WriteImage(images_itk, os.path.join(output_save_path, img_id[:img_id.find('_ct')+1] + args.lumbar_number+'.nii.gz'))
                sitk.WriteImage(msk_itk, os.path.join(output_save_path, img_id[:img_id.find('_ct')+1] + args.lumbar_number + '_msk.nii.gz'))
            else:
                # 存储重采样后的输入和输出
                output_save_path = os.path.join(args.data_dir, 'output')
                if not os.path.exists(output_save_path):
                    os.mkdir(output_save_path)

                msk_itk = sitk.GetImageFromArray(msk_pred_cp)
                images_itk = sitk.GetImageFromArray(images)
                sitk.WriteImage(images_itk, os.path.join(output_save_path, img_id[:img_id.find(
                    '_ct') + 1] + args.lumbar_number + '.nii.gz'))
                sitk.WriteImage(msk_itk, os.path.join(output_save_path, img_id[:img_id.find(
                    '_ct') + 1] + args.lumbar_number + '_msk.nii.gz'))

        if ZNSeg == False:
            mean_dice_score = dice_score/len(data_loader)
            mean_Hausdorff_distance = my_Hausdorff_distance/len(data_loader)
            dice_standard_deviation = np.asarray(dice_standard_deviation,dtype='float32')
            dice_standard_deviation -= mean_dice_score
            dice_standard_deviation = np.sqrt(np.sum(np.square(dice_standard_deviation))/(len(data_loader)))

            Hausdorff_standard_deviation = np.asarray(Hausdorff_standard_deviation,dtype='float32')
            Hausdorff_standard_deviation -= mean_Hausdorff_distance
            Hausdorff_standard_deviation = np.sqrt(np.sum(np.square(Hausdorff_standard_deviation))/(len(data_loader)))

            print('In step of {}, test {} {} lumbar CTs, the mean dice score is {} +- {} ,the mean Hausdorff_distance is {} +- {}'\
            .format(args.mode, len(data_loader),self.lumbar_number, mean_dice_score,dice_standard_deviation,mean_Hausdorff_distance,Hausdorff_standard_deviation))
            return mean_dice_score,mean_Hausdorff_distance
        else:
            return 0,0