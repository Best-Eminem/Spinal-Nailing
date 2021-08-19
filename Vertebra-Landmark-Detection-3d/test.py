import torch
import numpy as np
from models import spinal_net
import cv2
import decoder
import os
from dataset import BaseDataset
import draw_points
import draw
import loss
from matplotlib import pyplot as plt

def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # heads表示的是最后一层输出的通道数
        heads = {'hm': args.num_classes * 5,

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
                                         head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh) #K为特征点的最大个数
        self.dataset = {'spinal': BaseDataset}
        self.criterion = loss.LossAll()
        self.points_num = args.K #K为特征点的最大个数
        self.down_ratio = args.down_ratio
        self.downsize = args.downsize
        self.mode = args.mode

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def map_mask_to_image(self, mask, img, color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)


    def test(self, args, save):
        save_path = 'E:\\Spinal-Nailing\\weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        #不启用 Batch Normalization 和 Dropout。
        self.model.eval()

        #就是dataset_module = BaseDataset
        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_s=args.input_s,
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio,downsize=args.downsize,
                               mode=self.mode)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)


        for cnt, data_dict in enumerate(data_loader):
            for name in data_dict:
                # 将数据放入显存中
                if name!='img_id':
                    data_dict[name] = data_dict[name].to(device=self.device)
            images = data_dict['images'][0]
            origin_images = data_dict['origin_images'][0]
            img_id = data_dict['img_id'][0]
            hm_gt = data_dict['hm']
            #reg_gt = data_dict['reg'].cpu().numpy()[0]
            pts_gt = data_dict['landmarks'].cpu().numpy()[0]
            pts_gt = np.asarray(pts_gt,dtype=np.float32)

            pts_gt *= (self.down_ratio*self.downsize)
            #print('reg_gt: ' , reg_gt)
            #pts_gt += reg_gt
            pts_gt = pts_gt.tolist()
            # for i in range(5):
            #     pts_gt.append([0,0,0])
            pts_gt.sort(key=lambda x: (x[0], x[1], x[2]))
            #pts_gt.append([0,0,0])
            #pts_gt.append([0, 0, 0])
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

            if self.mode == 'spine_localisation':
                hm = hm[0]
                pts_predict = []
                for i in range(5):
                    pts2 = self.decoder.ctdet_decode(hm[i].reshape((1, 1, int(args.input_s/self.down_ratio), int(args.input_h/self.down_ratio), int(args.input_w/self.down_ratio))), reg, True,down_ratio=self.down_ratio,downsize=self.downsize)
                    pts0 = pts2.copy()
                    pts0[:self.points_num, :3] *= (self.down_ratio * self.downsize)
                    pts_now = pts0[:self.points_num, :3].tolist()[0]
                    pts_predict.append(pts_now)
                    # pts_now = pts0[:self.points_num, :3].tolist()[1]
                    # pts_predict.append(pts_now)
            else:
                pts2 = self.decoder.ctdet_decode(hm, reg, True,down_ratio=self.down_ratio,downsize=self.downsize)   # 17, 11
                pts0 = pts2.copy()

                pts0[:self.points_num,:3] *= (self.down_ratio * self.downsize)
                pts_predict = pts0[:self.points_num,:3].tolist()

            print('totol pts num is {}'.format(len(pts_predict)))
            images = images.to('cpu')
            images = images.numpy()
            origin_images = origin_images.to('cpu')
            origin_images = origin_images.numpy()
            pts_predict.sort(key = lambda x:(x[0],x[1],x[2]))
            pts_predict = np.asarray(pts_predict,'float32')
            pts_predict = np.asarray(np.round(pts_predict), 'int32')
            print('pts_predict: ', pts_predict.tolist())
            print('pts_gt: ',pts_gt.tolist())

            draw.draw_points(origin_images,np.asarray(pts_predict),pts_gt,self.mode)
            # for i in range(3):
            #     img = images[0][0]
            #     pts = np.asarray(np.round(pts0),np.int32)
            #     slice = pts[i][0]
            #     if slice>=0 and slice<=35:
            #         y = pts[i][1]
            #         x = pts[i][2]
            #         img_slice = img[slice]
            #         img_slice[y][x] = -1
            #         plt.imshow(img_slice,cmap='gray')
            #         plt.show()

            # ori_image = dsets.load_image(dsets.img_ids.index(img_id))
            # ori_image_regress = cv2.resize(ori_image, (args.input_w, args.input_h))
            # ori_image_points = ori_image_regress.copy()
            #
            # h,w,c = ori_image.shape
            # pts0 = np.asarray(pts0, np.float32)
            # # pts0[:,0::2] = pts0[:,0::2]/args.input_w*w
            # # pts0[:,1::2] = pts0[:,1::2]/args.input_h*h
            # sort_ind = np.argsort(pts0[:,1])
            # pts0 = pts0[sort_ind]
            # #渲染
            #ori_image_regress, ori_image_points = draw_points.draw_landmarks_regress_test(pts0,
            #                                                                               ori_image_regress,
            #                                                                               ori_image_points)
            #
            # cv2.imshow('ori_image_regress', ori_image_regress)
            # cv2.imshow('ori_image_points', ori_image_points)
            # k = cv2.waitKey(0) & 0xFF
            # if k == ord('q'):
            #     cv2.destroyAllWindows()
            #     exit()
