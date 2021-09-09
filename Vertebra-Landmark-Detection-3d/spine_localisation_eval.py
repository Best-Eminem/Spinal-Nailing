import torch
import numpy as np
from models import spinal_net
import decoder
import os
from dataset import BaseDataset
import joblib
import loss
from matplotlib import pyplot as plt

class spine_localisation_eval(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # heads表示的是最后一层输出的通道数
        heads = {'hm': 5 * args.num_classes,
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
        self.decoder = decoder.DecDecoder(K=1, conf_thresh=args.conf_thresh) #K为特征点的最大个数
        self.dataset = {'spinal': BaseDataset}
        self.criterion = loss.LossAll()
        self.points_num = 1 #K为特征点的最大个数
        self.down_ratio = args.down_ratio
        self.downsize = args.down_size

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def eval(self, args, save):
        save_path = args.model_dir
        self.model = self.load_model(self.model, os.path.join(save_path, 'spine_localisation/one output/clean/model_200.pth'))
        self.model = self.model.to(self.device)
        #不启用 Batch Normalization 和 Dropout。
        self.model.eval()

        #就是dataset_module = BaseDataset
        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='spine_localisation_eval',
                               input_s=args.input_s,
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio,downsize=args.down_size,
                               mode='spine_localisation')

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)


        for cnt, data_dict in enumerate(data_loader):
            for name in data_dict:
                # 将数据放入显存中
                if name!='img_id' and name!='origin_size':
                    data_dict[name] = data_dict[name].to(device=self.device)
            images = data_dict['input'][0]
            origin_images = data_dict['origin_image'][0]
            img_id = data_dict['img_id'][0]
            origin_size = data_dict['origin_size'].numpy()[0]
            # hm_gt = data_dict['hm']
            # reg_gt = data_dict['reg'].cpu().numpy()[0]
            pts_gt = data_dict['landmarks'].cpu().numpy()[0]
            pts_gt = np.asarray(pts_gt, dtype=np.float32)

            pts_gt *= (self.down_ratio * self.downsize)
            # print('reg_gt: ' , reg_gt)
            # pts_gt += reg_gt
            pts_gt = pts_gt.tolist()
            pts_gt.sort(key=lambda x: (x[0], x[1], x[2]))
            pts_gt = np.asarray(pts_gt, dtype=np.int32)
            images = images.to('cuda')
            print('processing {}/{} image ... {}'.format(cnt + 1, len(data_loader), img_id))
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

                # wh = output['wh']
                # reg = output['reg']
                reg = 1
                # loss = self.criterion(output, data_dict)
            # 等待当前设备上所有流中的所有核心完成。
            torch.cuda.synchronize(self.device)
            # 将heatmap还原为坐标
            hm = hm[0]
            pts_predict = []
            for i in range(5):
                pts2 = self.decoder.ctdet_decode(hm[i].reshape((1, 1,
                                                                int(args.input_s // self.down_ratio // self.downsize),
                                                                int(args.input_h / self.down_ratio / self.downsize),
                                                                int(args.input_w / self.down_ratio / self.downsize))),
                                                 reg, True, down_ratio=self.down_ratio, downsize=self.downsize)
                pts0 = pts2.copy()
                pts0[:self.points_num, :3] *= (self.down_ratio * self.downsize)
                pts_now = pts0[:self.points_num, :3].tolist()[0]
                pts_predict.append(pts_now)
            print('totol pts num is {}'.format(len(pts_predict)))
            pts_predict.sort(key=lambda x: (x[0], x[1], x[2]))
            pts_predict = np.asarray(pts_predict, 'float32')

            pts_predict[:, 0] = pts_predict[:, 0] / args.input_s * origin_size[0]
            pts_predict[:, 1] = pts_predict[:, 1] / args.input_h * origin_size[1]
            pts_predict[:, 2] = pts_predict[:, 2] / args.input_w * origin_size[2]
            pts_predict = np.asarray(np.round(pts_predict), 'int32')

            pts_center = np.mean(pts_predict,axis=0)
            pts_center = np.asarray(np.round(pts_center),'int32')
            #将数据保存到本地
            print('pts_predict: ', pts_predict.tolist())
            print('pts_center: ', pts_center.tolist())
            pts_dict = {}
            pts_dict['pts'] = pts_predict
            pts_dict['pts_center'] = pts_center
            eval_store_path = 'eval\\spine_localisation_eval\\' + img_id[0:-7] + '.eval'
            joblib.dump(pts_dict, os.path.join(args.data_dir,eval_store_path))
