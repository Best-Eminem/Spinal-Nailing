import torch
import torch.nn as nn
import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
import numpy as np
from tqdm import tqdm
from dataset.KZDataset import KZDataset
from models import spinal_net, unet_transformer
from models.SCNet import unet3d,unet3d_spatial
from preprocess.pytorch_tools import EarlyStopping
from utils import decoder
from models.SCNet import unet3d
from models import DenseUNet3d
import loss
from dataset.dataset import BaseDataset
from matplotlib import pyplot as plt
from torchviz import make_dot
from tensorboardX import SummaryWriter
import time

def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            if name == 'hm':
                out_data_dict[name].append(sample[name])
            elif name !='itk_information':
                out_data_dict[name].append(torch.from_numpy(sample[name]))
            
    for name in out_data_dict:
        if name != 'itk_information':
            out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict

class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.mode = args.mode
        self.output_channel = args.output_channel
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #heads表示的是最后一层输出的通道数
        heads = {'msk': self.output_channel,
                 }
        # self.model = spinal_net.SpineNet(heads=heads,
        #                                  pretrained=False,
        #                                  down_ratio=args.down_ratio,
        #                                  final_kernel=1,
        #                                  head_conv=256,
        #                                  spatial=False,
        #                                  segmentation=True,
        #                                  net=args.net)
        # self.model = unet3d(class_nums = 1)
        # self.model = DenseUNet3d.DenseUNet3d()
        self.model = unet_transformer.UNETR(in_channels=1, out_channels=1, img_size=(64,80,160), feature_size=32, norm_name='batch')
        self.dataset = {'spinal': BaseDataset}
        #self.dataset = {'spinal': KZDataset} #use the K fold validation training
        self.writer = SummaryWriter(logdir='tensorboard_log/'+args.mode,comment='SpineNet')
        self.down_ratio = args.down_ratio


    def save_model(self, path, epoch, model):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        data = {'epoch': epoch, 
                'state_dict': state_dict,            
                'optimizer_state_dict': self.optimizer.state_dict(),
               }
        torch.save(data, path)

    def load_model(self, model, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        state_dict = {}

        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()

        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        # model.load_state_dict(state_dict, strict=False)
        model.load_state_dict(state_dict_, strict=False)
        # self.heatmap_sigmas = checkpoint['heatmap_sigmas']
        # self.heatmap_sigmas.requires_grad = True
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.sigma_optimizer.load_state_dict(checkpoint['sigma_optimizer_state_dict'])
        for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        return model

    def train_network(self, args):
        patience = 20	# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
        delta=0.0000001  # 表示每回合验证集loss的变化大小
        #early_stopping = EarlyStopping(patience, verbose=True, delta=delta)	# 关于 EarlyStopping 的代码可先看博客后面的内容
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        save_path = args.model_dir
        #训练中断时，可加载模型继续训练
        #self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        # args.dataset = 'spinal'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        

        # 自动调整学习率的调节器
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.975, last_epoch=-1)
        if args.ngpus>0:
            # 多显卡时，使用多个GPU加速训练
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
        #save_path_load = 'weights_' + args.dataset + '//without_point_loss'

        self.model.to(self.device)
        # criterion为loss函数
        criterion = loss.Seg_BceLoss()
        print('Setting up data...')
        
        # dataset_module就是baseDataSet类的实例
        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   input_s=args.input_s,
                                   down_ratio=args.down_ratio,
                                   down_size=args.down_size,
                                   mode=self.mode,
                                   sigmas=None
                                   )
                 for x in ['train', 'val']}
        # dsets = {x: dataset_module(data_dir=args.data_dir,
        #                             phase=x,
        #                             ki=args.ki,
        #                             K=args.k_fold,
        #                             input_h=args.input_h,
        #                             input_w=args.input_w,
        #                             input_s=args.input_s,
        #                             down_ratio=args.down_ratio,
        #                             downsize=args.down_size,
        #                             mode=self.mode,
        #                             sigmas=None)
        #             for x in ['train', 'val']}

        dsets_loader = {'train': torch.utils.data.DataLoader(dsets['train'],
                                                                batch_size=args.batch_size,
                                                                shuffle=True,
                                                                num_workers=args.num_workers,
                                                                pin_memory=True,
                                                                drop_last=True,
                                                                collate_fn=collater),

                        'val':torch.utils.data.DataLoader(dsets['val'],
                                                            batch_size=args.batch_size,
                                                            shuffle=False,
                                                            num_workers=args.num_workers,
                                                            pin_memory=True, #放入GPU专属内存
                                                            collate_fn=collater)}
        print('Starting training...')
        train_loss = []
        val_loss = []
        now_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
        if not os.path.exists(os.path.join(save_path,args.mode,args.lumbar_number)):
            os.mkdir(os.path.join(save_path,args.mode,args.lumbar_number))
        time_path = os.path.join(args.mode,args.lumbar_number,now_time)
        info_savepath = os.path.join(save_path,time_path)
        if not os.path.exists(info_savepath):
            os.mkdir(info_savepath)
        for epoch in range(1, args.num_epoch+1):
            # if epoch == 51:
            #     self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)

            print('-'*30)
            print('{}  Epoch: {}/{} '.format(args.lumbar_number,epoch, args.num_epoch))

            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion,fold_k = args.ki)
            self.writer.add_scalar('TrainLoss', epoch_loss,epoch)
            train_loss.append(epoch_loss)
            #调整学习率
            # scheduler.step(epoch)
            # scheduler_sigams.step(epoch)
            scheduler.step()

            epoch_loss = self.run_epoch(phase='val',
                                        data_loader=dsets_loader['val'],
                                        criterion=criterion,fold_k = args.ki)
            self.writer.add_scalar('ValLoss', epoch_loss, epoch)
            val_loss.append(epoch_loss)


            np.savetxt(os.path.join(info_savepath, 'train_loss.txt'), train_loss, fmt='%.9f')
            np.savetxt(os.path.join(info_savepath, 'val_loss.txt'), val_loss, fmt='%.9f')
            # save train lists and val lists for the current k fold validation training
            # file = open(os.path.join(info_savepath,'train_set.txt'), 'w')
            # file.write(str(dsets['train'].data_info))
            # file.close()
            # file = open(os.path.join(info_savepath,'val_set.txt'), 'w')
            # file.write(str(dsets['val'].data_info))
            # file.close()

            if epoch % 20 == 0 or epoch ==1:
                self.save_model(os.path.join(info_savepath, 'model_{}.pth'.format(epoch)), epoch, self.model)
            # 保存模型
            
            if len(val_loss)>1:
                if val_loss[-1]<np.min(val_loss[:-1]):
                    self.save_model(os.path.join(info_savepath, 'model_last_origin.pth'), epoch, self.model)

            # early_stopping(epoch_loss, self.model)
            # # 若满足 early stopping 要求
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     # 结束模型训练
            #     break

    def run_epoch(self, phase, data_loader, criterion, fold_k):
        if phase == 'train':
            # 设置为训练模式
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        # 循环从dataloader中取出batchsize个数据
        num = 0

        for data_dict in tqdm(data_loader,desc="Processing " + phase +", fold: "+str(fold_k)):
            num += 1
            for name in data_dict:
                # 将数据放入显存中
                if name != 'img_id' and name!='itk_information' and name!='landmarks':
                    data_dict[name] = data_dict[name].to(device=self.device)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs = self.model(data_dict['input'])
                    #pr_decs['hm'].retain_grad()

                    _s,_h,_w = data_dict['input'].shape[2:]
                    # msk_shape_s,msk_shape_h,msk_shape_w = np.asarray([_s/self.down_ratio,_h/self.down_ratio,_w/self.down_ratio],dtype=np.int32)

                    msk = data_dict['msk']
                    gt = {}

                    gt['msk'] = msk
                    msk_loss = criterion(pr_decs, gt)
                    loss = msk_loss

                    loss.backward()
                    self.optimizer.step()
                    
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    _s,_h,_w = data_dict['input'].shape[2:]
                    # hm_shape_s,hm_shape_h,hm_shape_w = np.asarray([_s/self.down_ratio,_h/self.down_ratio,_w/self.down_ratio],dtype=np.int32)

                    msk = data_dict['msk']
                    gt = {}

                    gt['msk'] = msk
                    msk_loss = criterion(pr_decs, gt)
                    loss = msk_loss
            
            # 清理显存
            del pr_decs
            del data_dict
            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} {} CTs,loss: {}'.format(phase, num, epoch_loss))

        return epoch_loss
