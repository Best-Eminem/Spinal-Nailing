import torch
from torch._C import device, dtype
import torch.nn as nn
import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
import numpy as np
from tqdm import tqdm
from dataset.KZDataset import KZDataset
from models import spinal_net
from models.SCNet import unet3d,unet3d_spatial
from preprocess.pytorch_tools import EarlyStopping
from utils import decoder
import loss
from dataset.dataset import BaseDataset
from matplotlib import pyplot as plt
from torchviz import make_dot
from tensorboardX import SummaryWriter
import time
from draw.draw_3dgaussian import *

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
        self.output_channel = args.output_channel
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #heads表示的是最后一层输出的通道数
        heads = {'hm': args.num_classes * self.output_channel,

                 # 若第一步输出5个hm的话，使用下面这一部分
                 # 'hm1': args.num_classes,
                 # 'hm2': args.num_classes,
                 # 'hm3': args.num_classes,
                 # 'hm4': args.num_classes,
                 # 'hm5': args.num_classes,
                 # 'reg': 3*args.num_classes,
                 # 'normal_vector': 3*args.num_classes
                 # 不需要计算corner offset
                 #'wh': 3*4
                 }
        self.mode = args.mode
        self.KZdataset = args.KZdataset
        self.heatmap_sigmas = torch.nn.Parameter(torch.FloatTensor(self.output_channel * [12]), requires_grad=args.learnable_sigma)
        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256,
                                         spatial=False)
        #self.model = unet3d_spatial(120,200,200)
        #self.model = unet3d()
        self.num_classes = args.num_classes # 1
        # *******************************解码器,解析出特征点位置
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh) #K为特征点的最大个数
        self.dataset = {'spinal': BaseDataset} if not self.KZdataset else {'spinal': KZDataset}
        # self.dataset = {'spinal': KZDataset} #use the K fold validation training
        self.writer = SummaryWriter(logdir='tensorboard_log/'+args.mode,comment='SpineNet')
        self.learnable_sigmas = args.learnable_sigma
        self.down_ratio = args.down_ratio


    def save_model(self, path, epoch, model):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        data = {'epoch': epoch, 
                'state_dict': state_dict,            
                'optimizer_state_dict': self.optimizer.state_dict(), 
                'sigma_optimizer_state_dict':None if not self.learnable_sigmas else self.sigma_optimizer.state_dict(),
                'heatmap_sigmas':self.heatmap_sigmas
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
        if args.learnable_sigma:
            self.sigma_optimizer = torch.optim.Adam([self.heatmap_sigmas], 0.01)
        save_path = args.model_dir
        #训练中断时，可加载模型继续训练
        # self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        # args.dataset = 'spinal'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        

        # 自动调整学习率的调节器
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.975, last_epoch=-1)
        if args.learnable_sigma:
            scheduler_sigams = torch.optim.lr_scheduler.ExponentialLR(self.sigma_optimizer, gamma=0.96, last_epoch=-1)
        if args.ngpus>0:
            # 多显卡时，使用多个GPU加速训练
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
        #save_path_load = 'weights_' + args.dataset + '//without_point_loss'

        self.model.to(self.device)
        #self.heatmap_sigmas = self.heatmap_sigmas.cuda()
        # criterion为loss函数
        # *******************************************待修改
        criterion = loss.LossAll()
        self.Sigma_loss = loss.Sigma_loss()
        print('Setting up data...')
        
        # dataset_module就是baseDataSet类的实例
        dataset_module = self.dataset[args.dataset]
        if not self.KZdataset:
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
        else:
            dsets = {x: dataset_module(data_dir=args.data_dir,
                                        phase=x,
                                        ki=args.ki,
                                        K=args.k_fold,
                                        input_h=args.input_h,
                                        input_w=args.input_w,
                                        input_s=args.input_s,
                                        down_ratio=args.down_ratio,
                                        down_size=args.down_size,
                                        mode=self.mode,
                                        sigmas=None)
                        for x in ['train', 'val']}

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
        heatmap_loss_train = []
        heatmap_loss_val= []
        now_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
        time_path = os.path.join(args.mode,now_time)
        info_savepath = os.path.join(save_path,time_path)
        if not os.path.exists(info_savepath):
            os.mkdir(info_savepath)
        for epoch in range(1, args.num_epoch+1):
            # if epoch == 51:
            #     self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
            #sigmas_item = self.heatmap_sigmas.cpu().detach().numpy()
            #sigmas_item = self.heatmap_sigmas.cpu().detach().numpy() / 0.001

            print('-'*30)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))

            epoch_loss,heatmap_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion,fold_k = args.ki)
            self.writer.add_scalar('TrainLoss', epoch_loss,epoch)
            train_loss.append(epoch_loss)
            heatmap_loss_train.append(heatmap_loss)
            #调整学习率
            # scheduler.step(epoch)
            # scheduler_sigams.step(epoch)
            scheduler.step()
            if args.learnable_sigma:
                scheduler_sigams.step()

            epoch_loss,heatmap_loss = self.run_epoch(phase='val',
                                        data_loader=dsets_loader['val'],
                                        criterion=criterion,fold_k = args.ki)
            self.writer.add_scalar('ValLoss', epoch_loss, epoch)
            val_loss.append(epoch_loss)
            heatmap_loss_val.append(heatmap_loss)

            np.savetxt(os.path.join(info_savepath, 'train_loss.txt'), train_loss, fmt='%.9f')
            np.savetxt(os.path.join(info_savepath, 'val_loss.txt'), val_loss, fmt='%.9f')
            np.savetxt(os.path.join(info_savepath, 'heatmap_loss_train.txt'), heatmap_loss_train, fmt='%.9f')
            np.savetxt(os.path.join(info_savepath, 'heatmap_loss_val.txt'), heatmap_loss_val, fmt='%.9f')

            if self.KZdataset:
                # save train lists and val lists for the current k fold validation training
                file = open(os.path.join(info_savepath,'train_set.txt'), 'w')
                file.write(str(dsets['train'].data_info))
                file.close()
                file = open(os.path.join(info_savepath,'val_set.txt'), 'w')
                file.write(str(dsets['val'].data_info))
                file.close()

            if epoch % 10 == 0 or epoch ==1:
                self.save_model(os.path.join(info_savepath, 'model_{}.pth'.format(epoch)), epoch, self.model)
            # 保存模型
            
            if len(heatmap_loss_val)>1:
                if heatmap_loss_val[-1]<np.min(heatmap_loss_val[:-1]):
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
            self.model.train()
        running_loss = 0.
        heatmaps_loss = 0.
        sigmas_loss = 0.
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
                if self.learnable_sigmas:
                    self.sigma_optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs = self.model(data_dict['input'])
                    #pr_decs['hm'].retain_grad()
                    ct_landmark_int = data_dict['landmarks'].numpy()[0]
                    
                    _s,_h,_w = data_dict['input'].shape[2:]
                    hm_shape_s,hm_shape_h,hm_shape_w = np.asarray([_s/self.down_ratio,_h/self.down_ratio,_w/self.down_ratio],dtype=np.int32)

                    hm = torch.zeros((1,self.output_channel,hm_shape_s,hm_shape_h,hm_shape_w),dtype=torch.float32,device=self.device)
                    gt = {}
                    
                    #只预测椎弓根的点
                    #print(self.heatmap_sigmas)
                    for k in range(self.output_channel):
                        #print(k)
                        hm[0][k] = draw_umich_gaussian_with_torch(hm[0][k], ct_landmark_int[k], radius=self.heatmap_sigmas[k])
                    gt['hm'] = hm
                    heatmap_loss = criterion(pr_decs, gt)
                    sigma_loss = self.Sigma_loss(self.heatmap_sigmas)
                    if self.learnable_sigmas:
                        self.heatmap_sigmas.retain_grad()
                        
                        loss = heatmap_loss+sigma_loss
                    else:
                        loss = heatmap_loss
                    #sigma_grad = self.heatmap_sigmas.grad
                    #print('sigma_grad: ',sigma_grad)
                    #print('sigma_loss:',sigma_loss)
                    #print('heatmap_loss:',heatmap_loss)
                    loss.backward()
                    if self.learnable_sigmas:
                        sigma_grad = self.heatmap_sigmas.grad
                    #print('sigma_grad: ',sigma_grad)
                    
                    
                    self.optimizer.step()
                    if self.learnable_sigmas:
                        self.sigma_optimizer.step()
                    
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    ct_landmark_int = data_dict['landmarks'].numpy()[0]
                    _s,_h,_w = data_dict['input'].shape[2:]
                    hm_shape_s,hm_shape_h,hm_shape_w = np.asarray([_s/self.down_ratio,_h/self.down_ratio,_w/self.down_ratio],dtype=np.int32)

                    hm = torch.zeros((1,self.output_channel,hm_shape_s,hm_shape_h,hm_shape_w),dtype=torch.float32,device=self.device)
                    gt = {}
                    
                    #只预测椎弓根的点
                    for k in range(self.output_channel):
                        hm[0][k] = draw_umich_gaussian_with_torch(hm[0][k], ct_landmark_int[k], radius=self.heatmap_sigmas[k])
                    gt['hm'] = hm
                    heatmap_loss = criterion(pr_decs, gt)
                    sigma_loss = self.Sigma_loss(self.heatmap_sigmas)
                    if self.learnable_sigmas:                      
                        loss = sigma_loss + heatmap_loss
                    else:
                        loss = heatmap_loss
            
            # 清理显存
            del pr_decs
            del data_dict
            running_loss += loss.item()
            heatmaps_loss += heatmap_loss.item()
            sigmas_loss  += sigma_loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} {} CTs,loss: {}'.format(phase, num, epoch_loss))
        print('{} {} CTs,heatmaps_loss: {}'.format(phase, num, heatmaps_loss / len(data_loader)))
        
        # if self.learnable_sigmas:
        print('{} {} CTs,sigmas_loss: {}'.format(phase, num, sigmas_loss /  len(data_loader)))
        print('sigmas: ',(self.heatmap_sigmas*2+1)/12)
        return epoch_loss,heatmaps_loss / len(data_loader)
