import torch
import torch.nn as nn
import os
import numpy as np
from models import spinal_net
import decoder
import loss
from dataset import BaseDataset
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
            if name !='itk_information':
                out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        if name != 'itk_information':
            out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict

class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #heads表示的是最后一层输出的通道数
        heads = {'hm': args.num_classes * 5,

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
        self.model = spinal_net.SpineNet(heads=heads,
                                         pretrained=True,
                                         down_ratio=args.down_ratio,
                                         final_kernel=1,
                                         head_conv=256)

        self.num_classes = args.num_classes # 1
        # *******************************解码器 待修改,后面没用上？
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh) #K为特征点的最大个数
        self.dataset = {'spinal': BaseDataset}
        self.writer = SummaryWriter(logdir='spine_localisation_one',comment='SpineNet')


    def save_model(self, path, epoch, model):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        data = {'epoch': epoch, 'state_dict': state_dict}
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
        return model

    def train_network(self, args):

        # args.dataset = 'spinal'
        save_path = args.model_dir
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)

        # 自动调整学习率的调节器
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)
        if args.ngpus>0:
            # 多显卡时，使用多个GPU加速训练
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
        #save_path_load = 'weights_' + args.dataset + '//without_point_loss'
        #self.model = self.load_model(self.model, os.path.join(save_path_load, args.resume))
        self.model.to(self.device)
        # criterion为loss函数
        # *******************************************待修改
        criterion = loss.LossAll()
        print('Setting up data...')

        # dataset_module就是baseDataSet类的实例
        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   input_s=args.input_s,
                                   down_ratio=args.down_ratio,
                                   downsize=args.down_size,
                                   mode=self.mode)
                 for x in ['train', 'val']}

        dsets_loader = {'train': torch.utils.data.DataLoader(dsets['train'],
                                                             batch_size=args.batch_size,
                                                             shuffle=True,
                                                             num_workers=args.num_workers,
                                                             pin_memory=True,
                                                             drop_last=True,
                                                             collate_fn=collater),

                        'val':torch.utils.data.DataLoader(dsets['val'],
                                                          batch_size=1,
                                                          shuffle=False,
                                                          num_workers=1,
                                                          pin_memory=True, #放入GPU专属内存
                                                          collate_fn=collater)}


        print('Starting training...')
        train_loss = []
        val_loss = []
        now_time = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
        time_path = os.path.join(args.mode,now_time)
        info_savepath = os.path.join(save_path,time_path)
        if not os.path.exists(info_savepath):
            os.mkdir(info_savepath)
        for epoch in range(1, args.num_epoch+1):
            print('-'*30)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion)
            self.writer.add_scalar('TrainLoss', epoch_loss,epoch)
            train_loss.append(epoch_loss)
            #调整学习率
            scheduler.step(epoch)

            epoch_loss = self.run_epoch(phase='val',
                                        data_loader=dsets_loader['val'],
                                        criterion=criterion)
            self.writer.add_scalar('ValLoss', epoch_loss, epoch)
            val_loss.append(epoch_loss)

            np.savetxt(os.path.join(info_savepath, 'train_loss.txt'), train_loss, fmt='%.9f')
            np.savetxt(os.path.join(info_savepath, 'val_loss.txt'), val_loss, fmt='%.9f')

            if epoch % 10 == 0 or epoch ==1:
                self.save_model(os.path.join(info_savepath, 'model_{}.pth'.format(epoch)), epoch, self.model)
            # 保存模型
            if len(val_loss)>1:
                if val_loss[-1]<np.min(val_loss[:-1]):
                    self.save_model(os.path.join(info_savepath, 'model_last_origin.pth'), epoch, self.model)

    def run_epoch(self, phase, data_loader, criterion):
        if phase == 'train':
            # 设置为训练模式
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        # 循环从dataloader中取出batchsize个数据
        num = 0
        for data_dict in data_loader:
            num += 1
            for name in data_dict:
                # 将数据放入显存中
                if name != 'img_id' and name!='itk_information':
                    data_dict[name] = data_dict[name].to(device=self.device)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs = self.model(data_dict['input'])

                    loss = criterion(pr_decs, data_dict)

                    loss.backward()
                    # graph = make_dot(loss)  # 画计算图
                    #
                    # graph.view('model_structure1', '.\\imgs\\')
                    # for name, parms in self.model.named_parameters():
                    #     print('-->name:', name, '-->is_leaf:',parms.is_leaf,'-->grad_requirs:', parms.requires_grad,
                    #           ' -->grad_value:', parms.grad)
                    # print(data_dict['input'].grad)
                    #loss.backward(torch.tensor(100.))
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)

            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} {} CTs,loss: {}'.format(phase, num, epoch_loss))
        return epoch_loss

