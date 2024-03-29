import sys
import os
o_path = os.getcwd()
sys.path.append(o_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import decoder


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2)
        ind = ind.expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        # permute将tensor中dim换位
        feat = feat.permute(0, 2, 3, 4, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(4))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        pred = self._tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        #print(pred * mask)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    def forward(self, pred, gt):
        # diff = pred - gt
        loss = F.mse_loss(pred, gt,reduction='sum')#/(torch.FloatTensor(2).cuda())
        return loss
class Sigma_loss(nn.Module):
    def __init__(self):
        super(Sigma_loss, self).__init__()
    def forward(self, pred):
        loss = torch.mean(torch.pow(pred,2)) * 0.01 *0.000001
        return loss
class Seg_BceLoss(nn.Module):
    def __init__(self):
        super(Seg_BceLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, pred, gt):
        return self.loss(pred['msk'],gt['msk'])
class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
      #由于pred里可能会有0出现，会导致log(0)为nan，所以将所有值裁剪到不为0的区间
      pred = torch.clamp(pred,1e-15,1)
      pos_inds = gt.eq(1).int()
      neg_inds = gt.lt(1).int()
      neg_weights = torch.pow(1 - gt, 4)

      loss = 0

      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    #   if torch.isnan(pos_loss.sum()) or torch.isnan(neg_loss.sum()):
    #       print('fuck nan occurs --------------------------------------')
    #       aa_min = torch.min(pos_loss)
    #       bb_min = torch.min(neg_loss)
    #       aa = torch.isnan(pos_loss).int().sum()
    #       pos_loss[torch.isnan(pos_loss)] = 0
    #       neg_loss[torch.isnan(neg_loss)] = 0
    #       bb = torch.isnan(neg_loss).int().sum()
    #       pp_min = torch.min(pred)
    #       pp_max = torch.max(pred)
      num_pos  = pos_inds.sum() + neg_inds.sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      loss = loss - (pos_loss + neg_loss) / num_pos
      if torch.isnan(loss) or torch.isinf(loss):
          print('nan occurs fuck--------------------------------------')
      return loss

class Point2PlaneLoss(nn.Module):
    def __init__(self):
        super(Point2PlaneLoss, self).__init__()
        self.points_num = 15
        self.down_ratio = 4
        self.conf_thresh = 0.2
        self.decoder = decoder.DecDecoder(self.points_num, self.conf_thresh)

    def distance(self,point,gt_plane):
        D = gt_plane[-1]
        other = gt_plane[:-1]
        sum = other.pow(2).sum().float()
        dis = (((point * other).sum() + D).abs())
        sum = (torch.sqrt(sum)+0.0001)
        point2plane = dis/sum
        # Ax, By, Cz, D = gt_plane
        # mod_d = Ax * point[0] + By * point[1] + Cz * point[2] + D
        # mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
        # dis = abs(mod_d) / mod_area
        #point2plane.requires_grad = True
        return point2plane
    def define_area(self,gt_points):
        """
            法向量    ：n={A,B,C}
            :return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
            """
        gt_planes = []
        for i in range(self.points_num//3):
            point1 = gt_points[0,3*i]
            point2 = gt_points[0,3*i+1]
            point3 = gt_points[0,3*i+2]
            AB = point2 - point1
            AC = point3 - point1
            #N = np.cross(AB, AC)  # 向量叉乘，求法向量
            N = torch.tensor([AB[1]*AC[2]-AB[2]*AC[1],AB[2]*AC[0]-AB[0]*AC[2],AB[0]*AC[1]-AB[1]*AC[0]])
            # Ax+By+Cz
            Ax = N[0]
            By = N[1]
            Cz = N[2]
            D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
            gt_planes.append([Ax, By, Cz, D])

        return torch.tensor(gt_planes,device='cuda')
    def get_pre_points(self,hm,reg):
        pts2 = self.decoder.ctdet_decode(hm, reg, False)
        #pts2[:self.points_num, :3] *= self.down_ratio
        pts_predict = pts2[:self.points_num, :3]
        pts_predict = pts2[:self.points_num, :3].tolist()
        pts_predict.sort(key=lambda x: (x[0], x[1]))
        pts_predict = np.asarray(pts_predict, 'float32')
        pts_predict = np.asarray(np.round(pts_predict), 'int32')
        pts_predict = torch.from_numpy(pts_predict).cuda()

        return pts_predict
    def forward(self,pr_hm,reg,gt_points):
        pred_points = self.get_pre_points(pr_hm,reg)
        pred_points = pred_points.float().round()
        #pred_points.requires_grad = True
        #pred_points.retain_grad()
        gt_planes = self.define_area(gt_points).float()
        #gt_planes.requires_grad = True
        #gt_planes.retain_grad()
        loss = 0
        loss_average = 0

        for i in range(self.points_num):
            loss +=  self.distance(pred_points[i],gt_planes[i//3])
            if (i+1)%3 == 0:
                loss_average += (loss/3)
                loss = 0
        return loss_average/5
    def backward(self, result):
        return result

class LossAll(torch.nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()

        self.L_hm = FocalLoss()
        self.L_off = RegL1Loss()
        self.L_dis = Point2PlaneLoss()
        self.L_normal_vector = RegL1Loss()
        self.L_L2_loss = L2Loss()
        self.Sigma_loss = Sigma_loss()

        # self.L_wh =  RegL1Loss()
    def forward(self, pr_decs, gt_batch):

        hm_loss  = 0
        #若第一步输出一个hm的话，使用这个
        hm_loss = self.L_hm(pr_decs['hm'],  gt_batch['hm'])
        # hm_loss = self.L_L2_loss(pr_decs['hm'],  gt_batch['hm'])
        # sigma_loss = self.Sigma_loss(heatmap_sigamas)
        # 若第一步输出5个hm的话，使用下面这一部分
        # hm_loss += self.L_hm(pr_decs['hm1'][0][0],  gt_batch['hm'][0][0])
        # hm_loss += self.L_hm(pr_decs['hm2'][0][0], gt_batch['hm'][0][1])
        # hm_loss += self.L_hm(pr_decs['hm3'][0][0], gt_batch['hm'][0][2])
        # hm_loss += self.L_hm(pr_decs['hm4'][0][0], gt_batch['hm'][0][3])
        # hm_loss += self.L_hm(pr_decs['hm5'][0][0], gt_batch['hm'][0][4])

        #point_dis_loss = self.L_dis(pr_decs['hm'],pr_decs['reg'],gt_batch['landmarks'])
        # 不需要 corner offset
        # wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        #normal_vector_loss = self.L_normal_vector(pr_decs['normal_vector'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['normal_vector'])
        #off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        loss_dec = hm_loss#+ off_loss + normal_vector_loss #+ point_dis_loss
        #loss_dec = point_dis_loss
        #print('hm_loss= ',loss_dec.item()) #, ' off_loss= ',off_loss.item() , ' normal_vector_loss= ',normal_vector_loss.item())
        return loss_dec

    # def backward(self,result):
    #     return self.L_dis.backward(result)
