import torch.nn.functional as F
import numpy as np
import torch

class DecDecoder(object):
    def __init__(self, K, conf_thresh):
        # k为landmarks数量
        self.K = K
        self.conf_thresh = conf_thresh

    def _topk(self, scores):
        batch, cat, slice, height, width = scores.size()


        #tensor.view()方法可以调整tensor的形状，但必须保证调整前后元素总数一致。
        scores = scores.view(batch, cat, -1)
        # torch.topk获取hm中前K大的点的值以及序号
        topk_scores, topk_inds = torch.topk(scores, self.K)
        topk_zs = topk_inds / (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_zs = self._gather_feat(topk_zs.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_zs, topk_ys, topk_xs


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool3d(heat, (kernel, kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

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
        #permute将tensor中dim换位
        feat = feat.permute(0, 2, 3, 4, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(4))
        feat = self._gather_feat(feat, ind)
        return feat

    def ctdet_decode(self, heat, reg):
        # output: num_obj x 7
        # 7: cenx, ceny, w, h, angle, score, cls
        batch, c, slice, height, width = heat.size()
        heat_a = heat[0,0].to('cpu')
        heat_a = heat_a.numpy()

        #max_points,index = torch.max(heat_a)
        heat = self._nms(heat)
        scores, inds, zs, ys, xs = self._topk(heat)
        scores = scores.view(batch, self.K, 1)
        reg = self._tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, self.K, 3)
        tp = reg[:, :, 2:3]
        zs = zs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        xs = xs.view(batch, self.K, 1) + reg[:, :, 2:3]
        #wh = self._tranpose_and_gather_feat(wh, inds)
        #wh = wh.view(batch, self.K, 2*4)

        # tl_x = xs - wh[:,:,0:1]
        # tl_y = ys - wh[:,:,1:2]
        # tr_x = xs - wh[:,:,2:3]
        # tr_y = ys - wh[:,:,3:4]
        # bl_x = xs - wh[:,:,4:5]
        # bl_y = ys - wh[:,:,5:6]
        # br_x = xs - wh[:,:,6:7]
        # br_y = ys - wh[:,:,7:8]

        pts = torch.cat([zs, ys, xs,
                         # tl_x,tl_y,
                         # tr_x,tr_y,
                         # bl_x,bl_y,
                         # br_x,br_y,
                         scores], dim=2).squeeze(0)
        return pts.data.cpu().numpy()

    def get_landmarks(self,heat):
        return 0
