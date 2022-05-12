import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.fft import rfft2, irfft2
# heat = torch.randn((1,1,25),requires_grad=True)
# topk_scores, topk_inds = torch.topk(heat, 3)
# topk_inds = topk_inds.float()
# topk_inds.requires_grad = True
# sum = topk_inds.sum()
# sum.backward()
# print(heat.grad)

class BadFFTFunction(Function):
    @staticmethod
    def forward(ctx, input):
        numpy_input = input.detach().numpy()
        result = numpy_input.sum()
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        # numpy_go = grad_output.numpy()
        # result = numpy_go / 2
        # new_result = grad_output.new(result)
        return torch.tensor([[1,1,1],[1,1,1],[1,1,1]])

# since this layer does not have any parameters, we can
# simply declare this as a function, rather than as an nn.Module class


def incorrect_fft(input):
    return BadFFTFunction.apply(input)

class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
      pos_inds = gt.eq(1).int()
      neg_inds = gt.lt(1).int()
      neg_weights = torch.pow(1 - gt, 4)

      loss = 0

      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.sum() + neg_inds.sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      loss = loss - (pos_loss + neg_loss) / num_pos
      return loss
def draw_umich_gaussian_with_torch(heatmap, center, radius, k=1,scale = 1):
    diameter = 2 * radius + 1 #直径
    gaussian = gaussian3D_with_torch((diameter, diameter, diameter), sigma=diameter / 12, scale=scale) #
    #print(torch.max(gaussian))
    radius = int(radius)
    z, y, x = int(center[0]), int(center[1]),int(center[2])

    slice, height, width = heatmap.shape[0:3]

    behind, front = min(z, radius), min(slice - z, radius + 1)
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    heatmap[z - behind:z + front, y - top:y + bottom, x - left:x + right] = gaussian
    # masked_heatmap = heatmap[z - behind:z + front, y - top:y + bottom, x - left:x + right]
    # print(masked_heatmap.shape)
    # masked_gaussian = gaussian[radius - behind:radius + front, radius - top:radius + bottom, radius - left:radius + right]
    # print(masked_gaussian.shape)
    # if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
    #     masked_heatmap = torch.maximum(masked_heatmap, masked_gaussian * k)
    # print(torch.max(masked_heatmap))

    return heatmap

def gaussian3D_with_torch(shape, sigma=1, scale=1):
    #sigma = shape[0]/6
    s, m, n = [(ss.item() - 1.) / 2. for ss in shape]
    #生成三个array
    z = torch.arange(-s,s+1)
    y = torch.arange(-m,m+1)
    x = torch.arange(-n,n+1)
    #z, y, x = np.ogrid[-s:s+1,-m:m+1,-n:n+1]
    grid = torch.meshgrid(x,y,z)
    grid_stack = torch.stack(grid,axis=3)
    grid_stack.reshape(int(s*2+1),int(s*2+1),int(s*2+1),3)
    squared_distances = torch.sum(torch.pow(grid_stack, 2.0),axis=-1)
    #print(torch.max(squared_distances))
    h = scale * torch.exp((-squared_distances) / (2 * torch.pow(sigma,2)))
    #让h中极小的值取0
    #h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
# input = torch.randn(3, 3, requires_grad=True)
# print(input)
# result = incorrect_fft(input)
# print(result)
# result.backward(result)
# print(input.grad)

# input = np.zeros((120, 200, 200), dtype=np.float32)
# input = torch.from_numpy(input)
# sigma = torch.nn.Parameter(torch.FloatTensor([12]*10), requires_grad=True)
# heat = draw_umich_gaussian_with_torch(input,[50,50,50],sigma[0])
# ss = heat.detach().numpy()
# print(np.max(ss),np.sum(ss ==1))
# print(heat.shape)
# facal_loss = FocalLoss()
# loss = F.mse_loss(heat,torch.randn(heat.shape),reduction='sum')
# #loss = facal_loss(heat,torch.randn(heat.shape))
# sigma.retain_grad()
# #heat.requires_grad_(True) 
# loss.backward()

# print(sigma.grad)

# t = torch.rand(3,3,3)
# s = torch.randn(3,3,4)
# index = torch.tensor([[[0,0,0,0],[1,1,1,1]],[[1,1,1,1],[0,0,0,0]],[[0,0,0,0],[2,2,2,2]]]) 
# print(index.shape)
# #我这里想更新的是第一个tensor矩阵的第0，第1行，第二个tensor矩阵的第1，第0行，第三个tensor矩阵的第0和第2行。

# s.scatter_(1,index,t) 
# tt = torch.ones((3,3,3))
# tt[0:2,0:2,0:2] = 0
# print(tt)
# ss = torch.ones((2,2,2))
# c = torch._s_where(tt==0,ss,tt)
# print(tt)

tt = torch.tensor([1,2,3])
print((tt*2+1)/12)
print(tt)




