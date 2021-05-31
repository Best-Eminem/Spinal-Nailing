import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import decoder

heat = torch.randn((1,1,25),requires_grad=True)
topk_scores, topk_inds = torch.topk(heat, 3)
topk_inds = topk_inds.float()
topk_inds.requires_grad = True
sum = topk_inds.sum()
sum.backward()
print(heat.grad)




