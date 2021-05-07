import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import decoder

heat = torch.randn((5,5,5),requires_grad=True)
topk_scores, topk_inds = torch.topk(heat, 5)
topk_inds = topk_inds.float()
topk_inds.requires_grad = True
sum = topk_scores.sum()
sum.backward()
print(heat.grad)




