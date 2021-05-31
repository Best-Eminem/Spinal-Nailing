import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import decoder
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

input = torch.randn(3, 3, requires_grad=True)
print(input)
result = incorrect_fft(input)
print(result)
result.backward(result)
print(input.grad)





