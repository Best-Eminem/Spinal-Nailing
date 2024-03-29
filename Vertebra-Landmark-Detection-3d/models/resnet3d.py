import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet_34': '/home/gpu/Spinal-Nailing/ZN-CT-nii/MedicalNet_pytorch_files/pretrain/resnet_34.pth',
    'resnet_34_23dataset': '/home/gpu/Spinal-Nailing/ZN-CT-nii/MedicalNet_pytorch_files/pretrain/resnet_34_23dataset.pth',
    'resnet_18': '/home/gpu/Spinal-Nailing/ZN-CT-nii/MedicalNet_pytorch_files/pretrain/resnet_18.pth',
    'resnet_10': '/home/gpu/Spinal-Nailing/ZN-CT-nii/MedicalNet_pytorch_files/pretrain/resnet_10.pth',
    'resnet_50': '/home/gpu/Spinal-Nailing/ZN-CT-nii/MedicalNet_pytorch_files/pretrain/resnet_50.pth',
}

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes = 1000,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2, dilation=1)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2, dilation=1)

        # self.conv_seg = nn.Sequential(
        #     nn.ConvTranspose3d(
        #         512 * block.expansion,
        #         32,
        #         2,
        #         stride=2
        #     ),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(
        #         32,
        #         32,
        #         kernel_size=3,
        #         stride=(1, 1, 1),
        #         padding=(1, 1, 1),
        #         bias=False),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(
        #         32,
        #         num_seg_classes,
        #         kernel_size=1,
        #         stride=(1, 1, 1),
        #         bias=False)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        feat = []
        feat.append(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feat.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat.append(x)
        x = self.layer2(x)
        feat.append(x)
        x = self.layer3(x)
        feat.append(x)
        x = self.layer4(x)
        feat.append(x)
        # x = self.conv_seg(x)

        return feat

def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        #加载预训练模型参数

        # state_dict = load_state_dict_from_url(model_urls[arch])
        # model.load_state_dict(state_dict, strict=False)
        state_dict = torch.load(model_urls[arch])
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    #model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    model = _resnet("resnet_10", BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = _resnet("resnet_18", BasicBlock, [2, 2, 2, 2], **kwargs)
    #model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    resnet_34_23dataset or resnet_34
    """
    model = _resnet("resnet_34",BasicBlock,[3, 4, 6, 3], **kwargs)
    # model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = _resnet("resnet_50",Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

# net = resnet34(sample_input_D = 3,sample_input_H = 1024,sample_input_W = 512)
# print(net)
# X = torch.rand((1, 3, 32, 1024, 512))
# M = X
# for name, layer in net.named_children():
#     M = layer(M)
#     print(name, ' output shape:\t', M.shape)
# X = net(X)
# print(X[5].shape,len(X))
