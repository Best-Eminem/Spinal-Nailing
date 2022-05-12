import torch
from torch import nn

from .building_blocks.DenseBlock import DenseBlock
from .building_blocks.TransitionBlock import TransitionBlock
from .building_blocks.UpsamplingBlock import UpsamplingBlock


class DenseUNet3d(nn.Module):
    def __init__(self):
        """
        Create the layers for the model
        """
        super().__init__()
        # Initial Layers
        self.conv1 = nn.Conv3d(
            1, 96, kernel_size=(7, 7, 7), stride=2, padding=(3, 3, 3)
        )
        self.bn1 = nn.BatchNorm3d(96)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))

        # Dense Layers
        self.transition = TransitionBlock(32)
        self.dense1 = DenseBlock(96, 128, 32, 4)
        self.dense2 = DenseBlock(32, 128, 32, 12)
        self.dense3 = DenseBlock(32, 128, 32, 24)
        self.dense4 = DenseBlock(32, 32, 32, 36)

        # Upsampling Layers
        self.upsample1 = UpsamplingBlock(32 + 32, 504, size=(2, 2, 2))
        self.upsample2 = UpsamplingBlock(504 + 32, 224, size=(2, 2, 2))
        self.upsample3 = UpsamplingBlock(224 + 32, 192, size=(2, 2, 2))
        self.upsample4 = UpsamplingBlock(192 + 32, 96, size=(2, 2, 2))
        self.upsample5 = UpsamplingBlock(96 + 96, 64, size=(2, 2, 2))

        # Final output layer
        # Typo in the paper? Says stride = 0 but that's impossible
        self.conv_classifier = nn.Sequential(nn.Conv3d(64, 1, kernel_size=1, stride=1),
                                             nn.Sigmoid())
        self.fc1 = nn.Sequential(nn.Conv3d(64, 256, kernel_size=3, padding=1, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Conv3d(256, 1, kernel_size=1, stride=1,
                                         padding=1 // 2, bias=True)
                               )
        # self.fc1[-1].bias.data.fill_(-2.19)
        self.fill_fc_weights(self.fc1)

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model

        :param x:  image tensor
        :return:   output of the forward pass
        """
        residual1 = self.relu(self.bn1(self.conv1(x)))
        residual2 = self.dense1(self.maxpool1(residual1))
        residual3 = self.dense2(self.transition(residual2))
        residual4 = self.dense3(self.transition(residual3))
        output = self.dense4(self.transition(residual4))
        # print(residual1.shape)
        # print(residual2.shape)
        # print(residual3.shape)
        # print(residual4.shape)
        # print(output.shape)
        output = self.upsample1(output, output)
        output = self.upsample2(output, residual4)
        output = self.upsample3(output, residual3)
        output = self.upsample4(output, residual2)
        output = self.upsample5(output, residual1)

        # output = self.conv_classifier(output)
        output = self.fc1(output)
        dec_dict = {}
        dec_dict['msk'] = output
        return dec_dict
