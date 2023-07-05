""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class res_block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, num_res_blocks, first_block=False):
        super().__init__()
        self.blk = []

        self.blk.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for i in range(num_res_blocks):
            if i == 0 and not first_block:
                self.blk.append(res_block(in_channels, out_channels, use_1x1conv=True))
            else:
                self.blk.append(res_block(out_channels, out_channels))

        self.sequ = nn.Sequential(*self.blk)

    def forward(self, x):
        return self.sequ(x)


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        Y = self.Conv1x1(x)  # x:[bs,c,h,w] to q:[bs,1,h,w]
        Y = self.norm(Y)
        return x * Y  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        Y = self.avgpool(x)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        Y = self.Conv_Squeeze(Y) # shape: [bs, c/2]
        Y = self.Conv_Excitation(Y) # shape: [bs, c]
        Y = self.norm(Y)
        return x * Y.expand_as(x)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, x):
        U_sse = self.sSE(x)
        U_cse = self.cSE(x)
        return U_cse+U_sse


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, use_conv1x1=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = res_block(in_channels, out_channels, use_1x1conv=use_conv1x1)
        self.scSE = scSE(in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.scSE(x2)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        x = x2 + x1

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    res = res_block(in_channels=3, out_channels=64, use_1x1conv=True)
    x = torch.rand(1, 3, 128, 96)
    res(x)