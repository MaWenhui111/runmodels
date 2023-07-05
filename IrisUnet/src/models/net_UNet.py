import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import sys
#sys.path.append(r'D:\Users\userLittleWatermelon\codes\CE_Net_example\CE_Net_example')
sys.path.append('sdata/wenhui.ma/program/IrisUnet/src/')

from functools import partial
from src.models.vgg import *
import src.models.vgg as vg

nonlinearity = partial(F.relu, inplace=True)


################################################################################################
# 原始的u-net,基本上和原始的u-net相同,基于vgg

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# 如下为旧的上采样方式
# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=True):
#         super(up, self).__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
#
#         self.conv = double_conv(in_ch, out_ch)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffX = x1.size()[2] - x2.size()[2]
#         diffY = x1.size()[3] - x2.size()[3]
#         x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #scale_factor 指定输出尺寸是输入尺寸的倍数
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, kernel_size=4, stride=2, padding=1)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        #print('x/')
        #print(x)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x



class UNet(nn.Module):
    def __init__(self, n_classes=4, backbone='vgg16', pretrained=None):
        super(UNet, self).__init__()
#        self.backbone = VGG16_BN(pretrained_vgg=pretrained)
        #backbones = {'vgg16': vggnet.VGG}

        #backbones = {'vgg16':vg.VGG16_BN}
        #self.backbone = backbones[backbone](pretrained_vgg=pretrained)
        self.backbone  = vg.VGG16_BN(pretrained_vgg = pretrained)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        # Encoder
        x1, x2, x3, x4, x5 = self.backbone(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

if __name__ == '__main__':
    import torch

    random_input = torch.randn((2, 3, 32, 32))
    input = torch.randn((2, 3, 2, 3))
    print(input)

    net3 = UNet(backbone='vgg16',
                pretrained='/sdata/wenhui.ma/program/IrisUnet/src/pth/vgg16_bn-6c64b313.pth')
    x3 = net3(random_input)
