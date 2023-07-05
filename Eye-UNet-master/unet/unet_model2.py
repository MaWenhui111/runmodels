""" Full assembly of the parts to form the complete network """
import torch
from torchsummary import summary
from unet.unet_parts2 import *


class Eye_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (res_block(n_channels, 64, use_1x1conv=True))
        self.down1 = (Down(64, 64, 2))
        self.down2 = (Down(64, 128, 1))
        self.down3 = (Down(128, 256, 2))
        self.down4 = (Down(256, 512, 3))
        self.end = (res_block(512, 256, use_1x1conv=True))

        # self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(256, 128))
        self.up2 = (Up(128, 64))
        self.ds1 = (res_block(64, 32, use_1x1conv=True))
        self.ds2 = (res_block(32, n_classes, use_1x1conv=True))
        self.up3 = (Up(64, 64, use_conv1x1=False))
        # self.up4 = (Up(64, 32))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = res_block(64, 32, use_1x1conv=True)
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.end(x5)
        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        y = self.ds1(x)
        y = self.ds2(y)
        x = self.up3(x, x2)
        x = self.up(x)
        x = self.conv(x)
        logits = self.outc(x)
        # return logits
        return {'output': logits,
                'output_mini': y}

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.end = torch.utils.checkpoint(self.end)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == '__main__':
    model = Eye_UNet(3, 2)
    summary(model, (3, 96, 128), device='cpu')  # (3,128,96)为输入的图片尺寸
    x = torch.rand(2, 3, 96, 128)
    print(model(x)['output'].shape)
    print(model(x)['output_mini'].shape)
