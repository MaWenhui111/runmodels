import torch
from torchsummary import summary
from unet.unet_parts2 import *
from utils.utils import pad


class ThreeNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, deep_mask=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deep_mask = deep_mask

        self.inc = (res_block(n_channels, 64, use_1x1conv=True))
        self.down1 = (Down(64, 64, 2))
        self.down2 = (Down(64, 128, 1))
        self.down3 = (Down(128, 256, 2))
        self.down4 = (Down(256, 512, 3))

        # self.down4 = (Down(512, 1024 // factor))
        self.up1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv1x1 = nn.Conv2d(960, 64, kernel_size=1, bias=False)
        if self.deep_mask:
            self.ds1 = (res_block(160, 80, use_1x1conv=True))
            self.ds2 = (res_block(80, n_classes, use_1x1conv=True))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y1 = self.up1(x5)
        y2 = self.up2(x4)
        y3 = self.up3(x3)
        y4 = self.up4(x2)
        try:
            Y = torch.cat([y4, y3, y2, y1], dim=1)
        except:
            y1 = pad(y1, x)
            y2 = pad(y2, x)
            y3 = pad(y3, x)
            y3 = pad(y3, x)
            Y = torch.cat([y4, y3, y2, y1], dim=1)
        # Y = self.conv1x1(Y)
        if self.deep_mask:
            deep_Y = self.ds1(Y)
            deep_Y = self.ds2(deep_Y)
            return Y, deep_Y
        else:
            return Y

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)


class SFNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.sclera_net = ThreeNet(3, 1)
        self.Iris_net = ThreeNet(3, 1)
        self.pupil_net = ThreeNet(3, 1)
        self.scSE = scSE(2880)
        self.conv1x1 = nn.Conv2d(2880, n_classes, kernel_size=1, bias=False)

    def forward(self, x, y, z):  # x:sclera_img, y:Iris_img, z:pupil_img
        X = self.sclera_net(x)
        Y = self.Iris_net(y)
        Z = self.pupil_net(z)
        concat = torch.cat([X, Y, Z], dim=1)
        attention = self.scSE(concat)
        output = self.conv1x1(attention)
        return output


if __name__ == '__main__':
    model = SFNet(3, 4)
    x = torch.rand(2, 3, 170, 300)
    y = torch.rand(2, 3, 170, 300)
    z = torch.rand(2, 3, 170, 300)
    summary(model, [(3, 170, 300), (3, 170, 300), (3, 170, 300)], device='cpu')  # (3,128,96)为输入的图片尺寸
    print(model(x, y, z).shape)

