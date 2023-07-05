import torch
from torchsummary import summary
from unet.unet_parts import *
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
        self.conv1x1 = nn.Conv2d(960, 64, kernel_size=1, bias=False)
        if self.deep_mask:
            self.ds1 = (res_block(512, 256, use_1x1conv=True))
            self.ds2 = (res_block(256, self.n_classes, use_1x1conv=True))

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
        Y = self.conv1x1(Y)
        if self.deep_mask:
            deep_Y = self.ds1(y1)
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
    def __init__(self, n_channels=3, n_classes=4, deep_mask=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.sclera_net = ThreeNet(3, 2, deep_mask)
        self.Iris_net = ThreeNet(3, 2, deep_mask)
        self.pupil_net = ThreeNet(3, 2, deep_mask)
        self.scSE = scSE(192)
        self.conv1x1 = nn.Conv2d(192, n_classes, kernel_size=1, bias=False)

    def forward(self, x, y, z):  # x:sclera_img, y:Iris_img, z:pupil_img
        X, deepX = self.sclera_net(x)
        Y, deepY = self.Iris_net(y)
        Z, deepZ = self.pupil_net(z)
        concat = torch.cat([X, Y, Z], dim=1)
        attention = self.scSE(concat)
        output = self.conv1x1(attention)
        return output, deepX, deepY, deepZ

    def use_checkpointing(self):
        self.sclera_net = torch.utils.checkpoint(self.sclera_net)
        self.Iris_net = torch.utils.checkpoint(self.Iris_net)
        self.pupil_net = torch.utils.checkpoint(self.pupil_net)
        self.scSE = torch.utils.checkpoint(self.scSE)
        self.conv1x1 = torch.utils.checkpoint(self.conv1x1)


if __name__ == '__main__':
    model = SFNet(3, 4, deep_mask=True)
    x = torch.rand(2, 3, 102, 180)
    y = torch.rand(2, 3, 102, 180)
    z = torch.rand(2, 3, 102, 180)
    summary(model, [(3, 102, 180), (3, 102, 180), (3, 102, 180)], device='cpu')  # (3,128,96)为输入的图片尺寸
    mask, deep1, deep2, deep3 = model(x, y, z)
    print(mask.shape, deep1.shape, deep2.shape, deep3.shape)
