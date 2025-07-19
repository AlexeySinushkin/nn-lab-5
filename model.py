import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, num_classes=35):
        super(UNet, self).__init__()
        self.conv1 = self.double_conv(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.double_conv(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = self.double_conv(64, 128, kernel_size=7)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = self.double_conv(128 + 64, 64, kernel_size=5)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = self.double_conv(64 + 32, 32, kernel_size=3)

        self.final = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def double_conv(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)

        up1 = self.up1(c3)
        merge1 = torch.cat([c2, up1], dim=1)
        c4 = self.conv4(merge1)

        up2 = self.up2(c4)
        merge2 = torch.cat([c1, up2], dim=1)
        c5 = self.conv5(merge2)

        out = self.final(c5)
        return out