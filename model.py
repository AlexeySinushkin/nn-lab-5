import torch
import torch.nn as nn
from torchvision.models import vgg16

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





class VGGUNet(nn.Module):
    def __init__(self, num_classes=35, input_size=(224, 224)):
        super(VGGUNet, self).__init__()

        vgg = vgg16(pretrained=True)
        features = list(vgg.features.children())

        self.enc1 = nn.Sequential(*features[:4])
        self.pool1 = features[4]
        self.enc2 = nn.Sequential(*features[5:9])
        self.pool2 = features[9]
        self.enc3 = nn.Sequential(*features[10:16])
        self.pool3 = features[16]
        self.enc4 = nn.Sequential(*features[17:23])
        self.pool4 = features[23]

        for param in vgg.parameters():
            param.requires_grad = False
        for param in self.enc3.parameters():
            param.requires_grad = True
        for param in self.enc4.parameters():
            param.requires_grad = True

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.dec4 = self.double_conv(1024 + 512, 512)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.double_conv(512 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(256 + 128, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.double_conv(128 + 64, 64)

        self.up_final = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=1)
        self.final = self.double_conv(64, num_classes, kernel_size=1)

        self.input_size = input_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def double_conv(self, in_ch, out_ch, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )

    def preprocess(self, x):
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        x = (x - self.mean) / self.std
        return x

    def crop_or_resize(to_resize, target_size):
        return nn.functional.interpolate(to_resize, size=target_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        original_size = x.shape[2:]
        x = self.preprocess(x)

        c1 = self.enc1(x); p1 = self.pool1(c1)
        c2 = self.enc2(p1); p2 = self.pool2(c2)
        c3 = self.enc3(p2); p3 = self.pool3(c3)
        c4 = self.enc4(p3); p4 = self.pool4(c4)

        b = self.bottleneck(p4)

        u4 = self.up4(b); u4 = torch.cat([u4, c4], dim=1); d4 = self.dec4(u4)
        u3 = self.up3(d4); u3 = torch.cat([u3, c3], dim=1); d3 = self.dec3(u3)
        u2 = self.up2(d3); u2 = torch.cat([u2, c2], dim=1); d2 = self.dec2(u2)
        u1 = self.up1(d2); u1 = torch.cat([u1, c1], dim=1); d1 = self.dec1(u1)

        out = self.up_final(d1)
        out = nn.functional.relu(out)
        out = nn.functional.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        out = self.final(out)
        return out
