""" Full assembly of the parts to form the complete network """

from .parts import *
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class UNet_NonLocal(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_NonLocal, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up_NonLocal(1024, 512 // factor, bilinear)
        self.up2 = Up_NonLocal(512, 256 // factor, bilinear)
        self.up3 = Up_NonLocal(256, 128 // factor, bilinear)
        self.up4 = Up_NonLocal(128, 64, bilinear)
        #分类
        self.outc = OutConv(64, n_classes)
        #gan
        self.outimg = OutConv(64, 3)
        self.tanh = nn.Tanh()
        #grid
        self.outgrid = OutConv(64, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #监督学习分支 学习mask
        logits = self.outc(x)
        #img分支
        imgx = self.outimg(x)
        #上下文损失分支
        gridx = self.outgrid(x)
        return logits, imgx, gridx

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
