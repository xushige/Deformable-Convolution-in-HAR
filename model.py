import torch.nn as nn
from v2 import DeformConv2d
import torch


class CNN(nn.Module):
    def __init__(self, seclast_div, last_div, classes, ifdeform, df=False):
        super(CNN, self).__init__()
        self.ifdeform = ifdeform
        self.deform_conv = nn.Sequential(
            DeformConv2d(1, 64, 3, 2, 1, modulation=df),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            DeformConv2d(64, 128, 3, 2, 1, modulation=df),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            DeformConv2d(128, 256, 3, 2, 1, modulation=df),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            DeformConv2d(256, 512, 3, 2, 1, modulation=df),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.normal_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.avgpool = nn.AvgPool2d((4, 1))
        self.flc = nn.Linear(512 * seclast_div * last_div, classes)
    def forward(self, x):
        if self.ifdeform:
            x = self.deform_conv(x)
        else:
            x = self.normal_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.flc(x)
        return x

#deform_RESNET
class Deform_Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride, df=False):
        super(Deform_Block, self).__init__()
        self.conv = nn.Sequential(
            DeformConv2d(inchannel, outchannel, 3, stride, 1, modulation=df),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            DeformConv2d(outchannel, outchannel, 3, 1, 1, modulation=df),
            nn.BatchNorm2d(outchannel)
        )
        self.short = nn.Sequential()
        if inchannel != outchannel or stride != 1:
            self.short = nn.Sequential(
                DeformConv2d(inchannel, outchannel, 3, stride, 1, modulation=df),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        x = self.conv(x) + self.short(x)
        return nn.ReLU()(x)
class Deform_resnet(nn.Module):
    def __init__(self, seclast_div, last_div, classes, df=False):
        super(Deform_resnet, self).__init__()
        self.df = df
        self.conv = nn.Sequential(
            DeformConv2d(1, 64, 3, 2, 1, modulation=self.df),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer = nn.Sequential(
            self.make_layers(64, 128, 2, 1),
            self.make_layers(128, 256, 2, 1),
            self.make_layers(256, 512, 2, 1),
            nn.AvgPool2d((4, 1))
        )
        self.flc = nn.Linear(512*seclast_div*last_div, classes)
    def make_layers(self, inchannel, outchannel, stride, times):
        layer = []
        layer.append(Deform_Block(inchannel, outchannel, stride, df=self.df))
        for i in range(1, times):
            layer.append(Deform_Block(outchannel, outchannel, 1, df=self.df))
        return nn.Sequential(*layer)
    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.flc(x)
        return x

#normal_RESNET
class Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1),
            nn.BatchNorm2d(outchannel)
        )
        self.short = nn.Sequential()
        if inchannel != outchannel or stride != 1:
            self.short = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 3, stride, 1),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        x = self.conv(x) + self.short(x)
        return nn.ReLU()(x)
class Resnet(nn.Module):
    def __init__(self, seclast_div, last_div, classes):
        super(Resnet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer = nn.Sequential(
            self.make_layers(64, 128, 2, 1),
            self.make_layers(128, 256, 2, 1),
            self.make_layers(256, 512, 2, 1),
            nn.AvgPool2d((4, 1))
        )
        self.flc = nn.Linear(512*seclast_div*last_div, classes)
    def make_layers(self, inchannel, outchannel, stride, times):
        layer = []
        layer.append(Block(inchannel, outchannel, stride))
        for i in range(1, times):
            layer.append(Block(outchannel, outchannel, 1))
        return nn.Sequential(*layer)
    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.flc(x)
        return x
