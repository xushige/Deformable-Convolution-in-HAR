import torch.nn as nn
from v2 import DeformConv2d
import torch

# class SKConv(nn.Module):
#     def __init__(self, features, WH, M, G, r, stride=1, L=32):
#         """ Constructor
#         Args:
#             features: input channel dimensionality.
#             WH: input spatial dimensionality, used for GAP kernel size.
#             M: the number of branchs.
#             G: num of convolution groups.
#             r: the radio for compute d, the length of z.
#             stride: stride, default 1.
#             L: the minimum dim of the vector z in paper, default 32.
#         """
#         super(SKConv, self).__init__()
#         d = max(int(features / r), L)
#         self.M = M
#         self.features = features
#         self.convs = nn.ModuleList([])
#         for i in range(M):
#             self.convs.append(nn.Sequential(
#                 nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
#                 nn.BatchNorm2d(features),
#                 nn.ReLU(inplace=False)
#             ))
#         # self.gap = nn.AvgPool2d(int(WH/stride))
#         self.fc = nn.Linear(features, d)
#         self.fcs = nn.ModuleList([])
#         for i in range(M):
#             self.fcs.append(
#                 nn.Linear(d, features)
#             )
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         for i, conv in enumerate(self.convs):
#             fea = conv(x).unsqueeze_(dim=1)
#             if i == 0:
#                 feas = fea
#             else:
#                 feas = torch.cat([feas, fea], dim=1)
#         fea_U = torch.sum(feas, dim=1)
#         # fea_s = self.gap(fea_U).squeeze_()
#         fea_s = fea_U.mean(-1).mean(-1)
#         fea_z = self.fc(fea_s)
#         for i, fc in enumerate(self.fcs):
#             vector = fc(fea_z).unsqueeze_(dim=1)
#             if i == 0:
#                 attention_vectors = vector
#             else:
#                 attention_vectors = torch.cat([attention_vectors, vector], dim=1)
#         attention_vectors = self.softmax(attention_vectors)
#         attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
#         fea_v = (feas * attention_vectors).sum(dim=1)
#         return fea_v
#
# class SKcnn(nn.Module):
#     def __init__(self, sec_div_num, last_div_num, classes):
#         super(SKcnn, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 64, 3, 2, 1),
#             SKConv(64, 32, 2, 8, 2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#
#             nn.Conv2d(64, 128, 3, 2, 1),
#             SKConv(128, 32, 2, 8, 2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#
#             nn.Conv2d(128, 256, 3, 2, 1),
#             SKConv(256, 32, 2, 8, 2),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#
#             nn.Conv2d(256, 512, 3, 2, 1),
#             SKConv(512, 32, 2, 8, 2),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )
#         self.avgpool = nn.AvgPool2d((4, 1))
#         self.flc = nn.Linear(512*sec_div_num*last_div_num, classes)
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.flc(x)
#         return x
#
# class SKUnit(nn.Module):
#     def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
#         """ Constructor
#         Args:
#             in_features: input channel dimensionality.
#             out_features: output channel dimensionality.
#             WH: input spatial dimensionality, used for GAP kernel size.
#             M: the number of branchs.
#             G: num of convolution groups.
#             r: the radio for compute d, the length of z.
#             mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
#             stride: stride.
#             L: the minimum dim of the vector z in paper.
#         """
#         super(SKUnit, self).__init__()
#         if mid_features is None:
#             mid_features = int(out_features / 2)
#         self.feas = nn.Sequential(
#             nn.Conv2d(in_features, mid_features, 3, stride, 1),
#             nn.BatchNorm2d(mid_features),
#             SKConv(mid_features, WH, M, G, r, stride=1, L=L),
#             nn.BatchNorm2d(mid_features),
#             nn.Conv2d(mid_features, out_features, 3, 1, 1),
#             nn.BatchNorm2d(out_features)
#         )
#         self.short = nn.Sequential()
#         if stride != 1 or in_features != out_features:
#             self.short = nn.Sequential(
#                 nn.Conv2d(in_features, out_features, 3, stride, 1),
#                 nn.BatchNorm2d(out_features)
#             )
#     def forward(self, x):
#         fea = self.feas(x)
#         return nn.ReLU()(fea + self.short(x))
#
#
# class SKNet(nn.Module):
#     def __init__(self, sec, last, class_num):
#         super(SKNet, self).__init__()
#         self.basic_conv = nn.Sequential(
#             nn.Conv2d(1, 64, 3, 2, 1),
#             nn.BatchNorm2d(64)
#         )  # 32x32
#         self.layer = nn.Sequential(
#             SKUnit(64, 128, 32, 2, 8, 2, stride=2),
#             SKUnit(128, 128, 32, 2, 8, 2, stride=1),
#
#             SKUnit(128, 256, 32, 2, 8, 2, stride=2),
#             SKUnit(256, 256, 32, 2, 8, 2, stride=1),
#
#             SKUnit(256, 512, 32, 2, 8, 2, stride=2),
#             SKUnit(512, 512, 32, 2, 8, 2, stride=1),
#             nn.AvgPool2d((4, 1))
#         )
#         self.flc = nn.Linear(512*sec*last, class_num)
#
#     def forward(self, x):
#         x = self.basic_conv(x)
#         x = self.layer(x)
#         x = x.view(x.size(0), -1)
#         x = self.flc(x)
#         return x

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