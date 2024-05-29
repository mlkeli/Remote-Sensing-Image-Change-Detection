import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models



class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, lightweight):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4

        self.head = nn.Sequential(conv3x3(in_channels, inter_channels, lightweight),
                                  nn.BatchNorm2d(inter_channels),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False),
                                  nn.Conv2d(inter_channels, out_channels, 1, bias=True))

    def forward(self, x):
        return self.head(x)

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate=1):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=atrous_rate, groups=in_channels,
                      dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )

    def forward(self, x):
        return self.conv(x)
def conv3x3(in_channels, out_channels, lightweight, atrous_rate=1):
    if lightweight:
        return DSConv(in_channels, out_channels, atrous_rate)
    else:
        return nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False)


class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, lightweight):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4

        self.conv5 = nn.Sequential(PyramidPooling(in_channels),
                                   conv3x3(in_channels + 4 * int(in_channels / 4), inter_channels, lightweight),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


class PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode="bilinear", align_corners=False)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode="bilinear", align_corners=False)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)


class BaseNet(nn.Module):
    def __init__(self,nclass, lightweight):
        super(BaseNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        # in_channels = self.backbone.channels[-1]
        self.head = FCNHead(2048, 6, lightweight)
        self.head_bin = PSPHead(2048, 2, lightweight)

    def base_forward(self, x1, x2):
        b, c, h, w = x1.shape

        x1 = self.resnet(x1)
        x2 = self.resnet(x2)

        out1 = self.head(x1)
        out2 = self.head(x2)

        out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)

        out_bin = torch.abs(x1 - x2)
        out_bin = self.head_bin(out_bin)
        out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
        # out_bin = torch.sigmoid(out_bin)

        return out1, out2, out_bin

    def forward(self, x1, x2, tta=False):
        if not tta:
            return self.base_forward(x1, x2)
        else:
            out1, out2, out_bin = self.base_forward(x1, x2)
            out1 = F.softmax(out1, dim=1)
            out2 = F.softmax(out2, dim=1)
            out_bin = out_bin.unsqueeze(1)
            origin_x1 = x1.clone()
            origin_x2 = x2.clone()

            x1 = origin_x1.flip(2)
            x2 = origin_x2.flip(2)
            cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
            out1 += F.softmax(cur_out1, dim=1).flip(2)
            out2 += F.softmax(cur_out2, dim=1).flip(2)
            out_bin += cur_out_bin.unsqueeze(1).flip(2)

            x1 = origin_x1.flip(3)
            x2 = origin_x2.flip(3)
            cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
            out1 += F.softmax(cur_out1, dim=1).flip(3)
            out2 += F.softmax(cur_out2, dim=1).flip(3)
            out_bin += cur_out_bin.unsqueeze(1).flip(3)

            x1 = origin_x1.transpose(2, 3).flip(3)
            x2 = origin_x2.transpose(2, 3).flip(3)
            cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
            out1 += F.softmax(cur_out1, dim=1).flip(3).transpose(2, 3)
            out2 += F.softmax(cur_out2, dim=1).flip(3).transpose(2, 3)
            out_bin += cur_out_bin.unsqueeze(1).flip(3).transpose(2, 3)

            x1 = origin_x1.flip(3).transpose(2, 3)
            x2 = origin_x2.flip(3).transpose(2, 3)
            cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
            out1 += F.softmax(cur_out1, dim=1).transpose(2, 3).flip(3)
            out2 += F.softmax(cur_out2, dim=1).transpose(2, 3).flip(3)
            out_bin += cur_out_bin.unsqueeze(1).transpose(2, 3).flip(3)

            x1 = origin_x1.flip(2).flip(3)
            x2 = origin_x2.flip(2).flip(3)
            cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
            out1 += F.softmax(cur_out1, dim=1).flip(3).flip(2)
            out2 += F.softmax(cur_out2, dim=1).flip(3).flip(2)
            out_bin += cur_out_bin.unsqueeze(1).flip(3).flip(2)

            out1 /= 6.0
            out2 /= 6.0
            out_bin /= 6.0

            return out1, out2, out_bin.squeeze(1)