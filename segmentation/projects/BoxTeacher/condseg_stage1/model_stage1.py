import math
import torch
import torch.nn as nn
# from resnet import resnet50
import torch.nn.functional as F
from detectron2.layers import ShapeSpec

from .resnet import resnet50
from .image_fusion_net import ImageFusionNet

from detectron2.modeling import build_model, build_backbone
import torch.nn.functional as F

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class adjust(nn.Module):
    def __init__(self, in_c1, in_c2, in_c3, in_c4, out_c):
        super().__init__()
        self.conv1 = CBR(in_c1, 64, kernel_size=1, padding=0, act=True)
        self.conv2 = CBR(in_c2, 64, kernel_size=1, padding=0, act=True)
        self.conv3 = CBR(in_c3, 64, kernel_size=1, padding=0, act=True)
        self.conv4 = CBR(in_c4, 64, kernel_size=1, padding=0, act=True)
        self.conv_fuse = nn.Conv2d(4 * 64, out_c, kernel_size=1, padding=0, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_fuse(x)
        x = self.sig(x)
        return x


class ConDSegStage1(nn.Module):
    def __init__(self, cfg, H=256, W=256):
        super().__init__()

        self.H = H
        self.W = W

        """ Backbone: ResNet50 """
        """从ResNet50中提取出layer0, layer1, layer2, layer3"""
        self.pre_image_fusion_layer = ImageFusionNet()
        backbone = build_backbone(cfg)
        # backbone = resnet50()
        # self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # [batch_size, 64, h/2, w/2]
        # self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)  # [batch_size, 256, h/4, w/4]
        # self.layer2 = backbone.layer2  # [batch_size, 512, h/8, w/8]
        # self.layer3 = backbone.layer3  # [batch_size, 1024, h/16, w/16]


        self.layer0 = nn.Sequential(backbone.bottom_up.stem.conv1)  # [batch_size, 64, h/2, w/2]
        self.layer1 = backbone.bottom_up.res2  # [batch_size, 256, h/4, w/4]
        self.layer2 = backbone.bottom_up.res3  # [batch_size, 512, h/8, w/8]
        self.layer3 = backbone.bottom_up.res4  # [batch_size, 1024, h/16, w/16]

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8x8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up_16x16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        self.head = adjust(64, 256, 512, 1024, 1)

    def forward(self, original_images1, original_images2):
        """ Backbone: ResNet50 """
        x0 = self.pre_image_fusion_layer(original_images1, original_images2)

        x1 = self.layer0(x0)  ## [-1, 64, h/2, w/2]
        x1 = F.relu_(x1)
        x1_ = F.max_pool2d(x1, kernel_size=3, stride=2, padding=1)
        x2 = self.layer1(x1_)  ## [-1, 256, h/4, w/4]
        x3 = self.layer2(x2)  ## [-1, 512, h/8, w/8]
        x4 = self.layer3(x3)  ## [-1, 1024, h/16, w/16]

        x1 = self.up_2x2(x1)
        x2 = self.up_4x4(x2)
        x3 = self.up_8x8(x3)
        x4 = self.up_16x16(x4)

        # pred
        pred = self.head(x1, x2, x3, x4)

        return pred


if __name__ == "__main__":
    model = ConDSegStage1().cuda()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    output = model(input_tensor)
    print(output.shape)
