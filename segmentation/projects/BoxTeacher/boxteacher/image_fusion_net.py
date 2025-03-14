import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["ImageFusionNet"]

import torch

class ImageFusionNet(nn.Module):
    """
    融合两张图片
    """
    def __init__(self, input_channels=3, feature_dim=64):
        super(ImageFusionNet, self).__init__()
        # 定义两个独立的卷积层，用于提取两张图片的特征
        self.conv1_img1 = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv1_img2 = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 定义融合后的卷积网络（全卷积）
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 2, input_channels, kernel_size=3, padding=1),  # 输出3通道
        )

        # 上采样层
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, img1, img2):
        assert img1.shape == img2.shape
        # 分别通过两个卷积层提取特征
        feature1 = self.conv1_img1(img1)
        feature2 = self.conv1_img2(img2)

        # 拼接特征 (在通道维度上)
        fused_feature = torch.cat((feature1, feature2), dim=1)

        # 通过全卷积网络将特征融合
        output = self.fusion_layer(fused_feature)

        # 上采样到原始分辨率
        # output = self.upsample(fused_feature)

        return output