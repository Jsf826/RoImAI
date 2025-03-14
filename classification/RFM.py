import torch.nn as nn
import torch
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # Post-activation
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        # x = self.conv(x)
        return x


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


class MUL_HIFD2(nn.Module):
    def __init__(self, model, feature_size, dataset):
        super(MUL_HIFD2, self).__init__()

        self.features = nn.Sequential(*list(model.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.num_ftrs = 2048 * 1 * 1

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        if dataset == 'CUB':
            self.classifier_1 = nn.Sequential(
                nn.Linear(512, 13),
                nn.Sigmoid()
            )
            self.classifier_2 = nn.Sequential(
                nn.Linear(512, 38),
                nn.Sigmoid()
            )
            self.classifier_3 = nn.Sequential(
                nn.Linear(512, 200),
                nn.Sigmoid()
            )
            self.classifier_3_1 = nn.Sequential(
                nn.Linear(512, 200)
            )
        elif dataset == 'Air':
            self.classifier_1 = nn.Sequential(
                nn.Linear(512, 30),
                nn.Sigmoid()
            )
            self.classifier_2 = nn.Sequential(
                nn.Linear(512, 70),
                nn.Sigmoid()
            )
            self.classifier_3 = nn.Sequential(
                nn.Linear(512, 100),
                nn.Sigmoid()
            )
            self.classifier_3_1 = nn.Sequential(
                nn.Linear(512, 100)
            )
        if dataset == 'SSMG' or dataset == 'Mul_SSMG':
            self.classifier_1 = nn.Sequential(
                nn.Linear(512, 3),
                nn.Sigmoid()
            )
            self.classifier_2 = nn.Sequential(
                nn.Linear(512, 9),
                nn.Sigmoid()
            )
            self.classifier_2_1 = nn.Sequential(
                nn.Linear(512, 9),
                nn.Sigmoid()
            )

        self.image_fusion_layer = ImageFusionNet(input_channels=3, feature_dim=64)

    def forward(self, x1, x2):
        x = self.image_fusion_layer(x1, x2)
        x = self.features(x)
        # x_order = self.conv_block1(x)
        x_family = self.conv_block1(x)
        x_species = self.conv_block2(x)

        # x_order_fc = self.pooling(x_order)
        # x_order_fc = x_order_fc.view(x_order_fc.size(0), -1)
        # x_order_fc = self.fc1(x_order_fc)

        x_family_fc = self.pooling(x_family)
        x_family_fc = x_family_fc.view(x_family_fc.size(0), -1)
        x_family_fc = self.fc1(x_family_fc)
        x_species_fc = self.pooling(x_species)
        x_species_fc = x_species_fc.view(x_species_fc.size(0), -1)
        x_species_fc = self.fc2(x_species_fc)

        # y_order_sig = self.classifier_1(self.relu(x_order_fc))
        y_family_sig = self.classifier_1(self.relu(x_family_fc))
        y_species_sig = self.classifier_2(self.relu(x_species_fc + x_family_fc))
        y_species_sof = self.classifier_2_1(self.relu(x_species_fc + x_family_fc))

        # return y_order_sig, y_family_sig, y_species_sof, y_species_sig
        return y_family_sig, y_species_sof, y_species_sig


class HIFD2(nn.Module):
    def __init__(self, model, feature_size, dataset):
        super(HIFD2, self).__init__()

        self.features = nn.Sequential(*list(model.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.num_ftrs = 2048 * 1 * 1

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )


        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )


        if dataset == 'CUB':
            self.classifier_1 = nn.Sequential(
                nn.Linear(512, 13),
                nn.Sigmoid()
            )
            self.classifier_2 = nn.Sequential(
                nn.Linear(512, 38),
                nn.Sigmoid()
            )
            self.classifier_3 = nn.Sequential(
                nn.Linear(512, 200),
                nn.Sigmoid()
            )
            self.classifier_3_1 = nn.Sequential(
                nn.Linear(512, 200)
            )
        elif dataset == 'Air':
            self.classifier_1 = nn.Sequential(
                nn.Linear(512, 30),
                nn.Sigmoid()
            )
            self.classifier_2 = nn.Sequential(
                nn.Linear(512, 70),
                nn.Sigmoid()
            )
            self.classifier_3 = nn.Sequential(
                nn.Linear(512, 100),
                nn.Sigmoid()
            )
            self.classifier_3_1 = nn.Sequential(
                nn.Linear(512, 100)
            )
        if dataset == 'SSMG' or dataset == 'Mul_SSMG':
            self.classifier_1 = nn.Sequential(
                nn.Linear(512, 3),
                nn.Sigmoid()
            )
            self.classifier_2 = nn.Sequential(
                nn.Linear(512, 9),
                nn.Sigmoid()
            )
            self.classifier_2_1 = nn.Sequential(
                nn.Linear(512, 9),
                nn.Sigmoid()
            )


    def forward(self, x):
        x = self.features(x)
        # x_order = self.conv_block1(x)
        x_family = self.conv_block1(x)
        x_species = self.conv_block2(x)

        # x_order_fc = self.pooling(x_order)
        # x_order_fc = x_order_fc.view(x_order_fc.size(0), -1)
        # x_order_fc = self.fc1(x_order_fc)

        x_family_fc = self.pooling(x_family)
        x_family_fc = x_family_fc.view(x_family_fc.size(0), -1)
        x_family_fc = self.fc1(x_family_fc)
        x_species_fc = self.pooling(x_species)
        x_species_fc = x_species_fc.view(x_species_fc.size(0), -1)
        x_species_fc = self.fc2(x_species_fc)

        # y_order_sig = self.classifier_1(self.relu(x_order_fc))
        y_family_sig = self.classifier_1(self.relu(x_family_fc))
        y_species_sig = self.classifier_2(self.relu(x_species_fc + x_family_fc))
        y_species_sof = self.classifier_2_1(self.relu(x_species_fc + x_family_fc))

        # return y_order_sig, y_family_sig, y_species_sof, y_species_sig
        return y_family_sig, y_species_sof, y_species_sig

