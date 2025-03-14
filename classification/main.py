import os

from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torchvision import transforms, models
import torch.hub
import argparse
from torch.optim import lr_scheduler
import albumentations as A

from RFM import HIFD2, MUL_HIFD2

from tree_loss import TreeLoss
from dataset import CubDataset, CubDataset2, AirDataset, AirDataset2, SSMG_Dataset, Multi_Image_SSMG_Dataset
from train_test import test, test_AP
from train_test import train, mul_train


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch Deployment')
    parser.add_argument('--worker', default=4, type=int, help='number of workers')
    parser.add_argument('--model', type=str, default='./pre-trained/resnet50-19c8e357.pth',
                        help='Path of trained model')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--proportion', default=0.1, type=float, help='Proportion of species label')
    parser.add_argument('--epoch', type=int, help='Epochs')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--dataset', type=str, default='Mul_SSMG', help='dataset name')
    parser.add_argument('--img_size', type=str, default='448', help='image size')
    parser.add_argument('--lr_adjt', type=str, default='Cos', help='Learning rate schedual')
    parser.add_argument('--device', nargs='+', default='0', help='GPU IDs for DP training')

    args = parser.parse_args()

    if args.proportion == 0.1:
        args.epoch = 130
        args.batch = 8
        args.lr_adjt = 'Step'

    return args


if __name__ == '__main__':
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    args = arg_parse()
    print('==> proportion: ', args.proportion)
    print('==> epoch: ', args.epoch)
    print('==> batch: ', args.batch)
    print('==> dataset: ', args.dataset)
    print('==> img_size: ', args.img_size)
    print('==> device: ', args.device)
    print('==> lr_adjt: ', args.lr_adjt)

    # Hyper-parameters
    nb_epoch = args.epoch
    batch_size = args.batch
    num_workers = args.worker

    # Preprocess
    # transform_train = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.RandomCrop(448, padding=8),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    train_transforms = A.Compose([
        # A.Resize(500, 500),
        A.Resize(448, 448),
        # A.RandomCrop(448, 448, pad_if_needed=True),
        A.HorizontalFlip(p=0.5),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_transforms = A.Compose([
        # A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # A.RandomCrop(448, 448, pad_if_needed=True),
        A.Resize(448, 448),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_transforms_mul = A.Compose([
        # A.Resize(500, 500),
        A.Resize(448, 448),
        # A.RandomCrop(448, 448, pad_if_needed=True),
        A.HorizontalFlip(p=0.5),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'image2': 'image'})

    test_transforms_mul = A.Compose([
        # A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # A.RandomCrop(448, 448, pad_if_needed=True),
        A.Resize(448, 448),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'image2': 'image'})



    # trees = [
    #     [51, 11, 47],
    #     [52, 11, 47],
    #     [53, 11, 47],
    #     [54, 5, 21],
    #     [55, 3, 16],
    #     [56, 3, 16],
    #     [57, 3, 16],
    #     [58, 3, 16],
    #     [59, 7, 30],
    #     [60, 7, 30],
    #     [61, 7, 30],
    #     [62, 7, 30],
    #     [63, 7, 30],
    #     [64, 7, 25],
    #     [65, 7, 25],
    #     [66, 7, 25],
    #     [67, 7, 25],
    #     [68, 7, 38],
    #     [69, 7, 33],
    #     [70, 7, 31],
    #     [71, 7, 36],
    #     [72, 2, 15],
    #     [73, 12, 49],
    #     [74, 12, 49],
    #     [75, 12, 49],
    #     [76, 7, 30],
    #     [77, 7, 30],
    #     [78, 7, 26],
    #     [79, 7, 27],
    #     [80, 7, 27],
    #     [81, 5, 21],
    #     [82, 5, 21],
    #     [83, 5, 21],
    #     [84, 7, 28],
    #     [85, 7, 28],
    #     [86, 9, 45],
    #     [87, 7, 42],
    #     [88, 7, 42],
    #     [89, 7, 42],
    #     [90, 7, 42],
    #     [91, 7, 42],
    #     [92, 7, 42],
    #     [93, 7, 42],
    #     [94, 12, 50],
    #     [95, 11, 48],
    #     [96, 0, 13],
    #     [97, 7, 28],
    #     [98, 7, 28],
    #     [99, 7, 30],
    #     [100, 10, 46],
    #     [101, 10, 46],
    #     [102, 10, 46],
    #     [103, 10, 46],
    #     [104, 7, 25],
    #     [105, 7, 28],
    #     [106, 7, 28],
    #     [107, 7, 25],
    #     [108, 3, 16],
    #     [109, 3, 17],
    #     [110, 3, 17],
    #     [111, 3, 17],
    #     [112, 3, 17],
    #     [113, 3, 17],
    #     [114, 3, 17],
    #     [115, 3, 17],
    #     [116, 3, 17],
    #     [117, 1, 14],
    #     [118, 1, 14],
    #     [119, 1, 14],
    #     [120, 1, 14],
    #     [121, 3, 18],
    #     [122, 3, 18],
    #     [123, 7, 27],
    #     [124, 7, 27],
    #     [125, 7, 27],
    #     [126, 7, 36],
    #     [127, 7, 42],
    #     [128, 7, 42],
    #     [129, 4, 19],
    #     [130, 4, 19],
    #     [131, 4, 19],
    #     [132, 4, 19],
    #     [133, 4, 19],
    #     [134, 4, 20],
    #     [135, 7, 23],
    #     [136, 6, 22],
    #     [137, 0, 13],
    #     [138, 7, 30],
    #     [139, 0, 13],
    #     [140, 0, 13],
    #     [141, 7, 33],
    #     [142, 2, 15],
    #     [143, 7, 27],
    #     [144, 7, 39],
    #     [145, 7, 30],
    #     [146, 7, 30],
    #     [147, 7, 30],
    #     [148, 7, 30],
    #     [149, 7, 35],
    #     [150, 8, 44],
    #     [151, 8, 44],
    #     [152, 7, 42],
    #     [153, 7, 42],
    #     [154, 7, 34],
    #     [155, 2, 15],
    #     [156, 3, 16],
    #     [157, 7, 27],
    #     [158, 7, 27],
    #     [159, 7, 35],
    #     [160, 5, 21],
    #     [161, 7, 32],
    #     [162, 7, 32],
    #     [163, 7, 36],
    #     [164, 7, 36],
    #     [165, 7, 36],
    #     [166, 7, 36],
    #     [167, 7, 36],
    #     [168, 7, 37],
    #     [169, 7, 36],
    #     [170, 7, 36],
    #     [171, 7, 36],
    #     [172, 7, 36],
    #     [173, 7, 36],
    #     [174, 7, 36],
    #     [175, 7, 36],
    #     [176, 7, 36],
    #     [177, 7, 36],
    #     [178, 7, 36],
    #     [179, 7, 36],
    #     [180, 7, 36],
    #     [181, 7, 36],
    #     [182, 7, 36],
    #     [183, 7, 36],
    #     [184, 7, 40],
    #     [185, 7, 29],
    #     [186, 7, 29],
    #     [187, 7, 29],
    #     [188, 7, 29],
    #     [189, 7, 25],
    #     [190, 7, 25],
    #     [191, 3, 17],
    #     [192, 3, 17],
    #     [193, 3, 17],
    #     [194, 3, 17],
    #     [195, 3, 17],
    #     [196, 3, 17],
    #     [197, 3, 17],
    #     [198, 7, 36],
    #     [199, 7, 33],
    #     [200, 7, 33],
    #     [201, 7, 43],
    #     [202, 7, 43],
    #     [203, 7, 43],
    #     [204, 7, 43],
    #     [205, 7, 43],
    #     [206, 7, 43],
    #     [207, 7, 43],
    #     [208, 7, 35],
    #     [209, 7, 35],
    #     [210, 7, 35],
    #     [211, 7, 35],
    #     [212, 7, 35],
    #     [213, 7, 35],
    #     [214, 7, 35],
    #     [215, 7, 35],
    #     [216, 7, 35],
    #     [217, 7, 35],
    #     [218, 7, 35],
    #     [219, 7, 35],
    #     [220, 7, 35],
    #     [221, 7, 35],
    #     [222, 7, 35],
    #     [223, 7, 35],
    #     [224, 7, 35],
    #     [225, 7, 35],
    #     [226, 7, 35],
    #     [227, 7, 35],
    #     [228, 7, 35],
    #     [229, 7, 35],
    #     [230, 7, 35],
    #     [231, 7, 35],
    #     [232, 7, 35],
    #     [233, 7, 35],
    #     [234, 7, 35],
    #     [235, 7, 24],
    #     [236, 7, 24],
    #     [237, 9, 45],
    #     [238, 9, 45],
    #     [239, 9, 45],
    #     [240, 9, 45],
    #     [241, 9, 45],
    #     [242, 9, 45],
    #     [243, 7, 41],
    #     [244, 7, 41],
    #     [245, 7, 41],
    #     [246, 7, 41],
    #     [247, 7, 41],
    #     [248, 7, 41],
    #     [249, 7, 41],
    #     [250, 7, 35]
    # ]
    trees = [
        # [0, 0],
        # [1, 1],
        # [2, 2],
        [3, 0],
        [4, 0],
        [5, 1],
        [6, 1],
        [7, 2],
        [8, 2],
        [9, 2],
        [10, 2],
        [11, 2],
    ]
    # trees = [
    #     [1, 1],
    #     [2, 2],
    #     [3, 3],
    #     [4, 1],
    #     [5, 1],
    #     [6, 2],
    #     [7, 2],
    #     [8, 3],
    #     [9, 3],
    #     [10, 3],
    #     [11, 3],
    #     [12, 3],
    # ]
    levels = 2
    total_nodes = 12
    train_dir = '/home/pan/FJS/MC/HRN-main/Datasets/second_data/train'
    test_dir = '/home/pan/FJS/MC/HRN-main/Datasets/second_data/val'
    # cls2index = {
    #     '石英类': 0,
    #     '长石': 1,
    #     '岩屑': 2,
    # }
    if args.dataset == 'Mul_SSMG':
        trainset = Multi_Image_SSMG_Dataset(image_dir=train_dir, transforms=train_transforms_mul)
        testset = Multi_Image_SSMG_Dataset(image_dir=test_dir, transforms=test_transforms_mul)

    else:
        trainset = SSMG_Dataset(image_dir=train_dir, transforms=train_transforms)
        testset = SSMG_Dataset(image_dir=test_dir, transforms=test_transforms)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                             drop_last=True)

    # GPU
    device = torch.device("cuda:" + args.device[0])

    # RFM from scrach
    backbone = models.resnet50(pretrained=False)

    backbone.load_state_dict(torch.load('pre-trained/resnet50-19c8e357.pth'))
    # backbone = models.resnext101_32x8d(pretrained=False)
    # backbone.load_state_dict(torch.load('./pre-trained/resnext101_32x8d-8ba56ff5.pth'))

    if args.dataset == "Mul_SSMG":
        net = MUL_HIFD2(backbone, 1024, args.dataset)
    else:
        net = HIFD2(backbone, 1024, args.dataset)

    # RFM from trained model
    # net = torch.load(args.model)

    net.to(device)

    # Loss functions
    CELoss = nn.CrossEntropyLoss()
    tree = TreeLoss(trees, total_nodes, levels, device)

    if args.proportion > 0.1:  # for p > 0.1
        optimizer = optim.SGD([
            {'params': net.classifier_1.parameters(), 'lr': 0.002},
            {'params': net.classifier_2.parameters(), 'lr': 0.002},
            {'params': net.classifier_3.parameters(), 'lr': 0.002},
            {'params': net.classifier_3_1.parameters(), 'lr': 0.002},
            {'params': net.fc1.parameters(), 'lr': 0.002},
            {'params': net.fc2.parameters(), 'lr': 0.002},
            {'params': net.fc3.parameters(), 'lr': 0.002},
            {'params': net.conv_block1.parameters(), 'lr': 0.002},
            {'params': net.conv_block2.parameters(), 'lr': 0.002},
            {'params': net.conv_block3.parameters(), 'lr': 0.002},
            {'params': net.features.parameters(), 'lr': 0.0002}
        ],
            momentum=0.9, weight_decay=5e-4)

    else:  # for p = 0.1
        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-3)
        # optimizer = optim.AdamW(net.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    save_name = args.dataset + '_' + str(args.epoch) + '_' + str(args.img_size) + '_p' + str(
        args.proportion) + '_bz' + str(args.batch) + '_ResNet-50_' + '_' + args.lr_adjt

    if args.dataset == 'Mul_SSMG':
        mul_train(nb_epoch, net, trainloader, testloader, optimizer, scheduler, args.lr_adjt, args.dataset, CELoss, tree,
              device, args.device, save_name)
    else:
        train(nb_epoch, net, trainloader, testloader, optimizer, scheduler, args.lr_adjt, args.dataset, CELoss, tree,
              device, args.device, save_name)

    # Evaluate OA
    # test(net, testloader, CELoss, tree, device, args.dataset)

    # Evaluate Average PRC
    # test_AP(net, args.dataset, testset, testloader, device)
