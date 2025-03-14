import json
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from .utils import calculate_metrics, json2mask, apply_same_augmentation

#
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn.functional as F


def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path, "images", name) + ".jpg" for name in data]
    masks = [os.path.join(path, "masks", name) + ".jpg" for name in data]
    # images = [os.path.join(path, "images", name) + ".png" for name in data]
    # masks = [os.path.join(path, "masks", name) + ".png" for name in data]
    return images, masks


def load_data(path, val_name=None):
    train_names_path = f"{path}/train.txt"
    # valid_names_path = f"{path}/val.txt"
    if val_name is None:
        valid_names_path = f"{path}/val.txt"
    else:
        valid_names_path = f"{path}/val_{val_name}.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)  # 读取图像和掩码的路径

    return (train_x, train_y), (valid_x, valid_y)


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        # print("n_samples:", self.n_samples)
        # self.convert_edge=convert_edge
        self.size = size

    def __getitem__(self, index):
        """ Reading Image & Mask """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        background = mask.copy()
        background = 255 - background

        """ Applying Data Augmentation """
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, background=background)
            image = augmentations["image"]
            mask = augmentations["mask"]
            background = augmentations["background"]

        """ Image """
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        """ Mask """
        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        """ Background """
        background = cv2.resize(background, self.size)
        background = np.expand_dims(background, axis=0)
        background = background / 255.0

        return image, (mask, background)

    def __len__(self):
        return self.n_samples


class MULTI_IMAGE_DATASET(Dataset):
    def __init__(self, image_dir, json_file, size, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.json_file = json_file
        self.images_path = [file for file in os.listdir(self.image_dir) if file.endswith("-.jpg")]
        with open(json_file, "r") as f:
            self.json_data = json.load(f)
        self.transform = transform
        self.n_samples = len(self.images_path)
        # print("n_samples:", self.n_samples)
        # self.convert_edge=convert_edge
        self.size = size

    def __getitem__(self, index):
        """ Reading Image & Mask """
        img_path1 = os.path.join(self.image_dir, self.images_path[index])
        image1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
        image2 = cv2.imread(img_path1.replace('-', '+'), cv2.IMREAD_COLOR)
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        mask = json2mask(self.images_path[index], self.json_data)
        # background = mask.copy()
        # background = 255 - background

        """ Applying Data Augmentation """
        if self.transform is not None:
            # augmentations = apply_same_augmentation(image1, image2, mask,flip_prob=1)
            # image1, image2, mask = augmentations
            augmentations = self.transform(image=image1, image2=image2, mask=mask)
            image1 = augmentations["image"]
            image2 = augmentations["image2"]
            mask = augmentations["mask"]
            # background = augmentations["background"]

        """ Image """
        image1 = cv2.resize(image1, self.size)
        image1 = np.transpose(image1, (2, 0, 1))
        image1 = image1 / 255.0

        image2 = cv2.resize(image2, self.size)
        image2 = np.transpose(image2, (2, 0, 1))
        image2 = image2 / 255.0

        """ Mask """
        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        # mask = mask / 255.0

        """ Background """
        # background = cv2.resize(background, self.size)
        # background = np.expand_dims(background, axis=0)
        # background = background / 255.0

        # return image, (mask, background)
        return image1, image2, mask

    def __len__(self):
        return self.n_samples


class BinaryConsistencyLoss(nn.Module):
    def __init__(self):
        super(BinaryConsistencyLoss, self).__init__()

    def forward(self, mask1, mask2):
        mask1_binary = (mask1 > 0.5).float()
        mask2_binary = (mask2 > 0.5).float()

        loss1 = F.binary_cross_entropy(mask1, mask2_binary, reduction='mean')
        loss2 = F.binary_cross_entropy(mask2, mask1_binary, reduction='mean')

        loss = loss1 + loss2

        return loss


def train(model, loader, optimizer, loss_fn, device, consistency_loss_fn=BinaryConsistencyLoss()):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    augmentations = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        # transforms.RandomGrayscale(p=0.5),
        transforms.GaussianBlur(kernel_size=(1, 7), sigma=(0.1, 3))
    ])

    for i, (x1, x2, y1) in tqdm(enumerate(loader), desc="Training", total=len(loader)):

        x1 = x1.to(device, dtype=torch.float32)
        x2 = x2.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        mask_pred = model(x1, x2)

        # x_aug = augmentations(x)
        x_aug1 = augmentations(x1)
        x_aug2 = augmentations(x2)
        x_aug1 = x_aug1.to(device, dtype=torch.float32)
        x_aug2 = x_aug2.to(device, dtype=torch.float32)
        mask_pred_aug = model(x_aug1, x_aug2)

        loss_consistency = consistency_loss_fn(mask_pred, mask_pred_aug)

        loss_mask = loss_fn(mask_pred, y1)
        loss_mask_aug = loss_fn(mask_pred_aug, y1)

        loss = loss_mask + loss_mask_aug + loss_consistency

        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y1, mask_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss / len(loader)
    epoch_jac = epoch_jac / len(loader)
    epoch_f1 = epoch_f1 / len(loader)
    epoch_recall = epoch_recall / len(loader)
    epoch_precision = epoch_precision / len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x1, x2, y1) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x1 = x1.to(device, dtype=torch.float32)
            x2 = x2.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)

            mask_pred = model(x1, x2)

            loss_mask = loss_fn(mask_pred, y1)

            loss = loss_mask

            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y1, mask_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss / len(loader)
        epoch_jac = epoch_jac / len(loader)
        epoch_f1 = epoch_f1 / len(loader)
        epoch_recall = epoch_recall / len(loader)
        epoch_precision = epoch_precision / len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]
