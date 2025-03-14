import os
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from skimage.measure import label, regionprops, find_contours
from sklearn.utils import shuffle
from .metrics import precision, recall, F2, dice_score, jac_score
from sklearn.metrics import accuracy_score

""" Seeding the randomness. """


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


""" Create a directory """


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


""" Shuffle the dataset. """


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")


""" Convert a mask to border image """


def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border


""" Mask to bounding boxes """


def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes


def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]


def json2mask(image_name, coco_json):
    """
        输入：图像名称和 COCO 格式的实例分割数据（coco_json），返回对应的掩码。

        参数：
            image_name (str): 图像文件名（如 '1-.jpg'）。
            coco_json (dict): COCO 格式的标注数据，通常是通过读取 JSON 文件得到的。

        返回：
            mask (numpy.ndarray): 与输入图像对应的实例分割掩码。
    """

    # 加载 COCO 数据集的注释
    annotations = coco_json['annotations']
    images = coco_json['images']

    # 查找对应图像的 ID
    image_info = next(img for img in images if img['file_name'] == image_name)
    image_id = image_info['id']

    # 筛选出与该图像相关的标注
    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

    # 初始化一个空白掩码（假设图像大小为 256x256，可以根据实际情况修改）
    width, height = image_info['width'], image_info['height']
    mask = np.zeros((height, width), dtype=np.uint8)

    # 遍历每个标注，生成多边形掩码并叠加到总掩码上
    for ann in image_annotations:
        # 获取每个实例的多边形（polygon）坐标
        segmentation = ann['segmentation']
        if isinstance(segmentation, list):  # 如果是多边形列表形式
            for polygon in segmentation:
                # 使用 pycocotools 的方法将多边形转为掩码
                poly_mask = np.zeros((height, width), dtype=np.uint8)
                pts = np.array(polygon).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(poly_mask, [pts], 1)
                mask = np.maximum(mask, poly_mask)  # 叠加掩码

    return mask


def resize_shortest_edge_and_pad(image, target_size):
    """
    将图片的最短边缩放到目标尺寸，并将图片填充为正方形。

    参数:
    - image: 输入图片 (numpy array, H x W x C)
    - target_size: 目标尺寸 (int)

    返回:
    - resized_padded_image: 缩放并填充后的图片 (numpy array, target_size x target_size x C)
    """
    h, w = image.shape[:2]
    aspect_ratio = w / h

    # 确定最短边
    if h < w:
        new_h = target_size
        new_w = int(target_size * aspect_ratio)
    else:
        new_w = target_size
        new_h = int(target_size / aspect_ratio)

    # 等比例缩放
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算填充
    pad_h = max(target_size - new_h, 0)
    pad_w = max(new_w - target_size, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # 填充图片
    resized_padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return resized_padded_image


def apply_same_augmentation(image1, image2, mask, target_sizes=[640, 672, 704, 736, 768, 800], flip_prob=0.5):
    """
    对两张图片应用相同的数据增强结果，包括随机选择目标尺寸、ResizeShortestEdge 和随机翻转。

    参数:
    - image1: 第一张图片 (numpy array)
    - image2: 第二张图片 (numpy array)
    - target_sizes: 目标尺寸列表 (默认: [640, 672, 704, 736, 768, 800])
    - flip_prob: 随机翻转的概率 (默认: 0.5)

    返回:
    - augmented_image1: 增强后的第一张图片
    - augmented_image2: 增强后的第二张图片
    """
    # 随机选择一个目标尺寸
    target_size = random.choice(target_sizes)

    # 对两张图片应用相同的 ResizeShortestEdge 和填充
    resized_padded_image1 = resize_shortest_edge_and_pad(image1, target_size)
    resized_padded_image2 = resize_shortest_edge_and_pad(image2, target_size)
    resized_padded_mask = resize_shortest_edge_and_pad(mask, target_size)

    # 定义随机翻转操作
    transform = A.Compose([
        A.HorizontalFlip(p=flip_prob),  # 随机水平翻转
        A.VerticalFlip(p=flip_prob),
        ToTensorV2()
    ],additional_targets={'image2': 'image'})
    # 应用增强操作
    augmented = transform(image=resized_padded_image1,image2=resized_padded_image2, mask=resized_padded_mask)
    # 获取增强后的图片
    augmented_image1 = augmented['image']
    augmented_image2 = augmented['image2']
    augmented_mask = augmented['mask']

    return augmented_image1, augmented_image2, augmented_mask
