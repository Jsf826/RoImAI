import os

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer


def draw_dataset_sampler(data: DataLoader, num=8):
    sample_list = []
    for i, batch_data in enumerate(data):
        if i >= num:
            break
        sample = batch_data[0]
        # 获取图像张量和分割掩膜
        image_tensor1 = sample["aug_image1"]  # 假设这里是一个 (C, H, W) 的张量
        image_tensor2 = sample["aug_image2"]  # 假设这里是一个 (C, H, W) 的张量
        instances = sample["instances"]  # Instances 对象，包含实例信息  # 分割掩膜，可能是 RLE 或者多边形
        gt_boxes = instances.gt_boxes
        # 将图像张量转换为 numpy 数组
        image1 = image_tensor1.permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)
        image2 = image_tensor2.permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)
        image1 = np.clip(image1, 0, 255).astype(np.uint8)  # 保证图像范围在 [0, 255] 之间
        image2 = np.clip(image2, 0, 255).astype(np.uint8)  # 保证图像范围在 [0, 255] 之间

        # 获取数据集的元数据（比如类别标签、颜色等）
        metadata = MetadataCatalog.get("ssmg_coco_train")  # 替换为你的数据集名称

        # 使用 Visualizer 绘制图像和标注
        visualizer = Visualizer(image1[:, :, ::-1], metadata=metadata, scale=1)  # BGR 转 RGB
        # output = visualizer.draw_instance_predictions(instances.to("cpu"))
        output = visualizer.overlay_instances(boxes=gt_boxes)

        # 显示图像

        sample_list.append((image1, image2, output.get_image()[:, :, ::-1]))
    return sample_list
