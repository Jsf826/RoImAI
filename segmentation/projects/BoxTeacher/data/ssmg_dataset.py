import json
import os
from tqdm import tqdm
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode

__all__ = ["register_ssmg_dataset"]


class SSMGDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_image_dir = os.path.join(root_dir, 'train2017')
        self.val_image_dir = os.path.join(root_dir, 'val2017')
        self.train_json_file = os.path.join(root_dir, 'annotations', 'instances_train2017.json')
        self.val_json_file = os.path.join(root_dir, 'annotations', 'instances_val2017.json')

    def register_ssmg_coco(self, train_name, val_name):
        """
        Register COCO dataset for SSMG
        """
        self._register(train_name, self.train_image_dir, self.train_json_file)
        self._register(val_name, self.val_image_dir, self.val_json_file)
        print("Register SSMG_COCO dataset successfully!")

    def _register(self, dataset_name, image_dir, json_file):
        register_coco_instances(dataset_name, {}, json_file, image_dir)
        MetadataCatalog.get(dataset_name).set(thing_classes=["particle"])


class MultiImageSSMGDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_image_dir = os.path.join(root_dir, 'train2017')
        self.val_image_dir = os.path.join(root_dir, 'val2017')
        self.train_json_file = os.path.join(root_dir, 'annotations', 'instances_train2017.json')
        self.val_json_file = os.path.join(root_dir, 'annotations', 'instances_val2017.json')

    def load_mul_coco_json(self, image_root, json_file):
        """
        加载 COCO 格式的 JSON 文件，返回每个样本的标注信息和两个图像的路径
        """
        with open(json_file) as f:
            coco = json.load(f)

        dataset_dicts = []
        for image_info in tqdm(coco['images'], desc="Load Mul COCO Dataset"):
            record = {}
            image_id = image_info['id']
            file_name = image_info['file_name']

            # 获取两个图像路径
            img1_path = os.path.join(image_root, file_name)  # 第一张图片为 xxx-.jpg
            img2_path = os.path.splitext(img1_path)[0][:-1] + "+.jpg"  # 第二张图片命名为 xxx+.jpg`

            # 记录图像信息
            record["file_name1"] = img1_path
            record["file_name2"] = img2_path  # 第二张图片路径
            record["image_id"] = image_id
            record["height"] = image_info["height"]
            record["width"] = image_info["width"]

            # 获取图像标注
            annotations = []
            for ann in coco['annotations']:
                if ann['image_id'] == image_id:
                    ann_obj = {
                        "bbox": ann['bbox'],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        # "category_id": ann['category_id'],
                        "category_id": 0,
                        "iscrowd": ann['iscrowd'],
                    }
                    annotations.append(ann_obj)

            record["annotations"] = annotations
            dataset_dicts.append(record)

        return dataset_dicts

    def _register(self, dataset_name, image_dir, json_file):
        DatasetCatalog.register(
            dataset_name,
            lambda: self.load_mul_coco_json(image_dir, json_file)
        )
        MetadataCatalog.get(dataset_name).set(thing_classes=["particle"], evaluator_type="coco")

    # 注册多图片自定义数据集
    def register_mul_ssmg_coco(self, train_name, val_name):
        self._register(train_name, self.train_image_dir, self.train_json_file)
        self._register(val_name, self.val_image_dir, self.val_json_file)
        print("Register MUL_SSMG_COCO dataset successfully!")


# 注册数据集
def register_ssmg_dataset(cfg):
    # 注册自定义数据集
    if cfg.ROCK_SEG.IS_FUSION:
        DATASET_DIR = cfg.ROCK_SEG.MUL_SSMG_DATASET_DIR
        ssmg = MultiImageSSMGDataset(DATASET_DIR)
        ssmg.register_mul_ssmg_coco(cfg.DATASETS.TRAIN[0], cfg.DATASETS.TEST[0])
    else:
        DATASET_DIR = cfg.ROCK_SEG.SSMG_DATASET_DIR
        ssmg = SSMGDataset(DATASET_DIR)
        ssmg.register_ssmg_coco(cfg.DATASETS.TRAIN[0], cfg.DATASETS.TEST[0])
