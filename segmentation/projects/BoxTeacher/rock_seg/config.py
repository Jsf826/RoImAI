from detectron2.config import CfgNode as CN


def add_rock_seg_config(cfg):
    cfg.ROCK_SEG = CN()

    cfg.ROCK_SEG.SSMG_DATASET_DIR = r"datasets/ssmg_coco"  # 单模态数据

    cfg.ROCK_SEG.MUL_SSMG_DATASET_DIR = r"datasets/mul_ssmg_coco"  # 多模态数据

    cfg.ROCK_SEG.IS_FUSION = True  # 是否融合

    cfg.ROCK_SEG.FUSION_MODE = "pre"  # 融合模式

    cfg.IS_PRETRAINED_BACKBONE = True  # 是否加载预训练backbone权重

    cfg.PRETRAINED_BACKBONE_PATH = r"output/checkpoints/pretrained_backbone.pth"  # 预训练backbone权重路径
