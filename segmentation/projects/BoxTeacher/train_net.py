# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you toz use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import copy
from typing import List, Any, Dict, OrderedDict, Set
import logging
import os
import itertools
import collections
import cv2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from torch.nn.parallel import DistributedDataParallel
from rock_seg.utils import draw_dataset_sampler

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results, CityscapesInstanceEvaluator,
)
from detectron2.modeling import GeneralizedRCNNWithTTA, build_model
from detectron2.utils.logger import setup_logger

from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer

from boxteacher import (
    BoxTeacherEMAHook,
    add_box_teacher_config,
    AugmentDatasetMapper
)
from rock_seg import (
    add_rock_seg_config,
    AugmentMulDatasetMapper
)

# from cityscapes_eval import CityscapesInstanceEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from data import register_ssmg_dataset


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = super().build_model(cfg)
        # 加载backbone权重
        if cfg.IS_PRETRAINED_BACKBONE:
            backbone_state_dict = torch.load(cfg.PRETRAINED_BACKBONE_PATH)
            # 初始化两个空的OrderedDict
            # pre_image_fusion_layer_state_dict = collections.OrderedDict()
            # resnet_state_dict = collections.OrderedDict()

            # 遍历训练权重，将权重按前缀分配
            # for name, param in backbone_state_dict.items():
            #     if name.startswith('pre_image_fusion_layer'):
            #         pre_image_fusion_layer_state_dict[name] = param
            #     elif name.startswith('layer'):
            #         resnet_state_dict[name] = param
            #     else:
            #         print("Warning: Unknown layer name! ", name)
            name_mapper = {
                "teacher.pre_image_fusion_layer":"pre_image_fusion_layer",
                "teacher.backbone.bottom_up.stem.conv1":"layer0.0",
                "teacher.backbone.bottom_up.res2":"layer1",
                "teacher.backbone.bottom_up.res3":"layer2",
                "teacher.backbone.bottom_up.res4":"layer3",

                "student.pre_image_fusion_layer":"pre_image_fusion_layer",
                "student.backbone.bottom_up.stem.conv1":"layer0.0",
                "student.backbone.bottom_up.res2":"layer1",
                "student.backbone.bottom_up.res3":"layer2",
                "student.backbone.bottom_up.res4":"layer3",
            }

            keys = name_mapper.keys()
            names = []
            for name, param in model.named_parameters():
                # 加载权重
                for key in keys:
                    if name.startswith(key):
                        backbone_name = name.replace(key, name_mapper[key])
                        if backbone_name in backbone_state_dict.keys():
                            param.data = copy.deepcopy(backbone_state_dict[backbone_name])
                            names.append(name)


            # print(f"已加载权重数量: {len(names)}\n已加载权重: {names}")
        # missing_keys, unexpected_keys = model.load_state_dict(backbone_state_dict, strict=False)
        return model

    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(
                    self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)

        ret.append(
            BoxTeacherEMAHook(
                momentum=self.cfg.MODEL.BOX_TEACHER.MOMENTUM,
            )
        )

        return ret

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        if cfg.ROCK_SEG.IS_FUSION:
            # 多模态
            mapper = AugmentMulDatasetMapper(cfg, True)
        else:
            # 单模态
            mapper = AugmentDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name, output_folder)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict[{k + "_TTA": v for k, v in res.items()}]
        return res

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_box_teacher_config(cfg)
    add_rock_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet", color=False)

    return cfg


def main(args):
    cfg = setup(args)

    # 注册数据集
    register_ssmg_dataset(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)  # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    draw_sample(cfg, trainer)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def draw_sample(cfg, trainer, num=1):
    """
        绘制数据样本
        :param num: 样本数量
        :return:
    """
    sample_dir = os.path.join(cfg.OUTPUT_DIR, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    data = trainer.data_loader
    sample_list = draw_dataset_sampler(data, num=num)
    for i, sample in enumerate(sample_list):
        image1, image2, mask = sample
        cv2.imwrite(os.path.join(sample_dir, f"image_1_{i}.png"), image1)
        cv2.imwrite(os.path.join(sample_dir, f"image_2_{i}.png"), image2)
        cv2.imwrite(os.path.join(sample_dir, f"mask{i}.png"), mask)


if __name__ == "__main__":
    # 训练 python train_net.py --config-file configs/coco/boxteacher_r50_1x.yaml --num-gpus 1
    # 验证 python train_net.py --config-file configs/coco/boxteacher_r50_1x.yaml --num-gpus 1 --eval MODEL.WEIGHTS output/boxteacher_r50_1x/model_0004999.pth
    # 修改rock_seg/config中的cfg.ROCK_SEG.IS_FUSION变量，即可更改单模态多模态
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
