import os
import cv2
import torch
import argparse

from detectron2.projects.point_rend import add_pointrend_config

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from projects.PointSup.point_sup import add_point_sup_config


def main(args):
    # 配置模型
    cfg = get_cfg()
    add_pointrend_config(cfg)
    add_point_sup_config(cfg)
    cfg.merge_from_file(args.config_file)  # 从命令行传入的配置文件路径
    cfg.MODEL.WEIGHTS = os.path.join(args.weights_dir, "model_final.pth")  # 权重文件路径
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置检测的阈值

    # 创建模型预测器
    predictor = DefaultPredictor(cfg)

    # 读取并推理图片文件夹中的所有图片
    for image_name in os.listdir(args.input_dir):
        image_path = os.path.join(args.input_dir, image_name)
        if not os.path.isfile(image_path):
            continue

        # 读取图像
        image = cv2.imread(image_path)

        # 进行推理
        outputs = predictor(image)

        # 可视化预测结果
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)  # 转换为RGB格式
        out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

        # 保存或显示结果
        output_image_path = os.path.join(args.output_dir, f"pred_{image_name}")
        cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])  # 保存预测结果
        print(f"Saved prediction to {output_image_path}")

        # 如果需要显示图片
        # cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
        # cv2.waitKey(0)

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Run inference with a PointSup model in Detectron2")

    parser.add_argument("--config_file", required=True, help="Path to the model configuration file (e.g. config.yaml)")
    parser.add_argument("--weights_dir", required=True,help="Path to the folder containing the model weights (model_final.pth)")
    parser.add_argument("--input_dir", required=True, help="Path to the folder containing the input images")
    parser.add_argument("--output_dir",required=True,help="Path to the folder where predictions will be saved")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数进行推理
    main(args)
