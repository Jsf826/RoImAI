# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from adet.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from multi_predictor import MultiImageVisualizationDemo
from projects.BoxTeacher.boxteacher import add_box_teacher_config
from projects.BoxTeacher.rock_seg import add_rock_seg_config

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    add_box_teacher_config(cfg)
    add_rock_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home/pan/FJS/MC/BoxTeacher-main/projects/BoxTeacher/configs/coco/boxteacher_r50_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        default="/home/pan/FJS/MC/BoxTeacher-main/projects/BoxTeacher/datasets/mul_ssmg_coco/val2017",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="/home/pan/FJS/MC/BoxTeacher-main/projects/BoxTeacher/datasets/inference/output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--is_multi_image",
        help="是否为多图片预测",
        default=True,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def inference_one_image(args, image_dir):
    image_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg')]
    for path in tqdm.tqdm(image_list, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit


def inference_multi_image(args, demo, image_dir):
    image_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('-.jpg')]
    for path1 in tqdm.tqdm(image_list, disable=not args.output):
        # use PIL, to be consistent with evaluation
        path2 = path1.replace('-.jpg', '+.jpg')
        img1 = read_image(path1, format="BGR")
        img2 = read_image(path2, format="BGR")
        img2 = cv2.resize(img2, (int(img1.shape[1]), int(img1.shape[0])))
        start_time = time.time()
        predictions, visualized_output1, visualized_output2 = demo.run_on_image(img1, img2)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path1,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path1))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output1.save(out_filename)
            visualized_output2.save(out_filename.replace("-.jpg", "+.jpg"))
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output1.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit


if __name__ == "__main__":
    # --config-file /home/pan/FJS/MC/BoxTeacher-main/projects/BoxTeacher/configs/coco/boxteacher_r50_1x.yaml
    # --input /home/pan/FJS/MC/BoxTeacher-main/projects/BoxTeacher/datasets/inference/input_image
    # --output /home/pan/FJS/MC/BoxTeacher-main/projects/BoxTeacher/datasets/inference/output
    # --opts MODEL.WEIGHTS /home/pan/FJS/MC/BoxTeacher-main/projects/BoxTeacher/output/checkpoints/model_final.pth
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    if args.is_multi_image:
        demo = MultiImageVisualizationDemo(cfg)
    else:
        demo = VisualizationDemo(cfg)

    if args.input:
        # if len(args.input) == 1:
        #     args.input = glob.glob(os.path.expanduser(args.input[0]))
        #     assert args.input, "The input path(s) was not found"
        image_dir = args.input
        assert isinstance(image_dir, str) and os.path.isdir(image_dir)
        if args.is_multi_image:
            inference_multi_image(args, demo, image_dir)
        else:
            inference_one_image(args, image_dir)
        # image_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg')]
        # for path in tqdm.tqdm(image_list, disable=not args.output):
        #     # use PIL, to be consistent with evaluation
        #     img = read_image(path, format="BGR")
        #     start_time = time.time()
        #     predictions, visualized_output = demo.run_on_image(img)
        #     logger.info(
        #         "{}: {} in {:.2f}s".format(
        #             path,
        #             "detected {} instances".format(len(predictions["instances"]))
        #             if "instances" in predictions
        #             else "finished",
        #             time.time() - start_time,
        #         )
        #     )
        #
        #     if args.output:
        #         if os.path.isdir(args.output):
        #             assert os.path.isdir(args.output), args.output
        #             out_filename = os.path.join(args.output, os.path.basename(path))
        #         else:
        #             assert len(args.input) == 1, "Please specify a directory with args.output"
        #             out_filename = args.output
        #         visualized_output.save(out_filename)
        #     else:
        #         cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        #         cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
        #         if cv2.waitKey(0) == 27:
        #             break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
