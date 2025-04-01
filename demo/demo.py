# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from PIL import Image

from x_agent import add_cat_seg_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "MaskFormer demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/ade20k-150/maskformer_R50_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
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


def compute_mean_attention_dist(patch_size, attention_weights, num_cls_tokens=1):
    """compute_mean_attention_dist: Computes the mean attention distance for the image

    Args:
        patch_size (int): the size of the patch
        attention_weights (np.ndarray): The attention weights for the image
        num_cls_tokens (int, optional): The number of class tokens. Defaults to 1.

    Returns:
        mean_distances (np.ndarray): The mean attention distance for the image
    """
    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    attention_weights = attention_weights[
        ..., num_cls_tokens:, num_cls_tokens:
    ]  # Removing the CLS token
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert length ** 2 == num_patches, "Num patches is not perfect square"

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    # The attention_weights along the last axis adds to 1
    # this is due to the fact that they are softmax of the raw logits
    # summation of the (attention_weights * distance_matrix)
    # should result in an average distance per token
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(
        mean_distances, axis=-1
    )  # sum along last axis to get average distance per token
    mean_distances = np.mean(
        mean_distances, axis=-1
    )  # now average across all the tokens

    return mean_distances


def compute_distance_matrix(patch_size, num_patches, length):
    """compute_distance_matrix: Computes the distance matrix for the patches in the image

    Args:
        patch_size (int): the size of the patch
        num_patches (int): the number of patches in the image
        length (int): the length of the image

    Returns:
        distance_matrix (np.ndarray): The distance matrix for the patches in the image
    """
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix


attention_results = {}
def get_attention_hook(layer_idx):
    def hook(module, input, output):
        attention = output[1].detach().cpu()
        # attention = attention[0]
        # attention = torch.nn.functional.softmax(attention, dim=-1)  # 显式归一化
        attention_results[layer_idx] = attention
    return hook


# def patch_attention(module):
#     original_forward = module.forward
#     def new_forward(q, k, v, **kwargs):
#         return original_forward(q, k, v, need_weights=True, average_attn_weights=False, **kwargs)
#     module.forward = new_forward


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    print("Output saved to: ", args.output)

    # for layer in demo.predictor.model.sem_seg_head.predictor.clip_model.visual.transformer.resblocks:
    #     patch_attention(layer.attn)

    for layer_idx, layer in enumerate(demo.predictor.model.sem_seg_head.predictor.clip_model.visual.transformer.resblocks):
        layer.attn.register_forward_hook(get_attention_hook(layer_idx))
    setattr(demo.predictor.model, "__VISUALIZATION__", True)

    idx = 0
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
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

            # 热力图可视化
            # logits = getattr(demo.predictor.model, "__data__")['logits']  # [cls, h, w]
            # # print(logits.shape)
            # # print(logits)
            # # exit(4)
            # eps = 1e-4
            # tau = 0.0
            # alpha = 0.5
            # logits[logits < tau] = eps
            # img_pil = Image.fromarray(img)
            # img_rgba = img_pil.convert("RGBA")
            # img_np = np.array(img_rgba).astype(float)
            # for all classes
            # blended_pil_all = []
            # for cls_idx in range(20):
            #     heatmap = logits[cls_idx].cpu().numpy()
            #     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            #     heatmap = np.uint8(255 * heatmap)
            #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)      # BGR转RGB
            #     heatmap_pil = Image.fromarray(heatmap)
            #     heatmap_rgba = heatmap_pil.convert("RGBA")
            #     heatmap_np = np.array(heatmap_rgba).astype(float)
            #     blended_np = (1 - alpha) * img_np[:, :, :3] + alpha * heatmap_np[:, :, :3]
            #     blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)
            #     blended_pil = Image.fromarray(blended_np)
            #     blended_pil_all.append(blended_pil)
            # vis = Image.new("RGB", [img_pil.width * 4, img_pil.height * 5])
            # # vis.paste(img_pil, (0, 0))
            # # vis.paste(visualized_output, (img_pil.width, 0))
            # for i in range(4):
            #     for j in range(5):
            #         vis.paste(blended_pil_all[i*4+j], (img_pil.width * int(i), img_pil.height * int(j)))
            # for one class
            # cls_idx = 15
            # heatmap = logits[cls_idx].cpu().numpy()
            # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            # heatmap = np.uint8(255 * heatmap)
            # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)      # BGR转RGB
            # heatmap_pil = Image.fromarray(heatmap)
            # heatmap_rgba = heatmap_pil.convert("RGBA")
            # heatmap_np = np.array(heatmap_rgba).astype(float)
            # blended_np = (1 - alpha) * img_np[:, :, :3] + alpha * heatmap_np[:, :, :3]
            # blended_np = np.clip(blended_np, 0, 255).astype(np.uint8)
            # blended_pil = Image.fromarray(blended_np)
            # vis = Image.new("RGB", [img_pil.width, img_pil.height])
            # vis.paste(blended_pil, (0, 0))

            # if args.output:
            #     if os.path.isdir(args.output):
            #         assert os.path.isdir(args.output), args.output
            #         out_filename = os.path.join(args.output, os.path.basename(path))
            #     else:
            #         assert len(args.input) == 1, "Please specify a directory with args.output"
            #         out_filename = args.output
            #     # visualized_output.save(out_filename)
            #     vis.save(out_filename)
            # else:
            #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            #     if cv2.waitKey(0) == 27:
            #         break  # esc to quit
            
            # exit(4)
            idx += 1
            if idx > 1:
                break
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

    for layer_idx, attn in attention_results.items():
        # print(layer_idx, attn.shape)
        mad = compute_mean_attention_dist(patch_size=16, attention_weights=attn.numpy())
        print(layer_idx, "mean attention distance", mad)