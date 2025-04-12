import cv2
import mmcv
import argparse
import os, sys
import os.path as osp
import time
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from loguru import logger

from mmdet.apis import init_detector, inference_detector

co_detr_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(co_detr_path)
sys.path.append(os.path.join(co_detr_path, 'Co-DETR', 'boxmot'))
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.boosttrack.boosttrack import BoostTrack
from tracking.timer import Timer

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


config_file = 'cropped_inference/co_dino_5scale_swin_large_3x_coco.py'
# checkpoint_file = 'cropped/epoch_1.pth'
checkpoint_file = 'cropped2/epoch_10.pth'




# img = '../MVA2025_elsaA/OC_SORT/datasets/cropped_images/train/00001_260_0_0_640_640.jpg'  # or img = mmcv.imread(img), which will only load it once
# img = '00001.jpg'
# result = inference_detector(model, img)

# print(f"result {len(result)} {result[0].shape}")


def get_video_image_dict(root_path):
    video_image_dict = {}
    
    for video_name in os.listdir(root_path):
        video_path = osp.join(root_path, video_name)
        
        if not osp.isdir(video_path):
            continue

        image_paths = []
        for maindir, _, file_name_list in os.walk(video_path):
            for filename in file_name_list:
                ext = osp.splitext(filename)[1].lower()
                if ext in IMAGE_EXT:
                    image_paths.append(osp.join(maindir, filename))

        video_image_dict[video_name] = sorted(image_paths)

    return video_image_dict

def draw_top_boxes(image_path, prediction, output_path="output_with_boxes.jpg", conf_thresh=0.2):
    """
    Draw the predicted boxes whose confidence is above conf_thresh.
    
    Args:
        image_path (str): Path to the input image.
        prediction (torch.Tensor): [1, num_queries, 6] tensor with [x_center, y_center, w, h, score, 1].
        orig_size (torch.Tensor): Original image size [h, w].
        target_size (torch.Tensor): Resized image size [new_h, new_w].
        output_path (str): Path to save the output image.
        conf_thresh (float): Confidence threshold.
    """
    # Load the image with OpenCV (BGR) and convert to RGB & PIL for drawing
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    
    # Extract predictions
    pred = prediction[0]  # [num_queries, 6]
    boxes = pred[:, :4]   # [num_queries, 4] with format: [x_center, y_center, w, h]
    scores = pred[:, 4]   # [num_queries]
    
    # Filter boxes by confidence threshold
    keep = scores >= conf_thresh
    filtered_boxes = boxes[keep]
    filtered_scores = scores[keep]
    
    # Convert from [x_center, y_center, w, h] to [x1, y1, x2, y2]
    # boxes_xyxy = torch.zeros_like(filtered_boxes)
    # boxes_xyxy[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2  # x1
    # boxes_xyxy[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2  # y1
    # boxes_xyxy[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2  # x2
    # boxes_xyxy[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2  # y2
    
    orig_h, orig_w = img.shape[:2]  # original sizes
    
    # Draw each box on the image
    draw = ImageDraw.Draw(img_pil)
    for box, score in zip(filtered_boxes, filtered_scores):
        x1, y1, x2, y2 = box.tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 10), f"{score:.2f}", fill="red")
            print(f"Draw box at {x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f} with score {score:.2f}")
    
    img_pil.save(output_path)
    print(f"Saved image with boxes to {output_path}")


# draw_top_boxes(img, result, output_path="output_with_boxes.jpg", conf_thresh=0.2)

def predict_videos(num_workers=1):
    res_folder = 'tracking_outputs'
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    if not os.path.exists('verify'):
        os.makedirs('verify')
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    video_image_dict = get_video_image_dict('../DQ-DETR/data/pub_test')

    # def process_video(video_name, files):
    for video_name, files in video_image_dict.items():
        tracker = BotSort(
            reid_weights=Path("osnet_x0_25_msmt17.pt"),
            device=0,
            half=False,
            proximity_thresh=0.1,
            with_reid=False,
            match_thresh=0.3,
            new_track_thresh=0.4,
            track_high_thresh=0.2,
        )
        timer = Timer()
        results = []
        for frame_id, img_path in enumerate(files, 1):
            # outputs, img_info = predictor.inference(img_path, timer)
            img = cv2.imread(img_path)
            outputs = inference_detector(model, img_path) 
            draw_top_boxes(img_path, outputs, output_path=f"verify/img_{frame_id}.jpg", conf_thresh=0.1)
            if outputs[0] is not None and outputs[0].shape[0] > 0:
               
                # online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], img.shape[:2])
                print(f"outputs[0] shape {outputs[0].shape}")
                online_targets = tracker.update(
                    np.hstack((outputs[0], np.ones((outputs[0].shape[0], 1), dtype=np.int))),
                    cv2.imread(img_path),
                )
                online_tlwhs = []
                online_ids = []
                tids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > 4
                    if tid in tids:
                        logger.warning(f"Duplicate ID {tid} detected in frame {frame_id} of video {video_name}.")
                        continue
                    if tlwh[2] * tlwh[3] > 0 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        tids.append(tid)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1,1,1\n"
                        )
                        print(f"{video_name} {frame_id} {tid} ({tlwh[0]:.2f} {tlwh[1]:.2f} {tlwh[2]:.2f} {tlwh[3]:.2f})")
                timer.toc()
            else:
                timer.toc()

            if frame_id % 20 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # if args.save_result:
        res_file = osp.join(res_folder, f"{video_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

    # Use ThreadPoolExecutor to process videos in parallel
    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [
    #         executor.submit(process_video, video_name, files)
    #         for video_name, files in video_image_dict.items()
    #     ]
    #     for future in futures:
    #         future.result()  # Wait for all threads to complete


if __name__ == '__main__':
    predict_videos()