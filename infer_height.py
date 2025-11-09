import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import tifffile
import logging
from datetime import datetime

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def load_model(checkpoint_path, encoder='vits'):
    model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def infer_height(model, image_path, transform, device):
    raw_image = tifffile.imread(image_path)
    if raw_image.ndim == 2:
        raw_image = np.stack([raw_image] * 3, axis=-1)
    elif raw_image.shape[-1] == 1:
        raw_image = np.repeat(raw_image, 3, axis=-1)

    image = raw_image.astype(np.float32) / 255.0

    h, w = image.shape[:2]

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        height = model(image)

    height = F.interpolate(height.unsqueeze(1), (h, w), mode='bilinear', align_corners=False).squeeze()

    return height.cpu().numpy(), raw_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='DFC2023S', help='Name of the dataset')
    parser.add_argument('--dataset_base_path', type=str, default='/home/asfand/Ahmad/datasets/', help='Base path to datasets')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the fine-tuned checkpoint')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'], help='Encoder type')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'], help='Dataset split to infer on')
    parser.add_argument('--results_dir', type=str, default='results/height_adapted_01', help='Base results directory')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Base logs directory')

    args = parser.parse_args()

    # Setup directories
    results_dir = os.path.join(args.results_dir, args.dataset_name)
    logs_dir = os.path.join(args.logs_dir, args.dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logging with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'inference_{timestamp}.log'
    logging.basicConfig(filename=os.path.join(logs_dir, log_filename), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = load_model(args.checkpoint_path, args.encoder).to(device).eval()
    logging.info(f"Loaded model from {args.checkpoint_path}")

    # Transform
    transform = Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,
               ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # Dataset paths
    rgb_dir = os.path.join(args.dataset_base_path, args.dataset_name, args.split, 'rgb')
    dsm_dir = os.path.join(args.dataset_base_path, args.dataset_name, args.split, 'dsm')

    if not os.path.exists(rgb_dir):
        raise ValueError(f"RGB directory {rgb_dir} does not exist")

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.tif')])

    logging.info(f"Starting inference on {len(rgb_files)} images")

    for filename in tqdm(rgb_files, desc="Inferring"):
        rgb_path = os.path.join(rgb_dir, filename)
        dsm_path = os.path.join(dsm_dir, filename)

        # Infer height
        pred_height, raw_image = infer_height(model, rgb_path, transform, device)

        # Load ground truth
        gt_height = tifffile.imread(dsm_path).astype(np.float32)

        # Save prediction
        pred_filename = filename.replace('.tif', '_pred_height.npy')
        np.save(os.path.join(results_dir, pred_filename), pred_height)

        # Save visualization
        pred_norm = (pred_height - pred_height.min()) / (pred_height.max() - pred_height.min()) * 255
        pred_norm = pred_norm.astype(np.uint8)
        pred_color = cv2.applyColorMap(pred_norm, cv2.COLORMAP_INFERNO)

        vis_filename = filename.replace('.tif', '_height_vis.png')
        cv2.imwrite(os.path.join(results_dir, vis_filename), pred_color)

        # Log
        mse = np.mean((pred_height - gt_height) ** 2)
        mae = np.mean(np.abs(pred_height - gt_height))
        logging.info(f"{filename}: MSE={mse:.4f}, MAE={mae:.4f}")

    logging.info("Inference completed")