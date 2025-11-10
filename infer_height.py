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
import warnings

# Suppress warnings from deep learning libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def load_model(checkpoint_path, model_size='vits'):
    model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{model_size}14')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model


def infer_height(model, image_path, transform, device, output_size=512):
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

    # Interpolate to desired output size (512x512 for remote sensing)
    height = F.interpolate(height.unsqueeze(1), (output_size, output_size), mode='bilinear', align_corners=False).squeeze()

    return height.cpu().numpy(), raw_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='DFC2023S', help='Name of the dataset')
    parser.add_argument('--dataset_base_path', type=str, default='/home/asfand/Ahmad/datasets/', help='Base path to datasets')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the fine-tuned checkpoint')
    parser.add_argument('--model_size', type=str, default='vits', choices=['vits', 'vitb', 'vitl'], help='Model size (ViT variant)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'], help='Dataset split to infer on')
    parser.add_argument('--output_size', type=int, default=512, help='Output height map size (e.g., 512 for 512x512)')
    parser.add_argument('--results_dir', type=str, default='results/height_adapted_01', help='Base results directory')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Base logs directory')

    args = parser.parse_args()

    # Setup directories organized by dataset and model size
    results_dir = os.path.join(args.results_dir, args.dataset_name, args.model_size)
    logs_dir = os.path.join(args.logs_dir, args.dataset_name, args.model_size)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logging with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'inference_{timestamp}.log'
    logging.basicConfig(filename=os.path.join(logs_dir, log_filename), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model from fine-tuned checkpoint
    model = load_model(args.checkpoint_path, args.model_size).to(device).eval()
    logging.info(f"Loaded model from {args.checkpoint_path}")
    print(f"Loaded model from {args.checkpoint_path}")

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

        # Infer height with specified output size (512x512)
        pred_height, raw_image = infer_height(model, rgb_path, transform, device, output_size=args.output_size)

        # Load ground truth and resize to match prediction size
        gt_height = tifffile.imread(dsm_path).astype(np.float32)
        if gt_height.shape != (args.output_size, args.output_size):
            gt_height_tensor = torch.from_numpy(gt_height).unsqueeze(0).unsqueeze(0)
            gt_height_tensor = F.interpolate(gt_height_tensor, size=(args.output_size, args.output_size), mode='bilinear', align_corners=False)
            gt_height = gt_height_tensor.squeeze().numpy()

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