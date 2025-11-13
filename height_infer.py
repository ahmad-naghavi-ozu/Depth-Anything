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

from depth_anything.dpt import DepthAnything, DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from metrics_utils import compute_dsm_metrics


def load_model(checkpoint_path, model_size='vits'):
    # Model architecture parameters based on encoder size (must match training)
    if model_size == 'vits':
        features, out_channels = 64, [48, 96, 192, 384]
    elif model_size == 'vitb':
        features, out_channels = 128, [96, 192, 384, 768]
    else:  # vitl
        features, out_channels = 256, [256, 512, 1024, 1024]
    
    # Create model with same architecture as training
    model = DPT_DINOv2(encoder=model_size, features=features, out_channels=out_channels)
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
    parser.add_argument('--save_visualizations', action='store_true', default=False, help='Save PNG visualizations of predictions')
    parser.add_argument('--eval_only', action='store_true', default=False, help='Evaluate existing predictions without running inference')

    args = parser.parse_args()

    # Setup directories organized by dataset and model size
    results_dir = os.path.join(args.results_dir, args.dataset_name, args.model_size)
    logs_dir = os.path.join(args.logs_dir, args.dataset_name, args.model_size)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Setup logging to both file and console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'inference_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(logs_dir, log_filename))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and transform only if not in eval-only mode
    model = None
    transform = None
    
    if not args.eval_only:
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required when not using --eval_only mode")
        model = load_model(args.checkpoint_path, args.model_size).to(device).eval()
        transform = Compose([
            Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,
                   ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    # Log complete configuration
    mode = "Evaluation" if args.eval_only else "Inference"
    logging.info(f"{mode} Configuration:")
    logging.info(f"  Mode: {mode}")
    logging.info(f"  Dataset: {args.dataset_name}")
    logging.info(f"  Model size: {args.model_size}")
    if not args.eval_only:
        logging.info(f"  Checkpoint: {args.checkpoint_path}")
        logging.info(f"  Device: {device}")
        logging.info(f"  Save visualizations: {args.save_visualizations}")
    logging.info(f"  Split: {args.split}")
    logging.info(f"  Output size: {args.output_size}x{args.output_size}")

    # Dataset paths
    rgb_dir = os.path.join(args.dataset_base_path, args.dataset_name, args.split, 'rgb')
    dsm_dir = os.path.join(args.dataset_base_path, args.dataset_name, args.split, 'dsm')

    if not os.path.exists(rgb_dir):
        raise ValueError(f"RGB directory {rgb_dir} does not exist")

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.tif')])

    mode_text = "evaluation" if args.eval_only else "inference"
    logging.info(f"Starting {mode_text} on {len(rgb_files)} images")

    # Initialize metric accumulators
    total_mse = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_r2 = 0.0
    total_delta1 = 0.0
    total_delta2 = 0.0
    total_delta3 = 0.0
    total_rmse_building = 0.0
    total_rmse_matched = 0.0
    total_rmse_low_rise = 0.0
    total_rmse_mid_rise = 0.0
    total_rmse_high_rise = 0.0
    count_low_rise = 0
    count_mid_rise = 0
    count_high_rise = 0
    
    # Height category thresholds (in meters)
    low_rise_max = 15.0
    mid_rise_max = 40.0

    desc_text = "Evaluating" if args.eval_only else "Inferring"
    for filename in tqdm(rgb_files, desc=desc_text):
        rgb_path = os.path.join(rgb_dir, filename)
        dsm_path = os.path.join(dsm_dir, filename)

        if args.eval_only:
            # Load existing prediction
            pred_filename = filename.replace('.tif', '_pred_height.npy')
            pred_path = os.path.join(results_dir, pred_filename)
            if not os.path.exists(pred_path):
                logging.warning(f"Prediction file not found: {pred_path}, skipping {filename}")
                continue
            pred_height = np.load(pred_path)
        else:
            # Infer height with specified output size (512x512)
            pred_height, raw_image = infer_height(model, rgb_path, transform, device, output_size=args.output_size)
            
            # Save prediction
            pred_filename = filename.replace('.tif', '_pred_height.npy')
            np.save(os.path.join(results_dir, pred_filename), pred_height)

            # Save visualization if requested (handle near-zero predictions)
            if args.save_visualizations:
                pred_range = pred_height.max() - pred_height.min()
                if pred_range > 1e-6:  # Avoid division by zero
                    pred_norm = (pred_height - pred_height.min()) / pred_range * 255
                else:
                    # If predictions are all nearly the same, just visualize as is
                    pred_norm = np.clip(pred_height * 10, 0, 255)  # Scale up small values
                pred_norm = pred_norm.astype(np.uint8)
                pred_color = cv2.applyColorMap(pred_norm, cv2.COLORMAP_INFERNO)

                vis_filename = filename.replace('.tif', '_height_vis.png')
                cv2.imwrite(os.path.join(results_dir, vis_filename), pred_color)

        # Load ground truth and resize to match prediction size
        gt_height = tifffile.imread(dsm_path).astype(np.float32)
        if gt_height.shape != (args.output_size, args.output_size):
            gt_height_tensor = torch.from_numpy(gt_height).unsqueeze(0).unsqueeze(0)
            gt_height_tensor = F.interpolate(gt_height_tensor, size=(args.output_size, args.output_size), mode='bilinear', align_corners=False)
            gt_height = gt_height_tensor.squeeze().numpy()

        # Create building mask (buildings have height > 1m)
        gt_mask = (gt_height > 1.0).astype(np.uint8)
        
        # Store previous totals to compute per-sample metrics
        prev_mse = total_mse
        prev_mae = total_mae
        prev_rmse = total_rmse
        prev_r2 = total_r2
        prev_delta1 = total_delta1
        prev_delta2 = total_delta2
        prev_delta3 = total_delta3
        prev_rmse_building = total_rmse_building
        prev_rmse_low_rise = total_rmse_low_rise
        prev_rmse_mid_rise = total_rmse_mid_rise
        prev_rmse_high_rise = total_rmse_high_rise
        prev_count_low_rise = count_low_rise
        prev_count_mid_rise = count_mid_rise
        prev_count_high_rise = count_high_rise
        
        # Compute per-sample metrics using the utility function
        (
            total_delta1,
            total_delta2,
            total_delta3,
            total_mse,
            total_mae,
            total_rmse,
            total_rmse_building,
            total_rmse_matched,
            total_rmse_high_rise,
            total_rmse_mid_rise,
            total_rmse_low_rise,
            count_high_rise,
            count_mid_rise,
            count_low_rise,
            total_r2,
            _,
            _
        ) = compute_dsm_metrics(
            verbose=False,
            logger=logging,
            total_delta1=total_delta1,
            total_delta2=total_delta2,
            total_delta3=total_delta3,
            total_mse=total_mse,
            total_mae=total_mae,
            total_rmse=total_rmse,
            total_rmse_building=total_rmse_building,
            total_rmse_matched=total_rmse_matched,
            total_high_rise_rmse=total_rmse_high_rise,
            total_mid_rise_rmse=total_rmse_mid_rise,
            total_low_rise_rmse=total_rmse_low_rise,
            count_high_rise=count_high_rise,
            count_mid_rise=count_mid_rise,
            count_low_rise=count_low_rise,
            dsm_tile=gt_height,
            dsm_pred=pred_height,
            gt_mask=gt_mask,
            pred_mask=None,
            total_r2=total_r2,
            low_rise_max=low_rise_max,
            mid_rise_max=mid_rise_max
        )
        
        # Log per-sample metrics
        sample_mse = total_mse - prev_mse
        sample_mae = total_mae - prev_mae
        sample_rmse = total_rmse - prev_rmse
        sample_r2 = total_r2 - prev_r2
        sample_delta1 = total_delta1 - prev_delta1
        sample_delta2 = total_delta2 - prev_delta2
        sample_delta3 = total_delta3 - prev_delta3
        sample_rmse_building = total_rmse_building - prev_rmse_building
        
        # Calculate per-sample height category RMSE
        sample_rmse_low = total_rmse_low_rise - prev_rmse_low_rise if count_low_rise > prev_count_low_rise else 0.0
        sample_rmse_mid = total_rmse_mid_rise - prev_rmse_mid_rise if count_mid_rise > prev_count_mid_rise else 0.0
        sample_rmse_high = total_rmse_high_rise - prev_rmse_high_rise if count_high_rise > prev_count_high_rise else 0.0
        
        # Build log message with available metrics
        log_msg = (f"{filename}: MSE={sample_mse:.4f}, MAE={sample_mae:.4f}, RMSE={sample_rmse:.4f}, "
                  f"R²={sample_r2:.4f}, δ1={sample_delta1:.4f}, δ2={sample_delta2:.4f}, δ3={sample_delta3:.4f}, "
                  f"RMSE_building={sample_rmse_building:.4f}")
        
        # Add height category RMSE if applicable
        if count_low_rise > prev_count_low_rise:
            log_msg += f", RMSE_low_rise={sample_rmse_low:.4f}"
        if count_mid_rise > prev_count_mid_rise:
            log_msg += f", RMSE_mid_rise={sample_rmse_mid:.4f}"
        if count_high_rise > prev_count_high_rise:
            log_msg += f", RMSE_high_rise={sample_rmse_high:.4f}"
        
        logging.info(log_msg)

    # Compute average metrics
    num_samples = len(rgb_files)
    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples
    avg_rmse = total_rmse / num_samples
    avg_r2 = total_r2 / num_samples
    avg_delta1 = total_delta1 / num_samples
    avg_delta2 = total_delta2 / num_samples
    avg_delta3 = total_delta3 / num_samples
    avg_rmse_building = total_rmse_building / num_samples
    
    avg_rmse_low_rise = total_rmse_low_rise / count_low_rise if count_low_rise > 0 else 0.0
    avg_rmse_mid_rise = total_rmse_mid_rise / count_mid_rise if count_mid_rise > 0 else 0.0
    avg_rmse_high_rise = total_rmse_high_rise / count_high_rise if count_high_rise > 0 else 0.0
    
    # Report final averaged metrics
    logging.info("\n" + "="*60)
    logging.info("FINAL AVERAGED METRICS")
    logging.info("="*60)
    logging.info(f"Number of samples: {num_samples}")
    logging.info(f"Average MSE:       {avg_mse:.4f}")
    logging.info(f"Average MAE:       {avg_mae:.4f}")
    logging.info(f"Average RMSE:      {avg_rmse:.4f}")
    logging.info(f"Average R²:        {avg_r2:.4f}")
    logging.info(f"Average Delta1:    {avg_delta1:.4f}")
    logging.info(f"Average Delta2:    {avg_delta2:.4f}")
    logging.info(f"Average Delta3:    {avg_delta3:.4f}")
    logging.info(f"Average RMSE (buildings): {avg_rmse_building:.4f}")
    logging.info(f"Average RMSE (low-rise, <{low_rise_max}m): {avg_rmse_low_rise:.4f} (samples: {count_low_rise})")
    logging.info(f"Average RMSE (mid-rise, {low_rise_max}-{mid_rise_max}m): {avg_rmse_mid_rise:.4f} (samples: {count_mid_rise})")
    logging.info(f"Average RMSE (high-rise, >{mid_rise_max}m): {avg_rmse_high_rise:.4f} (samples: {count_high_rise})")
    logging.info("="*60)
    completion_text = "Evaluation" if args.eval_only else "Inference"
    logging.info(f"{completion_text} completed")