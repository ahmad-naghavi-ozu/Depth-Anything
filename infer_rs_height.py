#!/usr/bin/env python3
"""
Inference script for remote sensing height estimation using trained ZoeDepth model
Evaluates on DFC2023S test set and computes metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metric_depth'))

import argparse
import torch
import torch.nn.functional as F
import tifffile
import numpy as np
from tqdm import tqdm
from pathlib import Path

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.data.dfc2023s import DFC2023S
from metrics_utils import compute_dsm_metrics
import logging


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, 'inference.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def infer_batch(model, images, device):
    """Run inference on a batch of images"""
    images = images.to(device)
    
    with torch.no_grad():
        pred = model(images)
        
        # Handle different output formats
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('pred', None))
        
        if pred is None:
            raise ValueError("Model output format not recognized")
    
    return pred


def save_prediction(pred_height, output_path, reference_tif=None):
    """Save prediction as GeoTIFF with same georeferencing as reference"""
    # pred_height shape: (H, W)
    pred_height_np = pred_height.cpu().numpy().astype(np.float32)
    
    if reference_tif is not None:
        # TODO: Copy georeferencing metadata from reference
        # For now, just save as regular TIFF
        tifffile.imwrite(output_path, pred_height_np)
    else:
        tifffile.imwrite(output_path, pred_height_np)


def main():
    parser = argparse.ArgumentParser(description='RS Height Estimation Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset-root', type=str, 
                       default='/home/asfand/Ahmad/datasets/DFC2023S',
                       help='Root directory of DFC2023S dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--output-dir', type=str, default='./results/rs_height_zoedepth',
                       help='Output directory for predictions and metrics')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for inference')
    parser.add_argument('--model', type=str, default='zoedepth',
                       help='Model architecture')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save prediction GeoTIFFs')
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config and model
    logger.info(f"Loading model from {args.checkpoint}")
    config = get_config(args.model, "eval", "dfc2023s")
    config.pretrained_resource = None  # Don't load pretrained weights
    
    model = build_model(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    
    # Load dataset
    logger.info(f"Loading {args.split} set from {args.dataset_root}")
    dataset = DFC2023S(
        data_dir_root=args.dataset_root,
        split=args.split,
        resize_shape=(512, 512)
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Initialize metrics
    total_delta1 = 0.0
    total_delta2 = 0.0
    total_delta3 = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_r2 = 0.0
    total_rmse_building = 0.0
    total_rmse_matched = 0.0
    total_high_rise_rmse = 0.0
    total_mid_rise_rmse = 0.0
    total_low_rise_rmse = 0.0
    count_high_rise = 0
    count_mid_rise = 0
    count_low_rise = 0
    
    num_samples = 0
    
    # Create prediction output directory
    if args.save_predictions:
        pred_dir = os.path.join(args.output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        logger.info(f"Saving predictions to {pred_dir}")
    
    # Inference loop
    logger.info("Starting inference...")
    for idx, sample in enumerate(tqdm(dataloader, desc="Inference")):
        images = sample['image']
        depths_gt = sample['depth']
        
        # Inference
        pred_depths = infer_batch(model, images, device)
        
        # Process each sample in batch
        for i in range(images.shape[0]):
            pred_height = pred_depths[i].squeeze()  # (H, W)
            gt_height = depths_gt[i].squeeze()  # (H, W)
            
            # Move to CPU for metrics computation
            pred_height_np = pred_height.cpu().numpy()
            gt_height_np = gt_height.cpu().numpy()
            
            # Compute metrics
            (
                total_delta1, total_delta2, total_delta3,
                total_mse, total_mae, total_rmse,
                total_rmse_building, total_rmse_matched,
                total_high_rise_rmse, total_mid_rise_rmse, total_low_rise_rmse,
                count_high_rise, count_mid_rise, count_low_rise,
                total_r2, _, _
            ) = compute_dsm_metrics(
                verbose=False,
                logger=logger,
                total_delta1=total_delta1,
                total_delta2=total_delta2,
                total_delta3=total_delta3,
                total_mse=total_mse,
                total_mae=total_mae,
                total_rmse=total_rmse,
                total_rmse_building=total_rmse_building,
                total_rmse_matched=total_rmse_matched,
                total_high_rise_rmse=total_high_rise_rmse,
                total_mid_rise_rmse=total_mid_rise_rmse,
                total_low_rise_rmse=total_low_rise_rmse,
                count_high_rise=count_high_rise,
                count_mid_rise=count_mid_rise,
                count_low_rise=count_low_rise,
                dsm_tile=gt_height_np,
                dsm_pred=pred_height_np,
                total_r2=total_r2
            )
            
            # Save prediction if requested
            if args.save_predictions:
                sample_idx = idx * args.batch_size + i
                output_path = os.path.join(pred_dir, f'pred_{sample_idx:05d}.tif')
                save_prediction(pred_height, output_path)
            
            num_samples += 1
    
    # Compute average metrics
    avg_delta1 = total_delta1 / num_samples
    avg_delta2 = total_delta2 / num_samples
    avg_delta3 = total_delta3 / num_samples
    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples
    avg_rmse = total_rmse / num_samples
    avg_r2 = total_r2 / num_samples
    
    # Log results
    logger.info("\n" + "="*60)
    logger.info("INFERENCE RESULTS")
    logger.info("="*60)
    logger.info(f"Dataset: {args.split} split ({num_samples} samples)")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("-"*60)
    logger.info(f"MSE    : {avg_mse:.4f}")
    logger.info(f"MAE    : {avg_mae:.4f} meters")
    logger.info(f"RMSE   : {avg_rmse:.4f} meters")
    logger.info(f"R²     : {avg_r2:.4f}")
    logger.info(f"Delta1 : {avg_delta1:.4f} (< 1.25)")
    logger.info(f"Delta2 : {avg_delta2:.4f} (< 1.25²)")
    logger.info(f"Delta3 : {avg_delta3:.4f} (< 1.25³)")
    logger.info("="*60)
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Dataset: {args.split} ({num_samples} samples)\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write("-"*60 + "\n")
        f.write(f"MSE    : {avg_mse:.4f}\n")
        f.write(f"MAE    : {avg_mae:.4f} meters\n")
        f.write(f"RMSE   : {avg_rmse:.4f} meters\n")
        f.write(f"R²     : {avg_r2:.4f}\n")
        f.write(f"Delta1 : {avg_delta1:.4f}\n")
        f.write(f"Delta2 : {avg_delta2:.4f}\n")
        f.write(f"Delta3 : {avg_delta3:.4f}\n")
    
    logger.info(f"Metrics saved to {metrics_file}")
    logger.info("Inference completed!")


if __name__ == '__main__':
    main()
