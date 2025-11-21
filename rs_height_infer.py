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
from metrics_utils import compute_dsm_metrics, r2_score
import logging
import importlib


def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, 'inference.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # mode='w' overwrites the file
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration if logging was already set up
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
    
    # Extract dataset name from dataset root path and create dataset-specific output dir
    dataset_name = os.path.basename(args.dataset_root.rstrip('/'))
    args.output_dir = os.path.join(args.output_dir, dataset_name)
    
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    
    # Load dataset - detect dataset type from path
    logger.info(f"Loading {args.split} set from {args.dataset_root}")
    dataset_name = os.path.basename(args.dataset_root.rstrip('/')).lower()
    
    # Import the appropriate dataset class
    dataset_module = importlib.import_module(f'zoedepth.data.{dataset_name}')
    
    # Get the dataset class - look for class with matching name
    dataset_classes = [obj for name, obj in dataset_module.__dict__.items() 
                      if isinstance(obj, type) and name.lower().replace('_', '').replace('-', '') == dataset_name.replace('_', '').replace('-', '')]
    
    if not dataset_classes:
        # Fallback: look for any Dataset class in the module
        dataset_classes = [obj for name, obj in dataset_module.__dict__.items() 
                          if isinstance(obj, type) and 'dataset' in name.lower()]
    
    if not dataset_classes:
        raise ValueError(f"Could not find dataset class in module 'zoedepth.data.{dataset_name}'")
    
    DatasetClass = dataset_classes[0]
    logger.info(f"Using dataset class: {DatasetClass.__name__}")
    
    dataset = DatasetClass(
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
    
    # Create per-sample metrics file
    per_sample_file = os.path.join(args.output_dir, 'per_sample_metrics.txt')
    with open(per_sample_file, 'w', encoding='utf-8') as f:
        f.write("Sample\tMSE\tMAE\tRMSE\tR2\tDelta1\tDelta2\tDelta3\tRMSE_Building\tRMSE_Low\tRMSE_Mid\tRMSE_High\n")
    
    # Inference loop
    logger.info("Starting inference...")
    for idx, sample in enumerate(tqdm(dataloader, desc="Inference")):
        images = sample['image']
        depths_gt = sample['depth']
        filenames = sample['filename']
        
        # Inference
        pred_depths = infer_batch(model, images, device)
        
        # Process each sample in batch
        for i in range(images.shape[0]):
            pred_height = pred_depths[i].squeeze()  # (H, W)
            gt_height = depths_gt[i].squeeze()  # (H, W)
            filename = filenames[i]
            
            # Resize prediction to match ground truth size if needed
            if pred_height.shape != gt_height.shape:
                pred_height = F.interpolate(
                    pred_height.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
                    size=gt_height.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze()  # (H, W)
            
            # Get semantic mask if available
            gt_sem_mask = sample.get('sem_mask', None)
            if gt_sem_mask is not None:
                gt_sem_mask = gt_sem_mask[i].squeeze().cpu().numpy()
            
            # Move to CPU for metrics computation
            pred_height_np = pred_height.cpu().numpy()
            gt_height_np = gt_height.cpu().numpy()
            
            # Compute metrics
            (
                total_delta1, total_delta2, total_delta3,
                total_mse, total_mae, total_rmse,
                total_rmse_building, total_high_rise_rmse, total_mid_rise_rmse, total_low_rise_rmse,
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
                total_high_rise_rmse=total_high_rise_rmse,
                total_mid_rise_rmse=total_mid_rise_rmse,
                total_low_rise_rmse=total_low_rise_rmse,
                count_high_rise=count_high_rise,
                count_mid_rise=count_mid_rise,
                count_low_rise=count_low_rise,
                dsm_tile=gt_height_np,
                dsm_pred=pred_height_np,
                gt_mask=gt_sem_mask,
                total_r2=total_r2
            )
            
            # Save per-sample metrics
            with open(per_sample_file, 'a', encoding='utf-8') as f:
                # Compute individual metrics for this sample
                abs_diff = np.abs(pred_height_np - gt_height_np)
                mse = np.mean(abs_diff ** 2)
                mae = np.mean(abs_diff)
                rmse = np.sqrt(mse)
                r2 = r2_score(gt_height_np, pred_height_np)
                if r2 is None:
                    r2 = 0.0
                
                # Compute delta metrics with epsilon to avoid divide by zero
                epsilon = 1e-6
                with np.errstate(divide='ignore', invalid='ignore'):
                    max_ratio = np.maximum(
                        pred_height_np / (gt_height_np + epsilon), 
                        gt_height_np / (pred_height_np + epsilon)
                    )
                    # Filter out invalid values (inf, nan)
                    valid_mask = np.isfinite(max_ratio)
                    delta1 = np.mean(max_ratio[valid_mask] < 1.25) if np.any(valid_mask) else 0.0
                    delta2 = np.mean(max_ratio[valid_mask] < 1.25 ** 2) if np.any(valid_mask) else 0.0
                    delta3 = np.mean(max_ratio[valid_mask] < 1.25 ** 3) if np.any(valid_mask) else 0.0
                
                # Building-specific metrics
                rmse_building = 0.0
                rmse_low = 0.0
                rmse_mid = 0.0
                rmse_high = 0.0
                
                if gt_sem_mask is not None:
                    building_mask = (gt_sem_mask == 1).flatten()
                    if np.sum(building_mask) > 0:
                        pred_buildings = pred_height_np.flatten()[building_mask]
                        gt_buildings = gt_height_np.flatten()[building_mask]
                        rmse_building = np.sqrt(np.mean((pred_buildings - gt_buildings) ** 2))
                    
                    # Height categories
                    low_mask = building_mask & (gt_height_np.flatten() >= 1) & (gt_height_np.flatten() < 15.0)
                    mid_mask = building_mask & (gt_height_np.flatten() >= 15.0) & (gt_height_np.flatten() < 40.0)
                    high_mask = building_mask & (gt_height_np.flatten() >= 40.0)
                    
                    if np.sum(low_mask) > 0:
                        rmse_low = np.sqrt(np.mean((pred_height_np.flatten()[low_mask] - gt_height_np.flatten()[low_mask]) ** 2))
                    if np.sum(mid_mask) > 0:
                        rmse_mid = np.sqrt(np.mean((pred_height_np.flatten()[mid_mask] - gt_height_np.flatten()[mid_mask]) ** 2))
                    if np.sum(high_mask) > 0:
                        rmse_high = np.sqrt(np.mean((pred_height_np.flatten()[high_mask] - gt_height_np.flatten()[high_mask]) ** 2))
                
                f.write(f"{filename}\t{mse:.4f}\t{mae:.4f}\t{rmse:.4f}\t{r2:.4f}\t{delta1:.4f}\t{delta2:.4f}\t{delta3:.4f}\t{rmse_building:.4f}\t{rmse_low:.4f}\t{rmse_mid:.4f}\t{rmse_high:.4f}\n")
            
            # Save prediction if requested
            if args.save_predictions:
                output_path = os.path.join(pred_dir, f'{filename}.tif')
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
    avg_rmse_building = total_rmse_building / num_samples
    
    # Building-specific averages
    avg_high_rise_rmse = total_high_rise_rmse / count_high_rise if count_high_rise > 0 else 0.0
    avg_mid_rise_rmse = total_mid_rise_rmse / count_mid_rise if count_mid_rise > 0 else 0.0
    avg_low_rise_rmse = total_low_rise_rmse / count_low_rise if count_low_rise > 0 else 0.0
    
    # Log results
    logger.info("\n" + "="*60)
    logger.info("INFERENCE RESULTS")
    logger.info("="*60)
    logger.info(f"Dataset: {args.split} split ({num_samples} samples)")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("-"*60)
    logger.info("Overall Metrics:")
    logger.info(f"  MSE    : {avg_mse:.4f}")
    logger.info(f"  MAE    : {avg_mae:.4f} meters")
    logger.info(f"  RMSE   : {avg_rmse:.4f} meters")
    logger.info(f"  R2     : {avg_r2:.4f}")
    logger.info(f"  Delta1 : {avg_delta1:.4f} (< 1.25)")
    logger.info(f"  Delta2 : {avg_delta2:.4f} (< 1.25^2)")
    logger.info(f"  Delta3 : {avg_delta3:.4f} (< 1.25^3)")
    logger.info("-"*60)
    logger.info("Building-Specific Metrics:")
    logger.info(f"  RMSE Building : {avg_rmse_building:.4f} meters")
    logger.info(f"  RMSE Low-rise : {avg_low_rise_rmse:.4f} meters (samples: {count_low_rise})")
    logger.info(f"  RMSE Mid-rise : {avg_mid_rise_rmse:.4f} meters (samples: {count_mid_rise})")
    logger.info(f"  RMSE High-rise: {avg_high_rise_rmse:.4f} meters (samples: {count_high_rise})")
    logger.info("="*60)
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {args.split} ({num_samples} samples)\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write("-"*60 + "\n")
        f.write("Overall Metrics:\n")
        f.write(f"  MSE    : {avg_mse:.4f}\n")
        f.write(f"  MAE    : {avg_mae:.4f} meters\n")
        f.write(f"  RMSE   : {avg_rmse:.4f} meters\n")
        f.write(f"  R2     : {avg_r2:.4f}\n")
        f.write(f"  Delta1 : {avg_delta1:.4f}\n")
        f.write(f"  Delta2 : {avg_delta2:.4f}\n")
        f.write(f"  Delta3 : {avg_delta3:.4f}\n")
        f.write("-"*60 + "\n")
        f.write("Building-Specific Metrics:\n")
        f.write(f"  RMSE Building : {avg_rmse_building:.4f} meters\n")
        f.write(f"  RMSE Low-rise : {avg_low_rise_rmse:.4f} meters (samples: {count_low_rise})\n")
        f.write(f"  RMSE Mid-rise : {avg_mid_rise_rmse:.4f} meters (samples: {count_mid_rise})\n")
        f.write(f"  RMSE High-rise: {avg_high_rise_rmse:.4f} meters (samples: {count_high_rise})\n")
    
    logger.info(f"Per-sample metrics saved to: {per_sample_file}")
    logger.info(f"Summary metrics saved to: {metrics_file}")
    logger.info("Inference completed!")


if __name__ == '__main__':
    main()
