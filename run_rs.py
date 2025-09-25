#!/usr/bin/env python3

"""
Remote Sensing Height Estimation Inference Script

This script runs inference on Remote Sensing RGB imagery to produce heightmaps (DSM-like outputs).
For detailed documentation, see: docs/technical/README_RS.md

Usage:
    python run_rs.py --img-path /path/to/images/ --outdir ./results
"""

import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from PIL import Image

# Import from the original Depth Anything for inference
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def run_rs_inference():
    """
    Run inference on Remote Sensing RGB imagery to produce heightmaps (DSM-like outputs)
    """
    parser = argparse.ArgumentParser(description='Remote Sensing Height Estimation using Depth Anything')
    parser.add_argument('--img-path', type=str, required=True, help='Path to input RGB image(s)')
    parser.add_argument('--outdir', type=str, default='./rs_output', help='Output directory for height maps')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'], 
                       help='Encoder size')
    parser.add_argument('--pred-only', action='store_true', 
                       help='Only save the height prediction without visualization')
    parser.add_argument('--grayscale', action='store_true', 
                       help='Save height maps as grayscale instead of colored')
    parser.add_argument('--save-raw', action='store_true', 
                       help='Save raw height values as numpy arrays (.npy)')
    parser.add_argument('--height-scale', type=float, default=1.0,
                       help='Scale factor for height values (useful for unit conversion)')
    parser.add_argument('--max-height', type=float, default=None,
                       help='Maximum height for visualization clipping (in meters)')
    
    args = parser.parse_args()
    
    # Setup device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    # Load model
    depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{args.encoder}14').to(DEVICE).eval()
    
    # Define preprocessing transforms
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    # Prepare file list
    if os.path.isfile(args.img_path):
        filenames = [args.img_path]
    else:
        filenames = []
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.PNG', '.JPG', '.JPEG', '.TIF', '.TIFF']
        for filename in os.listdir(args.img_path):
            if any(filename.endswith(ext) for ext in extensions):
                filenames.append(os.path.join(args.img_path, filename))
        filenames.sort()
    
    if not filenames:
        print("No valid image files found!")
        return
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"Processing {len(filenames)} images...")
    
    for filename in tqdm(filenames):
        try:
            # Load and preprocess image
            raw_image = cv2.imread(filename)
            if raw_image is None:
                print(f"Warning: Could not load {filename}")
                continue
                
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            h, w = image.shape[:2]
            
            # Apply transforms
            image_tensor = transform({'image': image})['image']
            image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(DEVICE)
            
            # Inference
            with torch.no_grad():
                height_pred = depth_anything(image_tensor)
            
            # Post-process
            height_pred = F.interpolate(height_pred[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            height_pred = height_pred.cpu().numpy()
            
            # Apply height scaling
            height_pred = height_pred * args.height_scale
            
            # For remote sensing, we might want to invert the depth to get height above ground
            # This depends on your specific use case and coordinate system
            # height_pred = np.max(height_pred) - height_pred  # Uncomment if needed
            
            # Normalize for visualization (0-255 range)
            height_vis = height_pred.copy()
            if args.max_height is not None:
                height_vis = np.clip(height_vis, 0, args.max_height)
            
            height_vis = (height_vis - height_vis.min()) / (height_vis.max() - height_vis.min()) * 255.0
            height_vis = height_vis.astype(np.uint8)
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(filename))[0]
            
            # Save raw height values if requested
            if args.save_raw:
                raw_output_path = os.path.join(args.outdir, f"{base_name}_height.npy")
                np.save(raw_output_path, height_pred)
            
            if args.pred_only:
                # Save only the height prediction
                if args.grayscale:
                    output_path = os.path.join(args.outdir, f"{base_name}_height.png")
                    cv2.imwrite(output_path, height_vis)
                else:
                    # Apply colormap for better visualization
                    colored_height = cv2.applyColorMap(height_vis, cv2.COLORMAP_PLASMA)
                    output_path = os.path.join(args.outdir, f"{base_name}_height_colored.png")
                    cv2.imwrite(output_path, colored_height)
            else:
                # Create side-by-side visualization
                margin_width = 50
                caption_height = 60
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                
                # Prepare images for concatenation
                raw_image_resized = cv2.resize(raw_image, (w, h))
                
                if args.grayscale:
                    height_colored = cv2.cvtColor(height_vis, cv2.COLOR_GRAY2BGR)
                else:
                    height_colored = cv2.applyColorMap(height_vis, cv2.COLORMAP_PLASMA)
                
                # Create combined image
                combined_width = w * 2 + margin_width * 3
                combined_height = h + caption_height + margin_width * 2
                
                combined_img = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
                
                # Place images
                combined_img[caption_height + margin_width:caption_height + margin_width + h, 
                           margin_width:margin_width + w] = raw_image_resized
                combined_img[caption_height + margin_width:caption_height + margin_width + h, 
                           margin_width * 2 + w:margin_width * 2 + w * 2] = height_colored
                
                # Add captions
                cv2.putText(combined_img, "Original RGB", (margin_width, 35), 
                          font, font_scale, (0, 0, 0), font_thickness)
                cv2.putText(combined_img, "Height Map", (margin_width * 2 + w, 35), 
                          font, font_scale, (0, 0, 0), font_thickness)
                
                # Add height statistics
                stats_text = f"Height range: {height_pred.min():.1f}m - {height_pred.max():.1f}m"
                cv2.putText(combined_img, stats_text, (margin_width, combined_height - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                output_path = os.path.join(args.outdir, f"{base_name}_comparison.png")
                cv2.imwrite(output_path, combined_img)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"Processing complete! Results saved to {args.outdir}")


if __name__ == '__main__':
    run_rs_inference()