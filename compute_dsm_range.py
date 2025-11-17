#!/usr/bin/env python3
"""
Compute min/max height values from DFC2023S training DSM data
"""
import os
import numpy as np
import tifffile
from tqdm import tqdm

def compute_dsm_range(dsm_dir):
    """Compute min and max height values from DSM files"""
    dsm_files = sorted([f for f in os.listdir(dsm_dir) if f.endswith('.tif')])
    
    if not dsm_files:
        raise ValueError(f"No .tif files found in {dsm_dir}")
    
    print(f"Found {len(dsm_files)} DSM files in {dsm_dir}")
    print("Computing height range...")
    
    global_min = float('inf')
    global_max = float('-inf')
    
    for dsm_file in tqdm(dsm_files, desc="Scanning DSM files"):
        dsm_path = os.path.join(dsm_dir, dsm_file)
        dsm = tifffile.imread(dsm_path).astype(np.float32)
        
        # Filter out invalid values (assuming valid heights are > 0)
        valid_heights = dsm[dsm > 0]
        
        if valid_heights.size > 0:
            file_min = float(np.min(valid_heights))
            file_max = float(np.max(valid_heights))
            
            global_min = min(global_min, file_min)
            global_max = max(global_max, file_max)
    
    return global_min, global_max

if __name__ == '__main__':
    # Adjust this path if needed
    dataset_root = '/home/asfand/Ahmad/datasets/DFC2023S'
    train_dsm_dir = os.path.join(dataset_root, 'train', 'dsm')
    
    if not os.path.exists(train_dsm_dir):
        print(f"ERROR: Training DSM directory not found: {train_dsm_dir}")
        print("Please provide the correct path to DFC2023S dataset")
        exit(1)
    
    min_height, max_height = compute_dsm_range(train_dsm_dir)
    
    print("\n" + "="*50)
    print("DSM Height Range Statistics")
    print("="*50)
    print(f"Minimum height: {min_height:.4f} meters")
    print(f"Maximum height: {max_height:.4f} meters")
    print(f"Range: {max_height - min_height:.4f} meters")
    print("="*50)
    
    # Suggest bin configuration
    height_range = max_height - min_height
    print("\nRecommended ZoeDepth configuration:")
    print(f"  min_depth = {min_height:.2f}")
    print(f"  max_depth = {max_height:.2f}")
    print(f"  n_bins = 64  (default)")
    print(f"  n_bins = 128 (medium resolution, ~{height_range/128:.2f}m per bin)")
    print(f"  n_bins = 256 (high resolution, ~{height_range/256:.2f}m per bin)")
