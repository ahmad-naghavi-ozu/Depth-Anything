#!/usr/bin/env python3

"""
Remote Sensing Dataset Utilities
Provides tools for validating, converting, and preparing RS datasets
"""

import os
import glob
import numpy as np
from PIL import Image
import argparse
import sys
from pathlib import Path


def validate_dataset_structure(dataset_root):
    """
    Validate that the RS dataset follows the expected structure
    
    Expected structure:
    dataset_root/
    ├── train/
    │   ├── dsm/
    │   ├── rgb/
    │   └── sem/ (optional)
    ├── valid/
    │   ├── dsm/
    │   ├── rgb/
    │   └── sem/ (optional)
    └── test/
        ├── dsm/
        ├── rgb/
        └── sem/ (optional)
    """
    dataset_root = Path(dataset_root)
    
    if not dataset_root.exists():
        print(f"Error: Dataset root {dataset_root} does not exist!")
        return False
    
    splits = ['train', 'valid', 'test']
    required_folders = ['rgb', 'dsm']
    optional_folders = ['sem']
    
    print(f"Validating dataset structure at: {dataset_root}")
    print("=" * 60)
    
    valid_structure = True
    
    for split in splits:
        split_path = dataset_root / split
        print(f"\nChecking {split} split:")
        
        if not split_path.exists():
            print(f"  ❌ Missing {split} directory")
            valid_structure = False
            continue
        else:
            print(f"  ✅ {split} directory exists")
        
        # Check required folders
        for folder in required_folders:
            folder_path = split_path / folder
            if not folder_path.exists():
                print(f"    ❌ Missing required {folder} folder")
                valid_structure = False
            else:
                # Count files
                extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
                file_count = 0
                for ext in extensions:
                    file_count += len(list(folder_path.glob(ext)))
                    file_count += len(list(folder_path.glob(ext.upper())))
                
                print(f"    ✅ {folder} folder: {file_count} files")
                
                if file_count == 0:
                    print(f"    ⚠️  Warning: No image files found in {folder}")
        
        # Check optional folders
        for folder in optional_folders:
            folder_path = split_path / folder
            if folder_path.exists():
                extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
                file_count = 0
                for ext in extensions:
                    file_count += len(list(folder_path.glob(ext)))
                    file_count += len(list(folder_path.glob(ext.upper())))
                print(f"    ✅ {folder} folder (optional): {file_count} files")
    
    print("\n" + "=" * 60)
    if valid_structure:
        print("✅ Dataset structure validation PASSED")
        return True
    else:
        print("❌ Dataset structure validation FAILED")
        return False


def analyze_dataset_statistics(dataset_root):
    """
    Analyze the RS dataset and provide statistics about images and height values
    """
    dataset_root = Path(dataset_root)
    splits = ['train', 'valid', 'test']
    
    print(f"Analyzing dataset statistics for: {dataset_root}")
    print("=" * 60)
    
    for split in splits:
        split_path = dataset_root / split
        if not split_path.exists():
            continue
            
        print(f"\n{split.upper()} SPLIT:")
        
        rgb_path = split_path / "rgb"
        dsm_path = split_path / "dsm"
        
        if not rgb_path.exists() or not dsm_path.exists():
            print("  Skipping - missing required directories")
            continue
        
        # Get all image files
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
        rgb_files = []
        dsm_files = []
        
        for ext in extensions:
            rgb_files.extend(list(rgb_path.glob(ext)))
            rgb_files.extend(list(rgb_path.glob(ext.upper())))
            dsm_files.extend(list(dsm_path.glob(ext)))
            dsm_files.extend(list(dsm_path.glob(ext.upper())))
        
        print(f"  RGB files: {len(rgb_files)}")
        print(f"  DSM files: {len(dsm_files)}")
        
        if len(rgb_files) == 0 or len(dsm_files) == 0:
            print("  No files to analyze")
            continue
        
        # Sample a few files for analysis
        sample_size = min(10, len(dsm_files))
        sample_dsm_files = sorted(dsm_files)[:sample_size]
        sample_rgb_files = sorted(rgb_files)[:sample_size]
        
        print(f"  Analyzing {sample_size} sample files...")
        
        # RGB statistics
        rgb_shapes = []
        for rgb_file in sample_rgb_files[:5]:  # Analyze first 5
            try:
                img = Image.open(rgb_file)
                rgb_shapes.append((img.width, img.height, len(img.getbands())))
                img.close()
            except Exception as e:
                print(f"    Error loading RGB {rgb_file}: {e}")
        
        if rgb_shapes:
            print(f"  RGB image shapes (W×H×C): {set(rgb_shapes)}")
        
        # DSM statistics  
        height_stats = []
        dsm_shapes = []
        
        for dsm_file in sample_dsm_files:
            try:
                img = Image.open(dsm_file)
                dsm_shapes.append((img.width, img.height))
                
                # Convert to numpy for statistics
                if img.mode in ['I', 'I;16', 'F']:
                    height_data = np.asarray(img, dtype=np.float32)
                else:
                    height_data = np.asarray(img.convert('L'), dtype=np.float32)
                
                # Filter out potential no-data values
                valid_heights = height_data[(height_data >= 0) & (height_data < 10000)]
                
                if len(valid_heights) > 0:
                    height_stats.extend([
                        float(valid_heights.min()),
                        float(valid_heights.max()),
                        float(valid_heights.mean())
                    ])
                
                img.close()
                
            except Exception as e:
                print(f"    Error loading DSM {dsm_file}: {e}")
        
        if dsm_shapes:
            print(f"  DSM image shapes (W×H): {set(dsm_shapes)}")
        
        if height_stats:
            print(f"  Height statistics:")
            print(f"    Min height: {min(height_stats):.2f}")
            print(f"    Max height: {max(height_stats):.2f}")
            print(f"    Mean height: {np.mean(height_stats):.2f}")


def create_sample_dataset(output_dir, num_samples=10):
    """
    Create a small sample dataset for testing purposes
    """
    output_dir = Path(output_dir)
    print(f"Creating sample RS dataset at: {output_dir}")
    
    splits = ['train', 'valid', 'test']
    split_samples = {'train': num_samples, 'valid': max(1, num_samples//5), 'test': max(1, num_samples//5)}
    
    for split in splits:
        split_path = output_dir / split
        rgb_path = split_path / "rgb"
        dsm_path = split_path / "dsm"
        
        # Create directories
        rgb_path.mkdir(parents=True, exist_ok=True)
        dsm_path.mkdir(parents=True, exist_ok=True)
        
        n_samples = split_samples[split]
        
        for i in range(n_samples):
            # Create synthetic RGB image (512x512)
            rgb_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            rgb_pil = Image.fromarray(rgb_img)
            rgb_pil.save(rgb_path / f"sample_{i:03d}.png")
            
            # Create synthetic DSM (512x512) - simulate building heights
            height_map = np.zeros((512, 512), dtype=np.float32)
            
            # Add some random "buildings"
            for _ in range(np.random.randint(5, 15)):
                x = np.random.randint(50, 462)
                y = np.random.randint(50, 462)
                w = np.random.randint(20, 100)
                h = np.random.randint(20, 100)
                height = np.random.uniform(5, 50)  # Building height in meters
                
                height_map[y:y+h, x:x+w] = height
            
            # Add some noise
            height_map += np.random.normal(0, 0.5, (512, 512))
            height_map = np.clip(height_map, 0, 100)
            
            # Save as 16-bit image for better precision
            height_img = (height_map * 100).astype(np.uint16)  # Scale by 100 for storage
            Image.fromarray(height_img).save(dsm_path / f"sample_{i:03d}.tif")
    
    print(f"Sample dataset created with {sum(split_samples.values())} total samples")
    print(f"Note: This is synthetic data for testing. Heights are scaled by 100 in storage.")


def main():
    parser = argparse.ArgumentParser(description='Remote Sensing Dataset Utilities')
    parser.add_argument('action', choices=['validate', 'analyze', 'create-sample'],
                       help='Action to perform')
    parser.add_argument('--dataset-root', type=str, required=True,
                       help='Root directory of the RS dataset')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to create (for create-sample action)')
    
    args = parser.parse_args()
    
    if args.action == 'validate':
        success = validate_dataset_structure(args.dataset_root)
        sys.exit(0 if success else 1)
        
    elif args.action == 'analyze':
        if not validate_dataset_structure(args.dataset_root):
            print("\nDataset validation failed. Fix structure before analysis.")
            sys.exit(1)
        analyze_dataset_statistics(args.dataset_root)
        
    elif args.action == 'create-sample':
        create_sample_dataset(args.dataset_root, args.num_samples)


if __name__ == '__main__':
    main()