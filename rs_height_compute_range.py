#!/usr/bin/env python3
"""
Compute min/max height values from remote sensing DSM data
Works with datasets following the structure: <dataset_root>/train/dsm/*.tif
Also generates necessary configuration files and dataset loader for new datasets
"""
import os
import sys
import argparse
import numpy as np
import tifffile
from tqdm import tqdm
import json
import shutil

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


def setup_dataset_config(dataset_name, dataset_root, min_height, max_height, force=False):
    """Generate configuration and dataset loader files for a new dataset"""
    
    print("\n" + "="*70)
    print("Setting up dataset configuration files...")
    print("="*70)
    
    # Convert dataset name to lowercase for config key
    dataset_key = dataset_name.lower()
    dataset_var = dataset_key  # e.g., dfc2019_crp512_bin
    
    # Paths
    config_file = "metric_depth/zoedepth/utils/config.py"
    data_mono_file = "metric_depth/zoedepth/data/data_mono.py"
    dataset_loader_file = f"metric_depth/zoedepth/data/{dataset_key}.py"
    
    # Round max_height up to nearest 10
    max_depth = int(np.ceil(max_height / 10) * 10)
    
    # 1. Create dataset loader file
    print(f"\n1. Creating dataset loader: {dataset_loader_file}")
    if os.path.exists(dataset_loader_file) and not force:
        print(f"   ⚠️  File already exists. Use --force to overwrite.")
    else:
        template_content = f'''"""
{dataset_name} Remote Sensing Dataset for Height Estimation
Compatible with ZoeDepth training pipeline
"""

import os
import glob
import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ToTensor(object):
    """Convert numpy arrays to tensors"""
    def __init__(self, resize_shape=None):
        self.resize_shape = resize_shape
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        mask = sample.get('mask', None)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW
        depth = torch.from_numpy(depth).unsqueeze(0).float()  # HW -> 1HW
        if mask is not None:
            mask = torch.from_numpy(mask).unsqueeze(0).float()  # HW -> 1HW
        
        # Normalize image
        image = self.normalize(image)
        
        # Resize if specified
        if self.resize_shape is not None:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), 
                size=self.resize_shape, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), 
                size=self.resize_shape, 
                mode='nearest'
            ).squeeze(0)
            if mask is not None:
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0), 
                    size=self.resize_shape, 
                    mode='nearest'
                ).squeeze(0)
        
        sample['image'] = image
        sample['depth'] = depth
        if mask is not None:
            sample['mask'] = mask
        
        return sample


class {dataset_name.replace('_', '').replace('-', '')}(Dataset):
    """{dataset_name} Dataset for remote sensing height estimation"""
    
    def __init__(self, data_dir_root, split='train', resize_shape=None):
        """
        Args:
            data_dir_root: Root directory of {dataset_name} dataset
            split: 'train', 'val', or 'test'
            resize_shape: (H, W) tuple for resizing, or None
        """
        self.data_dir_root = data_dir_root
        self.split = split
        self.resize_shape = resize_shape
        
        # Paths
        self.rgb_dir = os.path.join(data_dir_root, split, 'rgb')
        self.dsm_dir = os.path.join(data_dir_root, split, 'dsm')
        
        # Get file lists
        self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, '*.tif')))
        self.dsm_files = sorted(glob.glob(os.path.join(self.dsm_dir, '*.tif')))
        
        assert len(self.rgb_files) == len(self.dsm_files), \\
            f"Mismatch: {{len(self.rgb_files)}} RGB vs {{len(self.dsm_files)}} DSM files"
        
        print(f"{dataset_name} {{split}} set: {{len(self.rgb_files)}} samples")
        
        self.transform = ToTensor(resize_shape=resize_shape)
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        # Load RGB image
        rgb_path = self.rgb_files[idx]
        rgb = tifffile.imread(rgb_path).astype(np.float32)
        
        # Ensure RGB has 3 channels
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        elif rgb.shape[-1] == 4:  # RGBA
            rgb = rgb[:, :, :3]
        
        # Normalize to [0, 1]
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        # Load DSM (height/depth map)
        dsm_path = self.dsm_files[idx]
        dsm = tifffile.imread(dsm_path).astype(np.float32)
        
        # Create valid mask (depths > 0)
        mask = (dsm > 0).astype(np.float32)
        
        # Get filename for tracking
        filename = os.path.splitext(os.path.basename(rgb_path))[0]
        
        sample = {{
            'image': rgb,
            'depth': dsm,
            'mask': mask,
            'filename': filename,
            'dataset': '{dataset_key}'
        }}
        
        # Apply transforms
        sample = self.transform(sample)
        
        return sample


def get_{dataset_var}_loader(data_dir_root, split='train', batch_size=4, 
                              num_workers=4, resize_shape=(512, 512), **kwargs):
    """Create DataLoader for {dataset_name}"""
    dataset = {dataset_name.replace('_', '').replace('-', '')}(
        data_dir_root=data_dir_root,
        split=split,
        resize_shape=resize_shape
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
'''
        
        with open(dataset_loader_file, 'w') as f:
            f.write(template_content)
        print(f"   ✓ Created: {dataset_loader_file}")
    
    # 2. Add import to data_mono.py
    print(f"\n2. Adding import to: {data_mono_file}")
    with open(data_mono_file, 'r') as f:
        data_mono_content = f.read()
    
    import_line = f"from .{dataset_key} import get_{dataset_var}_loader"
    
    if import_line in data_mono_content:
        print(f"   ⚠️  Import already exists")
    else:
        # Find the line with dfc2023s import and add after it
        lines = data_mono_content.split('\n')
        insert_idx = None
        for i, line in enumerate(lines):
            if 'from .dfc2023s import' in line or 'from .dfc2023mini import' in line:
                insert_idx = i + 1
        
        if insert_idx:
            lines.insert(insert_idx, import_line)
            with open(data_mono_file, 'w') as f:
                f.write('\n'.join(lines))
            print(f"   ✓ Added import statement")
        else:
            print(f"   ⚠️  Could not find insertion point. Please add manually:")
            print(f"      {import_line}")
    
    # 3. Add dataset loading logic to data_mono.py
    print(f"\n3. Adding dataset loading logic to: {data_mono_file}")
    with open(data_mono_file, 'r') as f:
        data_mono_content = f.read()
    
    dataset_logic = f'''        elif config.dataset == '{dataset_key}':
            self.data = get_{dataset_var}_loader(
                data_dir_root=config.{dataset_var}_root,
                split=mode,
                batch_size=config.batch_size,
                num_workers=config.workers,
                resize_shape=(config.input_height, config.input_width)
            )
            return
'''
    
    if f"config.dataset == '{dataset_key}'" in data_mono_content:
        print(f"   ⚠️  Dataset loading logic already exists")
    else:
        # Find dfc2023mini logic and add after it
        lines = data_mono_content.split('\n')
        insert_idx = None
        for i, line in enumerate(lines):
            if "config.dataset == 'dfc2023mini'" in line:
                # Find the return statement after this block
                for j in range(i, len(lines)):
                    if lines[j].strip() == 'return':
                        insert_idx = j + 1  # After the return statement
                        break
        
        if insert_idx:
            # Insert blank line then new dataset logic
            lines.insert(insert_idx, '')
            for line in reversed(dataset_logic.split('\n')):
                lines.insert(insert_idx + 1, line)
            
            with open(data_mono_file, 'w') as f:
                f.write('\n'.join(lines))
            print(f"   ✓ Added dataset loading logic")
        else:
            print(f"   ⚠️  Could not find insertion point. Please add manually after dfc2023mini block:")
            print(dataset_logic)
    
    # 4. Add config entry to config.py
    print(f"\n4. Adding config entry to: {config_file}")
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    config_entry = f'''    "{dataset_key}": {{
        "dataset": "{dataset_key}",
        "{dataset_var}_root": "{dataset_root}",
        "save_dir": os.path.expanduser("./checkpoints/rs_height_zoedepth/{dataset_name}"),
        "min_depth": 0.0,
        "max_depth": {max_depth}.0,
        "min_depth_eval": 0.0,
        "max_depth_eval": {max_depth}.0,
        "input_height": 512,
        "input_width": 512,
        "do_random_rotate": False,
        "degree": 0.0,
        "do_kb_crop": False,
        "garg_crop": False,
        "eigen_crop": False,
    }},'''
    
    if f'"{dataset_key}":' in config_content and f'{dataset_var}_root' in config_content:
        print(f"   ⚠️  Config entry already exists")
    else:
        # Find dfc2023mini entry and add after it
        lines = config_content.split('\n')
        insert_idx = None
        for i, line in enumerate(lines):
            if '"dfc2023mini":' in line:
                # Find the closing brace
                for j in range(i, len(lines)):
                    if lines[j].strip().startswith('}') and 'eigen_crop' in lines[j-1]:
                        insert_idx = j + 1
                        break
        
        if insert_idx:
            for line in reversed(config_entry.split('\n')):
                lines.insert(insert_idx, line)
            
            with open(config_file, 'w') as f:
                f.write('\n'.join(lines))
            print(f"   ✓ Added config entry")
        else:
            print(f"   ⚠️  Could not find insertion point. Please add manually after dfc2023mini:")
            print(config_entry)
    
    # 5. Update ALL_REMOTE_SENSING list
    print(f"\n5. Updating ALL_REMOTE_SENSING list in: {config_file}")
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    if f'"{dataset_key}"' in config_content and 'ALL_REMOTE_SENSING' in config_content:
        lines = config_content.split('\n')
        for i, line in enumerate(lines):
            if 'ALL_REMOTE_SENSING = [' in line:
                if dataset_key not in line:
                    # Add to the list
                    lines[i] = line.rstrip(']') + f', "{dataset_key}"]'
                    with open(config_file, 'w') as f:
                        f.write('\n'.join(lines))
                    print(f"   ✓ Added '{dataset_key}' to ALL_REMOTE_SENSING")
                else:
                    print(f"   ⚠️  Already in ALL_REMOTE_SENSING list")
                break
    
    # 6. Update get_config choices
    print(f"\n6. Updating dataset choices in get_config function")
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    if f'check_choices("Dataset", dataset,' in config_content:
        lines = config_content.split('\n')
        for i, line in enumerate(lines):
            if 'check_choices("Dataset", dataset,' in line and 'mode == "train"' in lines[i-1]:
                if f'"{dataset_key}"' not in line:
                    # Add to the list before the closing bracket
                    lines[i] = line.rstrip('])') + f', "{dataset_key}"])'
                    with open(config_file, 'w') as f:
                        f.write('\n'.join(lines))
                    print(f"   ✓ Added '{dataset_key}' to dataset choices")
                else:
                    print(f"   ⚠️  Already in dataset choices")
                break
    
    print("\n" + "="*70)
    print("✓ Setup complete!")
    print("="*70)
    print(f"\nYou can now train on {dataset_name} using:")
    print(f"  ./rs_height_train.sh --dataset {dataset_key}")
    print(f"\nOr using Python directly:")
    print(f"  python rs_height_train.py --model zoedepth --dataset {dataset_key}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute min/max height values from training DSM data'
    )
    parser.add_argument(
        'dataset_root',
        type=str,
        nargs='?',
        default=None,
        help='Root directory of the dataset (e.g., /home/asfand/Ahmad/datasets/DFC2023S)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset name (e.g., DFC2023S, DFC2019_crp512_bin, Huawei_Contest). Will use /home/asfand/Ahmad/datasets/<dataset>'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to analyze (default: train)'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Generate configuration and dataset loader files for this dataset'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing configuration files'
    )
    
    args = parser.parse_args()
    
    # Determine dataset root path
    if args.dataset_root is None and args.dataset is None:
        parser.error("Either provide dataset_root path or --dataset name")
    
    if args.dataset_root is None:
        # Use default datasets path with dataset name
        args.dataset_root = f'/home/asfand/Ahmad/datasets/{args.dataset}'
    
    # Construct DSM directory path
    train_dsm_dir = os.path.join(args.dataset_root, args.split, 'dsm')
    
    if not os.path.exists(train_dsm_dir):
        print(f"ERROR: DSM directory not found: {train_dsm_dir}")
        print(f"Please ensure the dataset follows this structure:")
        print(f"  {args.dataset_root}/")
        print(f"    {args.split}/")
        print(f"      dsm/  ← DSM .tif files should be here")
        sys.exit(1)
    
    # Extract dataset name from path
    dataset_name = os.path.basename(args.dataset_root.rstrip('/'))
    
    min_height, max_height = compute_dsm_range(train_dsm_dir)
    
    print("\n" + "="*70)
    print(f"DSM Height Range Statistics - {dataset_name} ({args.split} split)")
    print("="*70)
    print(f"Minimum height: {min_height:.4f} meters")
    print(f"Maximum height: {max_height:.4f} meters")
    print(f"Range: {max_height - min_height:.4f} meters")
    print("="*70)
    
    # Suggest bin configuration
    height_range = max_height - min_height
    print("\nRecommended ZoeDepth configuration:")
    print(f"  min_depth = {min_height:.2f}")
    print(f"  max_depth = {max_height:.2f}")
    print(f"  n_bins = 64  (default, ~{height_range/64:.2f}m per bin)")
    print(f"  n_bins = 128 (medium resolution, ~{height_range/128:.2f}m per bin)")
    print(f"  n_bins = 256 (high resolution, ~{height_range/256:.2f}m per bin)")
    print("="*70)
    
    # Setup dataset configuration if requested
    if args.setup:
        setup_dataset_config(
            dataset_name=dataset_name,
            dataset_root=args.dataset_root,
            min_height=min_height,
            max_height=max_height,
            force=args.force
        )
