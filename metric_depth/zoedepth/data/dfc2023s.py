"""
DFC2023S Remote Sensing Dataset for Height Estimation
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
        sem_mask = sample.get('sem_mask', None)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW
        depth = torch.from_numpy(depth).unsqueeze(0).float()  # HW -> 1HW
        if mask is not None:
            mask = torch.from_numpy(mask).unsqueeze(0).float()  # HW -> 1HW
        if sem_mask is not None:
            sem_mask = torch.from_numpy(sem_mask).unsqueeze(0).float()  # HW -> 1HW
        
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
            if sem_mask is not None:
                sem_mask = torch.nn.functional.interpolate(
                    sem_mask.unsqueeze(0), 
                    size=self.resize_shape, 
                    mode='nearest'
                ).squeeze(0)
        
        sample['image'] = image
        sample['depth'] = depth
        if mask is not None:
            sample['mask'] = mask
        if sem_mask is not None:
            sample['sem_mask'] = sem_mask
        
        return sample


class DFC2023S(Dataset):
    """DFC2023S Dataset for remote sensing height estimation"""
    
    def __init__(self, data_dir_root, split='train', resize_shape=None):
        """
        Args:
            data_dir_root: Root directory containing train/val/test folders
            split: 'train', 'val', or 'test'
            resize_shape: Tuple (H, W) for resizing, or None for original size
        """
        self.data_dir_root = data_dir_root
        self.split = split
        
        # Paths to RGB and DSM
        rgb_dir = os.path.join(data_dir_root, split, 'rgb')
        dsm_dir = os.path.join(data_dir_root, split, 'dsm')
        
        if not os.path.exists(rgb_dir):
            raise ValueError(f"RGB directory not found: {rgb_dir}")
        if not os.path.exists(dsm_dir):
            raise ValueError(f"DSM directory not found: {dsm_dir}")
        
        # Get sorted file lists
        self.image_files = sorted(glob.glob(os.path.join(rgb_dir, '*.tif')))
        self.depth_files = sorted(glob.glob(os.path.join(dsm_dir, '*.tif')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No .tif files found in {rgb_dir}")
        if len(self.image_files) != len(self.depth_files):
            raise ValueError(
                f"Mismatch: {len(self.image_files)} RGB files, "
                f"{len(self.depth_files)} DSM files"
            )
        
        self.transform = ToTensor(resize_shape=resize_shape)
        
        print(f"Loaded DFC2023S {split} set: {len(self.image_files)} samples")

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]
        
        # Extract filename without extension for reference
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Load RGB image (3-band TIF)
        image = tifffile.imread(image_path).astype(np.float32)
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Ensure HWC format
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[0] == 3:  # CHW -> HWC
            image = image.transpose(1, 2, 0)
        
        # Load DSM (height in meters)
        depth = tifffile.imread(depth_path).astype(np.float32)
        
        # Handle invalid values (set to 0 or mask later)
        depth[depth < 0] = 0.0
        
        # Ensure 2D
        if depth.ndim == 3:
            depth = depth.squeeze()
        
        # Create mask for valid pixels
        # In RS height estimation, height=0 is VALID (ground-level/background pixels)
        # Buildings have height > 0
        # No invalid values (NaN/negative) exist due to preprocessing
        # Use >= 0 to include ALL pixels (including ground at height=0)
        # SILog loss has built-in log-safety (alpha=1e-7) for numerical stability
        mask = (depth >= 0).astype(np.float32)
        
        # Try to load semantic mask if available
        sem_mask = None
        sem_dir = os.path.join(self.data_dir_root, self.split, 'sem')
        sem_path = os.path.join(sem_dir, f"{filename}.tif")
        if os.path.exists(sem_path):
            try:
                sem_mask = tifffile.imread(sem_path).astype(np.float32)
                # Ensure 2D
                if sem_mask.ndim == 3:
                    sem_mask = sem_mask.squeeze()
            except Exception as e:
                print(f"Warning: Could not load semantic mask {sem_path}: {e}")
        
        sample = {
            'image': image,
            'depth': depth,
            'mask': mask,
            'filename': filename,
            'dataset': 'dfc2023s'
        }
        
        if sem_mask is not None:
            sample['sem_mask'] = sem_mask
        
        sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return len(self.image_files)


def get_dfc2023s_loader(data_dir_root, split='train', batch_size=1, 
                        resize_shape=None, num_workers=4, **kwargs):
    """
    Create DataLoader for DFC2023S dataset
    
    Args:
        data_dir_root: Root directory of DFC2023S dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        resize_shape: Tuple (H, W) for resizing
        num_workers: Number of dataloader workers
        
    Returns:
        DataLoader instance
    """
    dataset = DFC2023S(data_dir_root, split=split, resize_shape=resize_shape)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
