"""
DFC2019_crp512_bin_mini Remote Sensing Dataset for Height Estimation
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


class DFC2019crp512binmini(Dataset):
    """DFC2019_crp512_bin_mini Dataset for remote sensing height estimation"""
    
    def __init__(self, data_dir_root, split='train', resize_shape=None):
        """
        Args:
            data_dir_root: Root directory of DFC2019_crp512_bin_mini dataset
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
        
        assert len(self.rgb_files) == len(self.dsm_files), \
            f"Mismatch: {len(self.rgb_files)} RGB vs {len(self.dsm_files)} DSM files"
        
        print(f"DFC2019_crp512_bin_mini {split} set: {len(self.rgb_files)} samples")
        
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
        
        sample = {
            'image': rgb,
            'depth': dsm,
            'mask': mask,
            'filename': filename,
            'dataset': 'dfc2019_crp512_bin_mini'
        }
        
        # Apply transforms
        sample = self.transform(sample)
        
        return sample


def get_dfc2019_crp512_bin_mini_loader(data_dir_root, split='train', batch_size=4, 
                              num_workers=4, resize_shape=(512, 512), **kwargs):
    """Create DataLoader for DFC2019_crp512_bin_mini"""
    dataset = DFC2019crp512binmini(
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
