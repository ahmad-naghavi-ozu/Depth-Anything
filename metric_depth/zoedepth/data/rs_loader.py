# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Adapted for Remote Sensing Height Estimation

import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):
    def __init__(self):
        # No normalization applied, keeping consistency with original
        self.normalize = lambda x : x

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = self.to_tensor(depth)

        return {'image': image, 'depth': depth, 'dataset': "remote_sensing"}

    def to_tensor(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class RemoteSensingDataset(Dataset):
    def __init__(self, data_dir_root, split="train", height_scale_factor=1.0, max_height=None):
        """
        Remote Sensing Dataset Loader
        
        Args:
            data_dir_root (str): Root directory containing the dataset
            split (str): Dataset split - "train", "valid", or "test"
            height_scale_factor (float): Scale factor to convert DSM values to meters (default: 1.0)
            max_height (float): Maximum height value in meters. Heights above this will be clamped.
        
        Expected directory structure:
        data_dir_root/
        ├── train/valid/test/
        │   ├── dsm/     # Digital Surface Model (height maps)
        │   ├── rgb/     # RGB imagery
        │   └── sem/     # semantic labels (ignored for now)
        """
        self.data_dir_root = data_dir_root
        self.split = split
        self.height_scale_factor = height_scale_factor
        self.max_height = max_height
        
        split_dir = os.path.join(data_dir_root, split)
        rgb_dir = os.path.join(split_dir, "rgb")
        dsm_dir = os.path.join(split_dir, "dsm")
        
        # Check if directories exist
        if not os.path.exists(rgb_dir):
            raise ValueError(f"RGB directory not found: {rgb_dir}")
        if not os.path.exists(dsm_dir):
            raise ValueError(f"DSM directory not found: {dsm_dir}")
        
        # Get all RGB image files
        self.image_files = []
        self.height_files = []
        
        # Support common image formats
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            self.image_files.extend(glob.glob(os.path.join(rgb_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(rgb_dir, ext.upper())))
        
        # Sort for consistency
        self.image_files.sort()
        
        # Generate corresponding height file paths
        for img_path in self.image_files:
            img_name = os.path.basename(img_path)
            # Remove extension and add common height map extensions
            img_base = os.path.splitext(img_name)[0]
            
            # Try different height map extensions
            height_path = None
            for ext in ['.png', '.tif', '.tiff']:
                potential_path = os.path.join(dsm_dir, img_base + ext)
                if os.path.exists(potential_path):
                    height_path = potential_path
                    break
            
            if height_path is None:
                print(f"Warning: No height map found for {img_path}")
                continue
            
            self.height_files.append(height_path)
        
        # Ensure we have matching pairs
        if len(self.image_files) != len(self.height_files):
            min_len = min(len(self.image_files), len(self.height_files))
            self.image_files = self.image_files[:min_len]
            self.height_files = self.height_files[:min_len]
        
        print(f"Loaded {len(self.image_files)} RGB-Height pairs from {split} split")
        
        self.transform = ToTensor()

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        height_path = self.height_files[idx]

        # Load RGB image
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32) / 255.0
        
        # Load height map (DSM)
        height_img = Image.open(height_path)
        
        # Handle different height map formats
        if height_img.mode in ['I', 'I;16', 'F']:
            # 16-bit or float images
            height = np.asarray(height_img, dtype=np.float32)
        else:
            # Convert to grayscale if needed and then to float
            if height_img.mode != 'L':
                height_img = height_img.convert('L')
            height = np.asarray(height_img, dtype=np.float32)
        
        # Apply height scaling
        height = height * self.height_scale_factor
        
        # Clamp maximum height if specified
        if self.max_height is not None:
            height = np.clip(height, 0, self.max_height)
        
        # Handle invalid/no-data values (typically represented as very high or very low values)
        # Assuming no-data values are negative or extremely high (>10000m)
        invalid_mask = (height < 0) | (height > 10000)
        height[invalid_mask] = -1  # Mark as invalid similar to other datasets
        
        # Expand dimensions to match expected format
        height = height[..., None]
        
        return self.transform(dict(image=image, depth=height))

    def __len__(self):
        return len(self.image_files)


def get_remote_sensing_loader(data_dir_root, split="train", batch_size=1, height_scale_factor=1.0, max_height=None, **kwargs):
    """
    Get data loader for remote sensing height estimation dataset
    
    Args:
        data_dir_root (str): Root directory containing the dataset
        split (str): Dataset split - "train", "valid", or "test"
        batch_size (int): Batch size for data loading
        height_scale_factor (float): Scale factor to convert DSM values to meters
        max_height (float): Maximum height value in meters
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = RemoteSensingDataset(data_dir_root, split, height_scale_factor, max_height)
    return DataLoader(dataset, batch_size, **kwargs)


if __name__ == "__main__":
    # Test the data loader
    import sys
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
        loader = get_remote_sensing_loader(data_root, split="train", batch_size=1)
        print("Total files", len(loader.dataset))
        for i, sample in enumerate(loader):
            print(f"Batch {i}:")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Height shape: {sample['depth'].shape}")
            print(f"  Dataset: {sample['dataset']}")
            print(f"  Height range: {sample['depth'].min():.2f} - {sample['depth'].max():.2f}")
            if i > 2:
                break
    else:
        print("Usage: python rs_loader.py <path_to_dataset_root>")