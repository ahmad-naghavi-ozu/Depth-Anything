import torch
from torch.utils.data import Dataset
import tifffile
import os
import numpy as np
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2


class RemoteSensingHeightDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to the dataset root, e.g., 'DFC2023S'
            split (str): 'train', 'valid', or 'test'
            transform: Optional transform to apply to the data
        """
        self.root_dir = root_dir
        self.split = split
        self.rgb_dir = os.path.join(root_dir, split, 'rgb')
        self.dsm_dir = os.path.join(root_dir, split, 'dsm')

        if not os.path.exists(self.rgb_dir):
            raise ValueError(f"RGB directory {self.rgb_dir} does not exist")
        if not os.path.exists(self.dsm_dir):
            raise ValueError(f"DSM directory {self.dsm_dir} does not exist")

        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.tif')])
        self.dsm_files = sorted([f for f in os.listdir(self.dsm_dir) if f.endswith('.tif')])

        if len(self.rgb_files) != len(self.dsm_files):
            raise ValueError(f"Number of RGB files ({len(self.rgb_files)}) does not match DSM files ({len(self.dsm_files)})")

        # Default transform similar to Depth_Anything inference
        # 518 = 14 * 37, ensuring perfect alignment with ViT patch size
        if transform is None:
            self.transform = Compose([
                Resize(
                    width=518,
                    height=518,
                    resize_target=True,  # IMPORTANT: Resize DSM to match RGB size
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Get corresponding RGB and DSM filenames
        rgb_filename = os.path.join(self.rgb_dir, self.rgb_files[idx])
        base_name = self.rgb_files[idx]
        dsm_filename = os.path.join(self.dsm_dir, base_name)
        
        # Load RGB image
        rgb = tifffile.imread(rgb_filename).astype(np.float32)
        
        # Load DSM (Digital Surface Model) as height ground truth
        dsm = tifffile.imread(dsm_filename).astype(np.float32)
        
        # Check for invalid values (NaN, Inf) in DSM
        if np.isnan(dsm).any() or np.isinf(dsm).any():
            print(f"Warning: Invalid values in DSM {dsm_filename}")
            dsm = np.nan_to_num(dsm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply transform
        if self.transform:
            transformed = self.transform({'image': rgb, 'depth': dsm})
            rgb = transformed['image']
            dsm = transformed['depth']
        
        return rgb, dsm