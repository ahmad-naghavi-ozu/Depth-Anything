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
                    resize_target=False,
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
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        dsm_path = os.path.join(self.dsm_dir, self.dsm_files[idx])

        # Load RGB image (assuming shape H x W x 3)
        rgb = tifffile.imread(rgb_path).astype(np.float32)
        if rgb.ndim == 2:
            # If grayscale, convert to RGB
            rgb = np.stack([rgb] * 3, axis=-1)
        elif rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)

        # Normalize RGB to [0, 1]
        rgb = rgb / 255.0

        # Load DSM (height map, shape H x W)
        dsm = tifffile.imread(dsm_path).astype(np.float32)

        # Apply transform to RGB
        rgb_transformed = self.transform({'image': rgb})['image']  # Should be tensor C x H x W

        # Resize DSM to match the transformed RGB size (518 x 518)
        # 518 = 14 * 37, perfect multiple for ViT patches
        dsm_resized = torch.from_numpy(dsm).unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W
        dsm_resized = torch.nn.functional.interpolate(dsm_resized, size=(518, 518), mode='bilinear', align_corners=False)
        dsm_resized = dsm_resized.squeeze()  # H x W

        return rgb_transformed, dsm_resized