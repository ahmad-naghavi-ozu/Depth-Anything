"""
DFC2023mini Remote Sensing Dataset for Height Estimation (Mini/Debug Version)
Same structure as DFC2023S but with fewer samples for quick debugging
Compatible with ZoeDepth training pipeline
"""

# Reuse DFC2023S implementation with different dataset name
from torch.utils.data import DataLoader
from .dfc2023s import DFC2023S, ToTensor, get_dfc2023s_loader


class DFC2023mini(DFC2023S):
    """DFC2023mini Dataset - Mini version of DFC2023S for debugging"""
    
    def __init__(self, data_dir_root, split='train', resize_shape=(512, 512)):
        """
        Initialize DFC2023mini dataset (inherits from DFC2023S)
        
        Args:
            data_dir_root: Root directory of DFC2023mini dataset
            split: 'train', 'valid', or 'test'
            resize_shape: Target size (H, W) for resizing images
        """
        # Call parent class with DFC2023mini root
        super().__init__(data_dir_root, split=split, resize_shape=resize_shape)
        # Update dataset name in printout
        print(f"[DFC2023mini] Mini/debug version with {len(self.image_files)} samples")


def get_dfc2023mini_loader(data_dir_root, split='train', batch_size=1, 
                           num_workers=4, resize_shape=(512, 512)):
    """
    Create DataLoader for DFC2023mini dataset
    
    Args:
        data_dir_root: Root directory of DFC2023mini dataset
        split: 'train', 'valid', or 'test'
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        resize_shape: Target size (H, W) for resizing
        
    Returns:
        DataLoader instance
    """
    dataset = DFC2023mini(data_dir_root, split=split, resize_shape=resize_shape)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader
