#!/usr/bin/env python3

"""
Generic Remote Sensing Dataset Configuration

This module provides configuration for various RS datasets (DFC2023, DFC2019, etc.)
with a unified interface that allows easy switching between datasets.

Supported datasets:
- DFC2023: IEEE Data Fusion Contest 2023
- DFC2023mini: Mini version for debugging
- DFC2019: IEEE Data Fusion Contest 2019  
- Custom: User-defined datasets following the same structure

Dataset structure (all datasets should follow this pattern):
[DATASET_NAME]/
├── train/
│   ├── dsm/     # Digital Surface Model files
│   ├── rgb/     # RGB imagery files
│   └── sem/     # Semantic labels/building masks (optional)
├── valid/
│   ├── dsm/
│   ├── rgb/
│   └── sem/
└── test/
    ├── dsm/
    ├── rgb/
    └── sem/
"""

import os
from typing import Dict, Optional, Tuple


class RSDatasetConfig:
    """Configuration class for Remote Sensing datasets"""
    
    # Predefined dataset configurations
    DATASET_CONFIGS = {
        'dfc2023': {
            'name': 'DFC2023',
            'description': 'IEEE Data Fusion Contest 2023',
            'default_root': '/home/asfand/Ahmad/datasets/DFC2023',
            'height_scale_factor': 1.0,
            'max_height': 150.0,
            'image_size': (512, 512),
            'file_extensions': {
                'rgb': ['.png', '.jpg', '.jpeg', '.tif', '.tiff'],
                'dsm': ['.tif', '.tiff', '.png'],
                'sem': ['.png', '.tif', '.tiff']
            },
            'wandb_settings': {
                'entity': 'ahmad-naghavi',
                'project_prefix': 'DepthAnything-RS-DFC2023'
            }
        },
        'dfc2023mini': {
            'name': 'DFC2023Mini',
            'description': 'Mini version of DFC2023 for debugging',
            'default_root': '/home/asfand/Ahmad/datasets/DFC2023mini',
            'height_scale_factor': 1.0,
            'max_height': 100.0,
            'image_size': (512, 512),
            'file_extensions': {
                'rgb': ['.png', '.jpg', '.jpeg', '.tif', '.tiff'],
                'dsm': ['.tif', '.tiff', '.png'],
                'sem': ['.png', '.tif', '.tiff']
            }
        },
        'dfc2019': {
            'name': 'DFC2019',
            'description': 'IEEE Data Fusion Contest 2019',
            'default_root': '/home/asfand/Ahmad/datasets/DFC2019',
            'height_scale_factor': 1.0,
            'max_height': 120.0,
            'image_size': (512, 512),
            'file_extensions': {
                'rgb': ['.png', '.jpg', '.jpeg', '.tif', '.tiff'],
                'dsm': ['.tif', '.tiff', '.png'],
                'sem': ['.png', '.tif', '.tiff']
            }
        }
    }
    
    def __init__(self, dataset_name: str, custom_root: Optional[str] = None):
        """
        Initialize RS dataset configuration
        
        Args:
            dataset_name: Name of the dataset (e.g., 'dfc2023', 'dfc2023mini')
            custom_root: Custom root directory (overrides default)
        """
        self.dataset_name = dataset_name.lower()
        
        if self.dataset_name in self.DATASET_CONFIGS:
            self.config = self.DATASET_CONFIGS[self.dataset_name].copy()
        else:
            # Create default config for unknown datasets
            self.config = self._create_default_config(dataset_name)
        
        # Override root directory if provided
        if custom_root:
            self.config['default_root'] = custom_root
            
    def _create_default_config(self, dataset_name: str) -> Dict:
        """Create default configuration for unknown datasets"""
        return {
            'name': dataset_name.upper(),
            'description': f'Custom RS dataset: {dataset_name}',
            'default_root': f'/home/asfand/Ahmad/datasets/{dataset_name}',
            'height_scale_factor': 1.0,
            'max_height': 150.0,
            'image_size': (512, 512),
            'file_extensions': {
                'rgb': ['.png', '.jpg', '.jpeg', '.tif', '.tiff'],
                'dsm': ['.tif', '.tiff', '.png'],
                'sem': ['.png', '.tif', '.tiff']
            }
        }
    
    def get_config(self) -> Dict:
        """Get the complete configuration dictionary"""
        return self.config
    
    def get_data_root(self) -> str:
        """Get the data root directory"""
        return self.config['default_root']
    
    def get_height_params(self) -> Tuple[float, float]:
        """Get height scale factor and max height"""
        return self.config['height_scale_factor'], self.config['max_height']
    
    def get_image_size(self) -> Tuple[int, int]:
        """Get default image size"""
        return self.config['image_size']
    
    def validate_dataset_structure(self, root_dir: Optional[str] = None) -> bool:
        """
        Validate that the dataset follows the expected structure
        
        Args:
            root_dir: Root directory to validate (uses default if None)
            
        Returns:
            True if structure is valid, False otherwise
        """
        if root_dir is None:
            root_dir = self.get_data_root()
            
        if not os.path.exists(root_dir):
            print(f"ERROR: Dataset root directory does not exist: {root_dir}")
            return False
        
        splits = ['train', 'valid', 'test']
        required_folders = ['rgb', 'dsm']
        optional_folders = ['sem']
        
        print(f"Validating {self.config['name']} dataset structure at: {root_dir}")
        print(f"Description: {self.config['description']}")
        print("-" * 60)
        
        all_valid = True
        for split in splits:
            split_path = os.path.join(root_dir, split)
            if not os.path.exists(split_path):
                print(f"WARNING: Missing {split} split directory")
                continue
                
            print(f"\n{split.upper()} split:")
            
            # Check required folders
            for folder in required_folders:
                folder_path = os.path.join(split_path, folder)
                if not os.path.exists(folder_path):
                    print(f"  ❌ MISSING: {folder}/ (required)")
                    all_valid = False
                else:
                    extensions = self.config['file_extensions'][folder]
                    file_count = len([f for f in os.listdir(folder_path) 
                                    if any(f.lower().endswith(ext) for ext in extensions)])
                    print(f"  ✅ {folder}/: {file_count} files")
            
            # Check optional folders
            for folder in optional_folders:
                folder_path = os.path.join(split_path, folder)
                if os.path.exists(folder_path):
                    extensions = self.config['file_extensions'][folder]
                    file_count = len([f for f in os.listdir(folder_path) 
                                    if any(f.lower().endswith(ext) for ext in extensions)])
                    print(f"  📁 {folder}/ (optional): {file_count} files")
                else:
                    print(f"  📁 {folder}/ (optional): not present")
        
        print("\n" + "=" * 60)
        if all_valid:
            print("✅ Dataset structure validation PASSED")
        else:
            print("❌ Dataset structure validation FAILED")
            
        return all_valid
    
    def get_training_config_dict(self) -> Dict:
        """Get configuration dictionary for training"""
        return {
            'dataset_name': self.dataset_name,
            'rs_root': self.get_data_root(),
            'height_scale_factor': self.config['height_scale_factor'],
            'max_height': self.config['max_height'],
            'image_size': self.config['image_size'],
            'dataset_info': {
                'name': self.config['name'],
                'description': self.config['description']
            }
        }
    
    @classmethod
    def list_available_datasets(cls) -> None:
        """Print list of available predefined datasets"""
        print("Available predefined RS datasets:")
        print("=" * 50)
        for key, config in cls.DATASET_CONFIGS.items():
            print(f"{key:15} - {config['name']}")
            print(f"{'':15}   {config['description']}")
            print(f"{'':15}   Default root: {config['default_root']}")
            print(f"{'':15}   Max height: {config['max_height']}m")
            print()


def get_rs_config(dataset_name: str, custom_root: Optional[str] = None) -> RSDatasetConfig:
    """
    Convenience function to get RS dataset configuration
    
    Args:
        dataset_name: Name of the dataset
        custom_root: Custom root directory (optional)
        
    Returns:
        RSDatasetConfig instance
    """
    return RSDatasetConfig(dataset_name, custom_root)


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description='RS Dataset Configuration Tool')
    parser.add_argument('--list', action='store_true', 
                       help='List available predefined datasets')
    parser.add_argument('--validate', type=str, 
                       help='Validate dataset structure for given dataset name')
    parser.add_argument('--root', type=str, 
                       help='Custom root directory for validation')
    
    args = parser.parse_args()
    
    if args.list:
        RSDatasetConfig.list_available_datasets()
    elif args.validate:
        config = get_rs_config(args.validate, args.root)
        config.validate_dataset_structure()
        print("\nTraining configuration:")
        from pprint import pprint
        pprint(config.get_training_config_dict())