#!/usr/bin/env python3

"""
Quick test script for RS dataset configurations

This script allows you to quickly test different RS dataset configurations
without running full training.

Usage:
    python test_rs_datasets.py --list
    python test_rs_datasets.py --validate dfc2023mini
    python test_rs_datasets.py --validate custom --root /path/to/dataset
"""

import argparse
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(__file__))

from rs_config import get_rs_config, RSDatasetConfig


def main():
    parser = argparse.ArgumentParser(description='Test RS Dataset Configurations')
    parser.add_argument('--list', action='store_true',
                       help='List all available predefined datasets')
    parser.add_argument('--validate', type=str,
                       help='Validate dataset structure for given dataset name')
    parser.add_argument('--root', type=str,
                       help='Custom root directory (overrides default)')
    parser.add_argument('--info', type=str,
                       help='Show configuration info for dataset')
    
    args = parser.parse_args()
    
    if args.list:
        RSDatasetConfig.list_available_datasets()
        
    elif args.validate:
        print(f"Validating dataset: {args.validate}")
        if args.root:
            print(f"Using custom root: {args.root}")
        
        rs_config = get_rs_config(args.validate, args.root)
        
        # Show configuration info first
        config = rs_config.get_config()
        print(f"\nDataset Configuration:")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Root directory: {rs_config.get_data_root()}")
        print(f"  Height scale factor: {config['height_scale_factor']}")
        print(f"  Max height: {config['max_height']}m")
        print(f"  Image size: {config['image_size']}")
        
        # Validate structure
        print("\nValidating dataset structure...")
        is_valid = rs_config.validate_dataset_structure()
        
        if is_valid:
            print("\n✅ Dataset is ready for training!")
            print("\nTo start training with this dataset:")
            print(f"python train_rs.py --rs-dataset {args.validate}", end="")
            if args.root:
                print(f" --rs-root {args.root}", end="")
            print()
        else:
            print("\n❌ Dataset structure issues detected.")
            print("Please fix the dataset structure before training.")
            
    elif args.info:
        print(f"Configuration for dataset: {args.info}")
        rs_config = get_rs_config(args.info, args.root)
        
        config = rs_config.get_config()
        train_config = rs_config.get_training_config_dict()
        
        print(f"\nGeneral Configuration:")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Root directory: {rs_config.get_data_root()}")
        
        print(f"\nHeight Parameters:")
        print(f"  Scale factor: {config['height_scale_factor']}")
        print(f"  Max height: {config['max_height']}m")
        
        print(f"\nImage Configuration:")
        print(f"  Default size: {config['image_size']}")
        
        print(f"\nFile Extensions:")
        for data_type, extensions in config['file_extensions'].items():
            print(f"  {data_type}: {', '.join(extensions)}")
            
        print(f"\nTraining Config Dict:")
        from pprint import pprint
        pprint(train_config)
        
    else:
        print("Please specify an action: --list, --validate, or --info")
        print("Use --help for more information")


if __name__ == "__main__":
    main()