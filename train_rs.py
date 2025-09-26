#!/usr/bin/env python3

"""
Training script for Remote Sensing Height Estimation using Depth Anything

This script supports multiple RS datasets (DFC2023, DFC2019, etc.) with a unified interface.
Just change the --rs-dataset parameter to switch between different datasets.

Supported datasets:
- dfc2023: IEEE Data Fusion Contest 2023
- dfc2023mini: Mini version for debugging  
- dfc2019: IEEE Data Fusion Contest 2019
- custom: User-defined datasets

Usage:
    python train_rs.py -m zoedepth --rs-dataset dfc2023mini
    python train_rs.py -m zoedepth --rs-dataset dfc2023 --rs-root /custom/path
    python train_rs.py -m zoedepth --rs-dataset custom --rs-root /path/to/dataset
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'metric_depth'))

from zoedepth.utils.misc import count_parameters, parallelize
from zoedepth.utils.config import get_config
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.trainers.builder import get_trainer
from zoedepth.models.builder import build_model
from zoedepth.data.data_mono import DepthDataLoader
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch
import numpy as np
from pprint import pprint
import argparse
import os

# Import our generic RS configuration
from rs_config import get_rs_config, RSDatasetConfig

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"


def fix_random_seed(seed: int):
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_ckpt(config, model, checkpoint_dir="./checkpoints", ckpt_type="best"):
    import glob
    import os
    from zoedepth.models.model_io import load_wts

    if hasattr(config, "checkpoint"):
        checkpoint = config.checkpoint
    elif hasattr(config, "ckpt_pattern"):
        pattern = config.ckpt_pattern
        matches = glob.glob(os.path.join(
            checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
        if not (len(matches) > 0):
            raise ValueError(f"No matches found for the pattern {pattern}")
        checkpoint = matches[0]
    else:
        return model
    
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model


def main_worker(gpu, ngpus_per_node, config):
    try:
        seed = config.seed if 'seed' in config and config.seed else 43
        fix_random_seed(seed)

        config.gpu = gpu

        model = build_model(config)
        
        # Load Depth Anything pre-trained weights
        model = load_ckpt(config, model)
        model = parallelize(config, model)

        total_params = f"{round(count_parameters(model)/1e6,2)}M"
        config.total_params = total_params
        print(f"Total parameters : {total_params}")

        # Use RS dataset loaders
        train_loader = DepthDataLoader(config, "train").data
        test_loader = DepthDataLoader(config, "online_eval").data

        trainer = get_trainer(config)(
            config, model, train_loader, test_loader, device=config.gpu)

        print("\nStarting Remote Sensing Height Estimation Training...")
        print(f"Total model parameters: {total_params}")
        if hasattr(config, 'dataset_info'):
            print(f"Training on: {config.dataset_info['name']}")
        
        trainer.train()
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        import wandb
        wandb.finish()


def validate_rs_dataset(rs_config):
    """Validate RS dataset using the generic configuration"""
    return rs_config.validate_dataset_structure()


if __name__ == '__main__':
    mp.set_start_method('forkserver')

    parser = argparse.ArgumentParser(description='Train Depth Anything for Remote Sensing Height Estimation')
    parser.add_argument("-m", "--model", type=str, default="zoedepth", 
                       help="Model architecture to use")
    parser.add_argument("-d", "--dataset", type=str, default='remote_sensing', 
                       choices=['remote_sensing', 'rs'],
                       help='Dataset type (always remote_sensing for RS datasets)')
    parser.add_argument("--trainer", type=str, default=None,
                       help="Trainer type (optional)")
    
    # New RS dataset configuration parameters
    parser.add_argument("--rs-dataset", type=str, default="dfc2023mini",
                       help="RS dataset name (dfc2023, dfc2023mini, dfc2019, custom)")
    parser.add_argument("--rs-root", type=str, default=None,
                       help="Root directory of RS dataset (overrides default)")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available predefined RS datasets")
    parser.add_argument("--validate-data", action="store_true",
                       help="Validate dataset structure before training")
    
    # Training parameters  
    parser.add_argument("--height-scale-factor", type=float, default=None,
                       help="Height scale factor (overrides dataset default)")
    parser.add_argument("--max-height", type=float, default=None,
                       help="Maximum height in meters (overrides dataset default)")

    args, unknown_args = parser.parse_known_args()
    
    # Handle dataset listing
    if args.list_datasets:
        RSDatasetConfig.list_available_datasets()
        exit(0)
    
    # Initialize RS dataset configuration
    print(f"Initializing RS dataset configuration for: {args.rs_dataset}")
    rs_config = get_rs_config(args.rs_dataset, args.rs_root)
    
    # Validate dataset if requested
    if args.validate_data:
        print("\n" + "="*60)
        if not validate_rs_dataset(rs_config):
            response = input("Dataset validation failed. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                exit(1)
        else:
            response = input("Dataset validation passed. Continue with training? (y/n): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                exit(0)
    
    overwrite_kwargs = parse_unknown(unknown_args)
    overwrite_kwargs["model"] = args.model
    if args.trainer is not None:
        overwrite_kwargs["trainer"] = args.trainer

    # Get RS configuration parameters
    rs_train_config = rs_config.get_training_config_dict()
    
    # Apply RS-specific overrides
    overwrite_kwargs["rs_root"] = rs_train_config["rs_root"]
    
    # Use custom parameters if provided, otherwise use dataset defaults
    if args.height_scale_factor is not None:
        overwrite_kwargs["height_scale_factor"] = args.height_scale_factor
    else:
        overwrite_kwargs["height_scale_factor"] = rs_train_config["height_scale_factor"]
        
    if args.max_height is not None:
        overwrite_kwargs["max_height"] = args.max_height  
    else:
        overwrite_kwargs["max_height"] = rs_train_config["max_height"]
        
    overwrite_kwargs["image_size"] = rs_train_config["image_size"]
    overwrite_kwargs["dataset_info"] = rs_train_config["dataset_info"]

    config = get_config(args.model, "train", args.dataset, **overwrite_kwargs)

    # Set up distributed training configuration
    if config.use_shared_dict:
        shared_dict = mp.Manager().dict()
    else:
        shared_dict = None
    config.shared_dict = shared_dict

    config.batch_size = config.bs
    config.mode = 'train'
    
    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')
        config.world_size = len(nodes)
        config.rank = int(os.environ['SLURM_PROCID'])
    except KeyError as e:
        # We are NOT using SLURM
        config.world_size = 1
        config.rank = 0
        nodes = ["127.0.0.1"]

    if config.distributed:
        print(config.rank)
        port = np.random.randint(15000, 15025)
        config.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(config.dist_url)
        config.dist_backend = 'nccl'
        config.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node
    
    print("\n" + "="*60)
    print("REMOTE SENSING TRAINING CONFIGURATION")
    print("="*60)
    print(f"Dataset: {rs_train_config['dataset_info']['name']}")
    print(f"Description: {rs_train_config['dataset_info']['description']}")
    print(f"Data root: {config.rs_root}")
    print(f"Height scale factor: {config.height_scale_factor}")
    print(f"Max height: {config.max_height}m")
    print(f"Image size: {config.image_size}")
    print(f"GPUs available: {ngpus_per_node}")
    print("="*60)
    
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = 0
        main_worker(config.gpu, ngpus_per_node, config)