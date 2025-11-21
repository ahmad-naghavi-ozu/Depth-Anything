#!/usr/bin/env python3
"""
Training script for remote sensing height estimation using ZoeDepth pipeline
Minimal changes from original train_mono.py
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import sys
import os

# Set environment variables to reduce output
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'metric_depth'))

import torch
import torch.multiprocessing as mp

from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import get_config, change_dataset
from zoedepth.trainers.builder import get_trainer


def main_worker(gpu, ngpus_per_node, config):
    """Main training worker"""
    try:
        from zoedepth.models.builder import build_model
        from zoedepth.data.data_mono import DepthDataLoader
        from zoedepth.utils.misc import count_parameters, parallelize
        
        # Fix random seed
        seed = config.seed if 'seed' in config and config.seed else 43
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        config.gpu = gpu
        
        # Build model
        model = build_model(config)
        model = parallelize(config, model)
        
        total_params = f"{round(count_parameters(model)/1e6, 2)}M"
        config.total_params = total_params
        print(f"Total parameters: {total_params}")
        
        # Get data loaders
        train_loader = DepthDataLoader(config, "train").data
        test_loader = DepthDataLoader(config, "online_eval").data
        
        # Get trainer instance
        trainer_cls = get_trainer(config)
        trainer = trainer_cls(config, model, train_loader, test_loader, device=config.gpu)
        
        # Train
        trainer.train()
        
    except Exception as e:
        print(f"Error in worker: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="zoedepth")
    parser.add_argument("-d", "--dataset", type=str, default='dfc2023s')
    parser.add_argument("--trainer", type=str, default=None)
    
    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)
    
    # Model and trainer
    overwrite_kwargs["model"] = args.model
    if args.trainer is not None:
        overwrite_kwargs["trainer"] = args.trainer
    
    # Get config
    config = get_config(args.model, "train", args.dataset, **overwrite_kwargs)
    
    # Setup
    config.batch_size = config.bs
    config.mode = 'train'
    
    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)
    
    # Check for distributed training
    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')
        config.world_size = len(nodes)
        config.rank = int(os.environ['SLURM_PROCID'])
        print(f"SLURM detected: world_size={config.world_size}, rank={config.rank}")
    except KeyError:
        config.world_size = 1
        config.rank = 0
        nodes = ["127.0.0.1"]
    
    if config.distributed:
        import numpy as np
        port = np.random.randint(15000, 15025)
        config.dist_url = f'tcp://{nodes[0]}:{port}'
        print(f"Distributed URL: {config.dist_url}")
        config.dist_backend = 'nccl'
        config.gpu = None
    
    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node
    
    print(f"Training configuration:")
    print(f"  Model: {config.model}")
    print(f"  Dataset: {config.dataset}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Workers: {config.workers}")
    print(f"  GPUs per node: {ngpus_per_node}")
    print(f"  Min depth: {config.min_depth}")
    print(f"  Max depth: {config.max_depth}")
    
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = 0
        main_worker(config.gpu, ngpus_per_node, config)
