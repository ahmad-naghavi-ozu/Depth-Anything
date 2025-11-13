import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from depth_anything.dpt import DepthAnything
from dataset import RemoteSensingHeightDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import warnings

# Suppress warnings from deep learning libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class HeightTrainer:
    def __init__(self, model, loss_type='l1', device='cuda', use_multi_gpu=False, gpu_ids=None, lr=3e-5, 
                 freeze_encoder=False, weight_decay=0.05, warmup_iters=1500, total_iters=None):
        self.device = device
        self.multi_gpu = use_multi_gpu
        self.model = model
        self.freeze_encoder = freeze_encoder
        self.lr = lr
        self.warmup_iters = warmup_iters
        self.current_iter = 0
        
        # Parse GPU IDs
        if gpu_ids is not None:
            if isinstance(gpu_ids, str):
                self.gpu_ids = [int(x) for x in gpu_ids.split(',')]
            else:
                self.gpu_ids = gpu_ids
        else:
            self.gpu_ids = list(range(torch.cuda.device_count()))
        
        # Multi-GPU training
        if self.multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
            # Clear GPU cache before starting
            torch.cuda.empty_cache()
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        
        self.model = self.model.to(self.device)
        
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Set up optimizer with differential learning rates
        # Backbone (DINOv2): 0.1x learning rate, Decoder (DPT): 1.0x learning rate
        model_to_optimize = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        if freeze_encoder:
            # Only train decoder parameters
            params = [{'params': model_to_optimize.scratch.parameters(), 'lr': lr}]
        else:
            # Differential learning rates: backbone gets 0.1x, decoder gets 1.0x
            backbone_params = []
            decoder_params = []
            
            for name, param in model_to_optimize.named_parameters():
                if 'pretrained' in name or 'blocks' in name:  # DINOv2 backbone
                    backbone_params.append(param)
                else:  # DPT decoder head
                    decoder_params.append(param)
            
            params = [
                {'params': backbone_params, 'lr': lr * 0.1},  # 10% for backbone
                {'params': decoder_params, 'lr': lr}  # 100% for decoder
            ]
        
        # Use AdamW optimizer (better for transformers)
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, 
                                           eps=1e-8, betas=(0.9, 0.999))
        
        # Set up PolyLR scheduler with warmup
        self.total_iters = total_iters  # Will be set in train()
        self.scheduler = None  # Will be initialized in train() when total_iters is known
        
        self.scaler = amp.GradScaler()
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_on_batch(self, rgb, height_gt):
        self.optimizer.zero_grad()

        rgb = rgb.to(self.device)
        height_gt = height_gt.to(self.device)
        
        # Check for invalid values
        if torch.isnan(rgb).any() or torch.isinf(rgb).any():
            logging.warning("NaN/Inf in RGB input, skipping batch")
            return 0.0
        if torch.isnan(height_gt).any() or torch.isinf(height_gt).any():
            logging.warning("NaN/Inf in height ground truth, skipping batch")
            return 0.0

        with amp.autocast():
            pred_height = self.model(rgb)  # Model outputs [B, H, W]
            
            # Check prediction validity
            if torch.isnan(pred_height).any() or torch.isinf(pred_height).any():
                logging.warning("NaN/Inf in predictions, skipping batch")
                return 0.0

            loss = self.criterion(pred_height.unsqueeze(1), height_gt.unsqueeze(1))  # Both [B, 1, H, W]
            
            # Check loss validity
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("NaN/Inf loss, skipping batch")
                return 0.0

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update learning rate with scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.current_iter += 1

        return loss.item()

    def validate(self, val_dataset, batch_size=4):
        """Validate the model on validation dataset"""
        self.model.eval()
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        total_val_loss = 0
        
        with torch.no_grad():
            for rgb, height_gt in val_dataloader:
                rgb = rgb.to(self.device)
                height_gt = height_gt.to(self.device)
                
                pred_height = self.model(rgb)
                loss = self.criterion(pred_height.unsqueeze(1), height_gt.unsqueeze(1))
                total_val_loss += loss.item()
        
        self.model.train()
        return total_val_loss / len(val_dataloader)
    
    def train(self, dataset, epochs=10, batch_size=4, val_dataset=None, patience=5, checkpoint_path=None):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize scheduler with total iterations
        if self.scheduler is None:
            self.total_iters = epochs * len(dataloader)
            
            def poly_lr_with_warmup(current_iter):
                """PolyLR scheduler with linear warmup"""
                if current_iter < self.warmup_iters:
                    # Linear warmup from 1e-6 to 1.0
                    return 1e-6 + (1.0 - 1e-6) * (current_iter / self.warmup_iters)
                else:
                    # PolyLR: (1 - (iter - warmup) / (total - warmup)) ^ power
                    # power=1.0 for linear decay
                    progress = (current_iter - self.warmup_iters) / (self.total_iters - self.warmup_iters)
                    return max(0.0, (1.0 - progress))
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=poly_lr_with_warmup)

        for epoch in tqdm(range(epochs), desc="Epochs"):
            # Clear GPU cache at start of each epoch to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            total_loss = 0
            for rgb, height_gt in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                loss = self.train_on_batch(rgb, height_gt)
                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
            
            # Validation and early stopping
            if val_dataset is not None:
                val_loss = self.validate(val_dataset, batch_size)
                logging.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
                
                # Early stopping logic with minimum improvement threshold
                min_delta = 0.001  # Minimum improvement required (0.001 absolute improvement)
                if val_loss < (self.best_val_loss - min_delta):
                    improvement = self.best_val_loss - val_loss
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if checkpoint_path:
                        best_checkpoint = checkpoint_path.replace('.pth', '_best.pth')
                        self.save_model(best_checkpoint)
                        logging.info(f"Best model saved to {best_checkpoint} with val_loss: {val_loss:.4f} (improvement: {improvement:.4f})")
                else:
                    self.patience_counter += 1
                    logging.info(f"No significant improvement (min_delta={min_delta}). Patience: {self.patience_counter}/{patience}")
                    
                    if self.patience_counter >= patience:
                        logging.info(f"Early stopping triggered after {epoch+1} epochs")
                        break

    def save_model(self, path):
        # Handle DataParallel wrapper
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save(model_to_save.state_dict(), path)


if __name__ == '__main__':
    import argparse
    import os
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='DFC2023S', help='Name of the dataset')
    parser.add_argument('--dataset_base_path', type=str, default='/home/asfand/Ahmad/datasets/', help='Base path to datasets')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2', 'smooth_l1'], help='Loss function to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate (default: 5e-5, optimized for height estimation)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW (default: 0.05)')
    parser.add_argument('--warmup_iters', type=int, default=1500, help='Warmup iterations for learning rate scheduler (default: 1500)')
    parser.add_argument('--use_validation', action='store_true', help='Use validation set for early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs)')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--model_size', type=str, default='vits', choices=['vits', 'vitb', 'vitl'], help='Model size (ViT variant)')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--results_dir', type=str, default='results/height_adapted_01', help='Directory to save results')
    parser.add_argument('--freeze_encoder', type=lambda x: str(x).lower() == 'true', default=False, help='Freeze DINOv2 encoder during training (default: False, uses differential LR: backbone=0.1x, decoder=1.0x)')
    parser.add_argument('--use_final_relu', action='store_true', default=False, help='Use final ReLU activation (clamps outputs to [0, inf); default: False for unbounded regression)')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs for training (DataParallel)')
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma-separated GPU IDs to use (e.g., "0,1,2,3" or "2,3")')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps to reduce memory usage')

    args = parser.parse_args()
    
    # Set visible GPUs if specified
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"Using GPUs: {args.gpu_ids}")

    # Create directories organized by dataset and model size
    checkpoints_dir = os.path.join(args.checkpoints_dir, args.dataset_name, args.model_size)
    logs_dir = os.path.join(args.logs_dir, args.dataset_name, args.model_size)
    results_dir = os.path.join(args.results_dir, args.dataset_name, args.model_size)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Setup logging to both file and console
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'training_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(logs_dir, log_filename))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    dataset_path = os.path.join(args.dataset_base_path, args.dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Import DPT_DINOv2 for direct model creation
    from depth_anything.dpt import DPT_DINOv2
    
    # Model architecture parameters based on encoder size
    if args.model_size == 'vits':
        features, out_channels = 64, [48, 96, 192, 384]
    elif args.model_size == 'vitb':
        features, out_channels = 128, [96, 192, 384, 768]
    else:  # vitl
        features, out_channels = 256, [256, 512, 1024, 1024]
    
    # Load pre-trained DepthAnything model from local checkpoint or resume from fine-tuned checkpoint
    if args.resume_from:
        logging.info(f"Resuming training from: {args.resume_from}")
        model = DPT_DINOv2(encoder=args.model_size, features=features, out_channels=out_channels, use_final_relu=args.use_final_relu)
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint)
    else:
        local_checkpoint = f'checkpoints/depth_anything_{args.model_size}14.pth'
        if os.path.exists(local_checkpoint):
            logging.info(f"Loading local checkpoint: {local_checkpoint}")
            model = DPT_DINOv2(encoder=args.model_size, features=features, out_channels=out_channels, use_final_relu=args.use_final_relu)
            checkpoint = torch.load(local_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint)
        else:
            logging.info("Local checkpoint not found, downloading from HuggingFace")
            model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{args.model_size}14')

    # Create datasets
    dataset = RemoteSensingHeightDataset(dataset_path, split='train')
    val_dataset = None
    if args.use_validation:
        val_dataset = RemoteSensingHeightDataset(dataset_path, split='valid')
        logging.info(f"Using validation set with {len(val_dataset)} samples")

    # Create trainer with GPU IDs and optimizer parameters
    gpu_ids = None
    if args.multi_gpu and args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    
    # HeightTrainer now handles optimizer setup with differential learning rates
    trainer = HeightTrainer(
        model, 
        loss_type=args.loss_type, 
        device=device, 
        use_multi_gpu=args.multi_gpu, 
        gpu_ids=gpu_ids,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
        weight_decay=args.weight_decay,
        warmup_iters=args.warmup_iters
    )
    
    # Log encoder training status
    if args.freeze_encoder:
        logging.info("Encoder (DINOv2) frozen - training only decoder (DPT Head)")
    else:
        logging.info("Training both encoder (DINOv2) and decoder (DPT Head)")
        logging.info("Using differential learning rates: Backbone=0.1x, Decoder=1.0x")
    
    # Log final ReLU status
    logging.info(f"Final ReLU activation: {'enabled' if args.use_final_relu else 'disabled (unbounded regression)'}")

    # Log complete training configuration
    logging.info(f"Training Configuration:")
    logging.info(f"  Dataset: {args.dataset_name}")
    logging.info(f"  Model size: {args.model_size}")
    logging.info(f"  Loss type: {args.loss_type}")
    logging.info(f"  Epochs: {args.epochs}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Learning rate: {args.lr}")
    logging.info(f"  Patience: {args.patience}")
    logging.info(f"  Multi-GPU: {args.multi_gpu}")
    if args.multi_gpu and args.gpu_ids:
        logging.info(f"  GPU IDs: {args.gpu_ids}")
    logging.info(f"  Use validation: {args.use_validation}")

    # Prepare checkpoint path for early stopping
    save_path = os.path.join(checkpoints_dir, f'depth_anything_height_finetuned_{args.loss_type}_{args.model_size}.pth')

    # Train
    trainer.train(dataset, epochs=args.epochs, batch_size=args.batch_size, 
                  val_dataset=val_dataset, patience=args.patience, checkpoint_path=save_path)

    # Save final model with _last suffix
    final_save_path = save_path.replace('.pth', '_last.pth')
    trainer.save_model(final_save_path)
    logging.info(f"Final model saved to {final_save_path}")