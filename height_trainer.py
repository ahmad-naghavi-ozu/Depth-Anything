import torch
import torch.cuda.amp as amp
import torch.nn as nn

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
    def __init__(self, model, loss_type='l1', device='cuda', use_multi_gpu=False, gpu_ids=None):
        self.device = device
        self.multi_gpu = use_multi_gpu
        self.model = model
        
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
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.scaler = amp.GradScaler()
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_on_batch(self, rgb, height_gt):
        self.optimizer.zero_grad()

        rgb = rgb.to(self.device)
        height_gt = height_gt.to(self.device)
        
        # Check for invalid values
        if torch.isnan(rgb).any() or torch.isinf(rgb).any():
            print("Warning: NaN/Inf in RGB input, skipping batch")
            return 0.0
        if torch.isnan(height_gt).any() or torch.isinf(height_gt).any():
            print("Warning: NaN/Inf in height ground truth, skipping batch")
            return 0.0

        with amp.autocast():
            pred_height = self.model(rgb)  # Model outputs [B, H, W]
            
            # Check prediction validity
            if torch.isnan(pred_height).any() or torch.isinf(pred_height).any():
                print("Warning: NaN/Inf in predictions, skipping batch")
                return 0.0

            loss = self.criterion(pred_height.unsqueeze(1), height_gt.unsqueeze(1))  # Both [B, 1, H, W]
            
            # Check loss validity
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN/Inf loss, skipping batch")
                return 0.0

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

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

        for epoch in tqdm(range(epochs), desc="Epochs"):
            # Clear GPU cache at start of each epoch to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            total_loss = 0
            for rgb, height_gt in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                loss = self.train_on_batch(rgb, height_gt)
                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
            logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
            
            # Validation and early stopping
            if val_dataset is not None:
                val_loss = self.validate(val_dataset, batch_size)
                print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
                logging.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
                
                # Early stopping logic
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if checkpoint_path:
                        best_checkpoint = checkpoint_path.replace('.pth', '_best.pth')
                        self.save_model(best_checkpoint)
                        print(f"Best model saved to {best_checkpoint}")
                        logging.info(f"Best model saved with val_loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"No improvement. Patience: {self.patience_counter}/{patience}")
                    logging.info(f"No improvement. Patience: {self.patience_counter}/{patience}")
                    
                    if self.patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
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
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--use_validation', action='store_true', help='Use validation set for early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--model_size', type=str, default='vits', choices=['vits', 'vitb', 'vitl'], help='Model size (ViT variant)')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--results_dir', type=str, default='results/height_adapted_01', help='Directory to save results')
    parser.add_argument('--freeze_encoder', action='store_true', default=True, help='Freeze DINOv2 encoder during training (default: True). Use --no-freeze_encoder to train encoder.')
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

    # Setup logging
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'training_{timestamp}.log'
    logging.basicConfig(filename=os.path.join(logs_dir, log_filename), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    dataset_path = os.path.join(args.dataset_base_path, args.dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained DepthAnything model from local checkpoint or resume from fine-tuned checkpoint
    if args.resume_from:
        print(f"Resuming training from: {args.resume_from}")
        model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{args.model_size}14')
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint)
        logging.info(f"Resumed training from checkpoint: {args.resume_from}")
    else:
        local_checkpoint = f'checkpoints/depth_anything_{args.model_size}14.pth'
        if os.path.exists(local_checkpoint):
            print(f"Loading local checkpoint: {local_checkpoint}")
            model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{args.model_size}14')
            checkpoint = torch.load(local_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint)
            logging.info(f"Loaded local checkpoint: {local_checkpoint}")
        else:
            print(f"Local checkpoint not found, downloading from HuggingFace")
            model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{args.model_size}14')
            logging.info(f"Downloaded model from HuggingFace")

    # Create datasets
    dataset = RemoteSensingHeightDataset(dataset_path, split='train')
    val_dataset = None
    if args.use_validation:
        val_dataset = RemoteSensingHeightDataset(dataset_path, split='valid')
        print(f"Using validation set with {len(val_dataset)} samples")
        logging.info(f"Using validation set with {len(val_dataset)} samples")

    # Create trainer with GPU IDs
    gpu_ids = None
    if args.multi_gpu and args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    trainer = HeightTrainer(model, loss_type=args.loss_type, device=device, use_multi_gpu=args.multi_gpu, gpu_ids=gpu_ids)
    
    # Freeze/unfreeze encoder based on --freeze_encoder flag
    if args.freeze_encoder:
        # Freeze encoder, train only decoder (default for transfer learning)
        for name, param in model.named_parameters():
            if 'pretrained' in name:  # DINOv2 encoder parameters
                param.requires_grad = False
        print("Encoder (DINOv2) frozen - training only decoder (DPT Head)")
        logging.info("Training mode: Decoder-only fine-tuning (encoder frozen)")
    else:
        # Train both encoder and decoder
        print("Training both encoder (DINOv2) and decoder (DPT Head)")
        logging.info("Training mode: Full fine-tuning (encoder + decoder)")
    
    trainer.optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad], 
        lr=args.lr
    )

    logging.info(f"Starting training with loss: {args.loss_type}, epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}")
    print(f"Starting training with model_size: {args.model_size}, loss: {args.loss_type}")

    # Prepare checkpoint path for early stopping
    save_path = os.path.join(checkpoints_dir, f'depth_anything_height_finetuned_{args.loss_type}_{args.model_size}.pth')

    # Train
    trainer.train(dataset, epochs=args.epochs, batch_size=args.batch_size, 
                  val_dataset=val_dataset, patience=args.patience, checkpoint_path=save_path)

    # Save final model with _last suffix
    final_save_path = save_path.replace('.pth', '_last.pth')
    trainer.save_model(final_save_path)
    logging.info(f"Final model saved to {final_save_path}")
    print(f"Final model saved to {final_save_path}")