import torch
import torch.cuda.amp as amp
import torch.nn as nn

from depth_anything.dpt import DepthAnything
from dataset import RemoteSensingHeightDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

class HeightTrainer:
    def __init__(self, model, loss_type='l1', device='cuda'):
        self.device = device
        self.model = model.to(device)
        
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

    def train_on_batch(self, rgb, height_gt):
        rgb = rgb.to(self.device)
        height_gt = height_gt.to(self.device)

        self.optimizer.zero_grad()

        with amp.autocast():
            pred_height = self.model(rgb)  # Model outputs [B, H, W]

            loss = self.criterion(pred_height.unsqueeze(1), height_gt.unsqueeze(1))  # Both [B, 1, H, W]

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def train(self, dataset, epochs=10, batch_size=4):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(epochs), desc="Epochs"):
            total_loss = 0
            for rgb, height_gt in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                loss = self.train_on_batch(rgb, height_gt)
                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


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
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--results_dir', type=str, default='results/height_adapted_01', help='Directory to save results')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'], help='Encoder type')

    args = parser.parse_args()

    # Create directories
    checkpoints_dir = os.path.join(args.checkpoints_dir, args.dataset_name)
    logs_dir = os.path.join(args.logs_dir, args.dataset_name)
    results_dir = os.path.join(args.results_dir, args.dataset_name)
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

    # Load pre-trained DepthAnything model
    model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{args.encoder}14')

    # Create dataset
    dataset = RemoteSensingHeightDataset(dataset_path, split='train')

    # Create trainer
    trainer = HeightTrainer(model, loss_type=args.loss_type, device=device)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=args.lr)

    logging.info(f"Starting training with loss: {args.loss_type}, epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}")
    print(f"Starting training with loss: {args.loss_type}")

    # Train
    trainer.train(dataset, epochs=args.epochs, batch_size=args.batch_size)

    # Save fine-tuned model
    save_path = os.path.join(checkpoints_dir, f'depth_anything_height_finetuned_{args.loss_type}_{args.encoder}.pth')
    trainer.save_model(save_path)
    logging.info(f"Model saved to {save_path}")
    print(f"Model saved to {save_path}")