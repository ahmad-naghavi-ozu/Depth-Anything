# Remote Sensing Height Estimation with ZoeDepth

This branch (`feature/remote-sensing-height-estimation-adaptation-2`) adapts the DepthAnything + ZoeDepth pipeline for remote sensing height estimation with **minimal changes** from the original depth estimation approach.

## Key Approach

- Uses **original ZoeDepth architecture** with bin-based metric depth prediction
- Uses **original loss functions**: SILog + GradL1 (no additional metric losses)
- **Bin centers** provide metric scale anchoring (0-150m range)
- Dataset-specific config with RS height range

## Dataset Configuration

**DFC2023S Statistics:**
- Training samples: 1419
- Height range: 0.0 - 147.45 meters
- Input resolution: 512×512 (RGB + DSM)
- Format: GeoTIFF files

**ZoeDepth Configuration:**
```python
min_depth = 0.0 meters
max_depth = 150.0 meters
n_bins = 64 (default)
```

## Setup

### 1. Environment
```bash
conda activate depth_anything_rs
```

### 2. Dataset Structure
```
/home/asfand/Ahmad/datasets/DFC2023S/
├── train/
│   ├── rgb/      # RGB TIF files
│   └── dsm/      # DSM (height) TIF files
├── val/
│   ├── rgb/
│   └── dsm/
└── test/
    ├── rgb/
    └── dsm/
```

## Training

### Using Shell Script (Recommended)

**Multi-GPU Training:**
```bash
# Default configuration (4 GPUs, batch_size=8, epochs=50)
./rs_height_train.sh

# Single GPU with custom parameters
./rs_height_train.sh --dataset dfc2023mini --distributed False --gpu-ids 0 --batch-size 2 --epochs 5

# Resume from checkpoint
./rs_height_train.sh --resume ./checkpoints/rs_height_zoedepth/DFC2023S/model_latest.pt

# Different dataset
./rs_height_train.sh --dataset dfc2019_crp512_bin --epochs 30 --patience 5
```

**Show help:**
```bash
./rs_height_train.sh --help
```

### Using Python Script Directly

**Basic Training Command (Single GPU):**
```bash
python rs_height_train.py \
    --model zoedepth \
    --dataset dfc2023s \
    --bs 2 \
    --epochs 5 \
    --distributed False
```

**Multi-GPU Training:**
```bash
python rs_height_train.py \
    --model zoedepth \
    --dataset dfc2023s \
    --midas_model_type dinov2_base \
    --bs 8 \
    --epochs 5 \
    --distributed True
```

**Training Configuration:**
- **Encoder Finetuning**: DINOv2 backbone is finetuned with 10x lower LR (better adaptation to RS imagery)
- **Batch Size**: 8 total (2 per GPU × 4 GPUs) - reduced due to encoder gradient computation
- **Learning Rate**: 0.000161 for metric head, 0.0000161 for encoder (10x factor)
- **Epochs**: 5 (sufficient with pre-trained DepthAnything weights)
- **Model**: ViT-B (97.78M params) for 11GB GPUs

**Output Locations:**
- Checkpoints: `./checkpoints/rs_height_zoedepth/`
- Terminal Logs: `./logs/rs_height_zoedepth/`
- WandB Logs: `./wandb/` (local) + online project `MonoDepth3-dfc2023s`

**Note**: Encoder finetuning (`train_midas=true`) is enabled by default for DFC2023S dataset via `config_zoedepth_dfc2023s.json`. This allows the DINOv2 backbone to adapt to remote sensing imagery characteristics.

## Inference

### Using Shell Script (Recommended)

```bash
# Basic inference on test set with predictions saved
./rs_height_infer.sh \
    --checkpoint ./checkpoints/rs_height_zoedepth/DFC2023S/model_best.pt \
    --dataset-root /home/asfand/Ahmad/datasets/DFC2023S

# Inference on validation set without saving predictions
./rs_height_infer.sh \
    --checkpoint ./checkpoints/rs_height_zoedepth/DFC2023S/model_best.pt \
    --dataset-root /home/asfand/Ahmad/datasets/DFC2023S \
    --split val \
    --no-save-predictions

# Different dataset
./rs_height_infer.sh \
    --checkpoint ./checkpoints/rs_height_zoedepth/DFC2023S/model_best.pt \
    --dataset-root /home/asfand/Ahmad/datasets/Huawei_Contest \
    --batch-size 4
```

**Show help:**
```bash
./rs_height_infer.sh --help
```

### Using Python Script Directly

```bash
python rs_height_infer.py \
    --checkpoint ./checkpoints/rs_height_zoedepth/best_model.pth \
    --dataset-root /home/asfand/Ahmad/datasets/DFC2023S \
    --split test \
    --output-dir ./results/rs_height_zoedepth \
    --save-predictions
```

**Output Structure:**
```
./checkpoints/rs_height_zoedepth/    # Model checkpoints (best_model.pth, latest.pth)
./logs/rs_height_zoedepth/           # Terminal output logs (training_*.log)
./results/rs_height_zoedepth/        # Inference results (predicted DSMs)
./wandb/                              # WandB local logs and metrics
```

## Utilities

### Compute Dataset Height Range

```bash
# Using dataset name (default path: /home/asfand/Ahmad/datasets/)
./rs_height_compute_range.py --dataset DFC2023S
./rs_height_compute_range.py --dataset DFC2019_crp512_bin
./rs_height_compute_range.py --dataset Huawei_Contest

# Using custom dataset path
./rs_height_compute_range.py /path/to/custom/dataset

# Analyze validation split instead of train
./rs_height_compute_range.py --dataset DFC2023S --split val
```

## Implementation Details

### Files Added/Modified

**New Files:**
- `metric_depth/zoedepth/data/dfc2023s.py` - DFC2023S dataset loader
- `rs_height_train.py` - Training script using ZoeDepth pipeline
- `rs_height_train.sh` - Shell wrapper for training with convenient options
- `rs_height_infer.py` - Inference and evaluation script
- `rs_height_infer.sh` - Shell wrapper for inference with convenient options
- `rs_height_compute_range.py` - Utility to compute dataset height range
- `metrics_utils.py` - DSM evaluation metrics (copied from other branch)

**Modified Files:**
- `metric_depth/zoedepth/utils/config.py` - Added DFC2023S dataset config
- `metric_depth/zoedepth/data/data_mono.py` - Registered DFC2023S dataloader

### Loss Functions (Unchanged from Original)

**Primary Loss: SILogLoss**
```python
g = log(pred) - log(gt)
Dg = var(g) + 0.15 * mean(g)²
loss = sqrt(Dg)
```
- Scale-invariant
- Learns relative structure
- Weight: w_si = 1.0

**Regularization: GradL1Loss**
```python
grad_x = pred[:, :, 1:] - pred[:, :-1]
grad_y = pred[:, 1:, :] - pred[:-1, :, :]
loss = |grad_x_pred - grad_x_gt| + |grad_y_pred - grad_y_gt|
```
- Edge preservation
- Weight: w_grad = 0.0 (disabled by default, can enable with 0.1-0.5)

### How Metric Scale is Learned

Despite using scale-invariant SILog loss, the model learns metric scale through:

1. **Bin Centers**: Pre-defined in meters (0-150m range, 64 bins)
2. **Bin Classification**: Model predicts which bin each pixel belongs to
3. **Attractor MLPs**: Learn local offsets from bin centers
4. **Final Output**: Weighted sum of bin probabilities × bin centers = metric depth

## Expected Results

Based on similar depth estimation tasks:

| Metric | Target Range |
|--------|-------------|
| RMSE | 5-15 meters |
| MAE | 3-10 meters |
| R² | 0.7-0.9 |
| Delta1 | 0.6-0.8 |

## Experiments to Try

1. **Baseline**: Default config (n_bins=64, w_si=1.0, w_grad=0.0)
2. **With Gradients**: Enable gradient loss (w_grad=0.1)
3. **Higher Resolution Bins**: n_bins=128 or 256
4. **Learning Rate Tuning**: Try 5e-5, 1e-4, 2e-4
5. **Longer Training**: 10-20 epochs instead of 5

## Comparison with Previous Approach

| Aspect | Previous (height_train.py) | This (rs_height_train.py) |
|--------|---------------------------|--------------------------|
| **Architecture** | Simple DPT (direct regression) | ZoeDepth (bins + attractors) |
| **Loss** | L1 only | SILog + GradL1 |
| **Metric Scale** | Direct from L1 loss | From bin centers + classification |
| **Code Base** | Custom trainer | Original ZoeDepth pipeline |
| **Modifications** | Many changes | Minimal changes |

## Troubleshooting

### Dataset Not Found
```bash
# Check dataset path in config
python -c "from zoedepth.utils.config import DATASETS_CONFIG; print(DATASETS_CONFIG['dfc2023s'])"
```

### Out of Memory
```bash
# Reduce batch size using shell script
./rs_height_train.sh --batch-size 4

# Or using Python directly
python rs_height_train.py --batch_size 4
```

### Check Model Output
```python
import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

config = get_config("zoedepth", "eval", "dfc2023s")
model = build_model(config)
print(model)
```

## References

- Original DepthAnything: https://github.com/LiheYoung/Depth-Anything
- ZoeDepth Paper: "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth"
- DFC2023 Challenge: https://ieee-dataport.org/competitions/2023-ieee-grss-data-fusion-contest

## Notes

- This implementation stays very close to the original depth estimation pipeline
- The key insight: bin-based architecture provides metric scale even with scale-invariant loss
- For comparison, you may want to train both this version and the simple DPT + L1 loss version
