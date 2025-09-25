# Remote Sensing Height Estimation with Depth Anything

This directory contains adaptations of the Depth Anything model for **Remote Sensing (RS) Height Estimation**. The model can process RGB satellite/aerial imagery to produce Digital Surface Model (DSM) style height maps for building height estimation and related tasks.

## Overview

The original Depth Anything model is designed for ground-level depth estimation. This adaptation reframes the problem for remote sensing applications:

- **Input**: Single RGB remote sensing images (satellite/aerial imagery)  
- **Output**: Single-channel height maps (similar to DSMs)
- **Application**: Building height estimation, urban analysis, 3D city modeling

## Dataset Format

Your RS dataset should follow this structure:

```
[DATASET_NAME]/
├── train/
│   ├── dsm/     # Digital Surface Model (height maps)
│   ├── rgb/     # RGB imagery
│   └── sem/     # Semantic labels (optional, currently ignored)
├── valid/
│   ├── dsm/
│   ├── rgb/
│   └── sem/
└── test/
    ├── dsm/
    ├── rgb/
    └── sem/
```

### Supported File Formats
- **RGB images**: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`
- **Height maps (DSM)**: `.png`, `.tif`, `.tiff` (16-bit recommended for precision)

## Quick Start

### 1. Dataset Preparation

First, validate your dataset structure:

```bash
python rs_utils.py validate --dataset-root /path/to/your/rs/dataset
```

Analyze dataset statistics:

```bash
python rs_utils.py analyze --dataset-root /path/to/your/rs/dataset
```

Create a sample dataset for testing:

```bash
python rs_utils.py create-sample --dataset-root ./sample_rs_data --num-samples 20
```

### 2. Training

Train the model for height estimation:

```bash
# Basic training
python train_rs.py -m zoedepth -d remote_sensing --rs-root /path/to/your/rs/dataset

# With custom parameters
python train_rs.py -m zoedepth -d remote_sensing \
    --rs-root /path/to/your/rs/dataset \
    --height-scale-factor 1.0 \
    --max-height 200 \
    --validate-data
```

### 3. Inference

Run inference on RS imagery:

```bash
# Single image
python run_rs.py --img-path /path/to/image.png --outdir ./results

# Directory of images  
python run_rs.py --img-path /path/to/images/ --outdir ./results --save-raw

# With custom height scaling
python run_rs.py --img-path /path/to/images/ --outdir ./results \
    --height-scale 1.0 --max-height 150
```

## Configuration Parameters

Key configuration parameters for RS height estimation:

### Dataset Configuration
- `rs_root`: Root directory of RS dataset
- `height_scale_factor`: Scale factor for DSM values (default: 1.0)
- `max_height`: Maximum height in meters for clipping (default: 200)
- `input_height/width`: Input image resolution (default: 512×512)

### Training Parameters
- `min_depth/max_depth`: Height range for training (0.1-200m)
- `do_random_rotate`: Disabled for RS (images are geo-referenced)
- `eigen_crop/garg_crop/do_kb_crop`: All disabled for RS

## Implementation Details

### Key Changes from Original Depth Anything:

1. **Dataset Loader (`rs_loader.py`)**:
   - Handles RS-specific directory structure
   - Supports multiple image formats including GeoTIFF
   - Handles invalid/no-data values in DSMs
   - Configurable height scaling and clipping

2. **Data Pipeline Integration**:
   - Added RS dataset support to `data_mono.py`
   - RS-specific configuration in `config.py`
   - Proper handling of train/valid/test splits

3. **Training Script (`train_rs.py`)**:
   - RS-specific parameter handling
   - Dataset validation before training
   - Adapted for height estimation metrics

4. **Inference Script (`run_rs.py`)**:
   - Optimized for RS imagery processing
   - Height-specific visualization and statistics
   - Raw height value export (.npy format)
   - Side-by-side comparison outputs

5. **Utilities (`rs_utils.py`)**:
   - Dataset structure validation
   - Statistical analysis of height data
   - Sample dataset generation for testing

### Model Architecture

The core architecture remains unchanged:
- **Encoder**: DINOv2 ViT (Vision Transformer)
- **Decoder**: DPT (Dense Prediction Transformer)
- **Output**: Single-channel height maps

The key adaptation is in data handling and interpretation:
- Input: RGB satellite/aerial images
- Ground truth: DSM height values
- Output: Predicted height values (meters above ground)

## Height vs Depth Considerations

**Important**: The model predicts "depth" from a camera perspective, but for remote sensing we want "height" above ground:

1. **Coordinate Systems**: 
   - Ground-level depth: Distance from camera (0 = camera, ∞ = far)
   - RS height: Elevation above ground (0 = ground level, + = above ground)

2. **Potential Inversion**: 
   - You may need to invert the output: `height = max_height - depth_prediction`
   - This depends on your specific coordinate system and DSM format

3. **Scale Calibration**: 
   - Use `height_scale_factor` to convert raw DSM values to meters
   - Use ground truth data to calibrate the scale

## Performance Tips

1. **Data Preprocessing**:
   - Ensure consistent image sizes (512×512 recommended)
   - Normalize height values appropriately  
   - Handle no-data/invalid values in DSMs

2. **Training**:
   - Start with pre-trained Depth Anything weights
   - Use appropriate height range (min_depth/max_depth)
   - Consider data augmentation carefully (avoid breaking geo-registration)

3. **Inference**:
   - Process images in batches for efficiency
   - Use `--save-raw` to preserve exact height values
   - Calibrate output scale with ground truth measurements

## Troubleshooting

### Common Issues:

1. **Dataset Structure**: 
   - Use `rs_utils.py validate` to check structure
   - Ensure RGB and DSM files have matching names

2. **Height Scaling**:
   - Check DSM units (meters, centimeters, etc.)
   - Adjust `height_scale_factor` accordingly

3. **No-Data Values**:
   - DSMs often contain no-data regions
   - The loader handles common no-data values automatically

4. **Memory Issues**:
   - Reduce batch size for large images
   - Consider image resizing for very high-resolution data

### Performance Monitoring:

The training script provides RS-specific metrics:
- Height range statistics
- Valid pixel ratios
- Scale-appropriate error metrics

## Citation

If you use this RS adaptation, please cite both the original Depth Anything paper and acknowledge the adaptation:

```bibtex
@article{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      journal={arXiv:2401.10891},
      year={2024},
}
```

## Contributing

This RS adaptation maintains compatibility with the original Depth Anything pipeline while adding RS-specific functionality. Contributions welcome for:

- Additional RS dataset formats
- Improved height calibration methods  
- Performance optimizations for large-scale RS data
- Evaluation metrics specific to height estimation