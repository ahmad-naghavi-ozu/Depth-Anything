# Remote Sensing Configuration Guide

This guide helps you configure the RS height estimation system for your specific dataset and requirements.

## Dataset Configuration

### 1. Directory Structure Setup

Ensure your dataset follows this structure:
```
your_dataset/
├── train/
│   ├── dsm/     # Digital Surface Models (height maps)
│   ├── rgb/     # RGB imagery
│   └── sem/     # Semantic segmentation (optional)
├── valid/
│   ├── dsm/
│   ├── rgb/
│   └── sem/
└── test/
    ├── dsm/
    ├── rgb/
    └── sem/
```

### 2. File Naming Convention

- RGB and DSM files should have **matching names** (except extension)
- Examples:
  - `rgb/image_001.png` ↔ `dsm/image_001.tif`
  - `rgb/tile_45_67.jpg` ↔ `dsm/tile_45_67.png`

## Key Configuration Parameters

### Height Scale Factor (`height_scale_factor`)

**Purpose**: Converts your DSM pixel values to meters

**Common scenarios**:
- DSM values are already in meters: `height_scale_factor = 1.0`
- DSM values are in centimeters: `height_scale_factor = 0.01`
- DSM values are in millimeters: `height_scale_factor = 0.001`
- DSM values are encoded (e.g., stored as uint16 scaled by 100): `height_scale_factor = 0.01`

**How to determine**:
1. Check your DSM metadata/documentation
2. Look at pixel value ranges: `python rs_utils.py analyze --dataset-root your_dataset`
3. Compare with known building heights in your area

### Maximum Height (`max_height`)

**Purpose**: Sets the upper limit for building heights (in meters)

**Typical values**:
- Residential areas: `50-100` meters
- Urban areas: `100-200` meters  
- High-rise cities: `200-500` meters
- Mixed areas: `200` meters (default)

### Input Resolution (`input_height`, `input_width`)

**Purpose**: Model input size (images are resized to this)

**Recommendations**:
- Default: `512 × 512` (good balance of speed/quality)
- High detail needed: `768 × 768` or `1024 × 1024`
- Fast inference: `384 × 384`
- Must be multiple of 14 (ViT constraint)

### Depth Range (`min_depth`, `max_depth`)

**Purpose**: Valid height range for training

**Settings**:
- `min_depth`: Usually `0.1` (10cm minimum)
- `max_depth`: Same as `max_height`

## Configuration Examples

### Example 1: High-resolution Urban Dataset
```python
# For city dataset with tall buildings
config = {
    "height_scale_factor": 1.0,      # DSM already in meters
    "max_height": 300,               # Up to 300m buildings  
    "input_height": 768,             # High resolution
    "input_width": 768,
    "min_depth": 0.1,
    "max_depth": 300
}
```

### Example 2: Suburban/Rural Dataset
```python
# For suburban area with mostly low buildings
config = {
    "height_scale_factor": 0.01,     # DSM in centimeters
    "max_height": 50,                # Max 50m buildings
    "input_height": 512,             # Standard resolution
    "input_width": 512,
    "min_depth": 0.1,
    "max_depth": 50
}
```

### Example 3: Mixed Development Dataset
```python
# For mixed urban/suburban with encoded DSM
config = {
    "height_scale_factor": 0.01,     # Uint16 scaled by 100
    "max_height": 200,               # Up to 200m
    "input_height": 512,
    "input_width": 512,
    "min_depth": 0.1,
    "max_depth": 200
}
```

## Command Line Usage

### Training with Custom Configuration
```bash
python train_rs.py \
    -m zoedepth \
    -d remote_sensing \
    --rs-root /path/to/dataset \
    --height-scale-factor 1.0 \
    --max-height 200 \
    --input_height 512 \
    --input_width 512 \
    --validate-data
```

### Inference with Custom Parameters
```bash
python run_rs.py \
    --img-path /path/to/images/ \
    --outdir ./results \
    --height-scale 1.0 \
    --max-height 200 \
    --save-raw
```

## Troubleshooting Common Issues

### Issue 1: Height values are too large/small
**Solution**: Adjust `height_scale_factor`
- If predicted heights are 100x too large: Use `height_scale_factor = 0.01`
- If predicted heights are 100x too small: Use `height_scale_factor = 100.0`

### Issue 2: Poor performance on tall buildings
**Solutions**: 
- Increase `max_height` parameter
- Use higher input resolution (`input_height/width`)
- Check if your training data includes tall buildings

### Issue 3: Model predicts negative heights
**Solutions**:
- Check DSM no-data value handling
- Ensure DSM represents heights above ground, not elevation
- Consider coordinate system differences

### Issue 4: Training crashes with memory error
**Solutions**:
- Reduce `input_height` and `input_width`
- Reduce batch size: `--bs 1`
- Use smaller model: `--encoder vits` instead of `vitl`

## Performance Optimization

### For Training:
1. **Data Loading**: Use fast storage (SSD) for dataset
2. **Batch Size**: Start with `bs=4`, adjust based on GPU memory
3. **Workers**: Set `--workers` to number of CPU cores
4. **Mixed Precision**: Enable `--use_amp` for faster training

### For Inference:
1. **Batch Processing**: Process multiple images at once
2. **Resolution**: Lower resolution for faster processing
3. **Output Format**: Skip visualization if only raw values needed

## Validation Checklist

Before training, verify:
- [ ] Dataset structure is correct (`python rs_utils.py validate`)
- [ ] Height values are in reasonable range (`python rs_utils.py analyze`) 
- [ ] `height_scale_factor` produces realistic building heights
- [ ] RGB and DSM images are properly paired
- [ ] No-data values are handled correctly

## Advanced Configuration

### Custom Dataset Root in Config File
Edit `metric_depth/zoedepth/utils/config.py`:
```python
"remote_sensing": {
    "rs_root": "/your/custom/path/to/dataset",
    "height_scale_factor": 1.0,
    "max_height": 200,
    # ... other parameters
}
```

### Environment Variables
```bash
export RS_DATASET_ROOT="/path/to/dataset"
export RS_HEIGHT_SCALE="1.0"
export RS_MAX_HEIGHT="200"
```

This configuration guide should help you adapt the system to your specific RS dataset and requirements.