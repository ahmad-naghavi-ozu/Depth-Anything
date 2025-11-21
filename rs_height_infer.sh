#!/bin/bash
################################################################################
# Inference script for Remote Sensing Height Estimation
# Usage: ./rs_height_infer.sh [OPTIONS]
################################################################################

# Default configuration
CONDA_ENV="depth_anything_rs"
DEFAULT_DATASET_ROOT="/home/asfand/Ahmad/datasets"
MODEL="zoedepth"
CHECKPOINT=""
DATASET=""
DATASET_ROOT=""
SPLIT="test"
OUTPUT_DIR="./results/rs_height_zoedepth"
BATCH_SIZE=1
SAVE_PREDICTIONS="--save-predictions"

# Help function
show_help() {
    cat << EOF
Usage: ./rs_height_infer.sh [OPTIONS]

Inference script for remote sensing height estimation using trained ZoeDepth model.

OPTIONS:
    -c, --checkpoint PATH      Path to trained model checkpoint (REQUIRED)
    -d, --dataset NAME         Dataset name (e.g., dfc2023s, DFC2019_crp512_bin)
                              Will be combined with --dataset-root
    --dataset-root PATH        Root directory for datasets (default: /home/asfand/Ahmad/datasets)
                              Can also provide full dataset path directly with -d
    -s, --split NAME           Dataset split to evaluate (default: test)
                              Options: train, val, test
    -o, --output-dir PATH      Output directory (default: ./results/rs_height_zoedepth)
    -b, --batch-size N         Batch size for inference (default: 1)
    --no-save-predictions      Do not save prediction GeoTIFFs (saves disk space)
    -h, --help                 Show this help message

EXAMPLES:
    # Basic inference on DFC2023S test set
    ./rs_height_infer.sh \\
        --checkpoint ./checkpoints/rs_height_zoedepth/DFC2023S/model_best.pt \\
        --dataset dfc2023s

    # Inference on DFC2019_crp512_bin
    ./rs_height_infer.sh \\
        --checkpoint ./checkpoints/rs_height_zoedepth/DFC2023S/model_best.pt \\
        --dataset DFC2019_crp512_bin

    # Inference on validation set without saving predictions
    ./rs_height_infer.sh \\
        --checkpoint ./checkpoints/rs_height_zoedepth/DFC2023S/model_best.pt \\
        --dataset dfc2023s \\
        --split val \\
        --no-save-predictions

    # Inference with full dataset path
    ./rs_height_infer.sh \\
        --checkpoint ./checkpoints/model.pt \\
        --dataset /custom/path/to/Huawei_Contest \\
        --batch-size 4

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-save-predictions)
            SAVE_PREDICTIONS=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint path is required (use -c or --checkpoint)"
    echo ""
    show_help
    exit 1
fi

if [[ -z "$DATASET" ]]; then
    echo "Error: Dataset name is required (use -d or --dataset)"
    echo ""
    show_help
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Determine full dataset path
if [[ "$DATASET" == /* ]]; then
    # Absolute path provided
    FULL_DATASET_PATH="$DATASET"
else
    # Relative dataset name, use dataset root
    if [[ -z "$DATASET_ROOT" ]]; then
        DATASET_ROOT="$DEFAULT_DATASET_ROOT"
    fi
    FULL_DATASET_PATH="$DATASET_ROOT/$DATASET"
fi

# Note: Dataset path validation is skipped because the inference script
# will handle the actual path and provide better error messages if needed

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/bin/activate "$CONDA_ENV" || {
    echo "Error: Failed to activate conda environment '$CONDA_ENV'"
    exit 1
}

# Extract dataset name for logging
DATASET_NAME=$(basename "$FULL_DATASET_PATH")

# Construct command
CMD="python rs_height_infer.py \
    --checkpoint $CHECKPOINT \
    --dataset-root $FULL_DATASET_PATH \
    --split $SPLIT \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --model $MODEL \
    $SAVE_PREDICTIONS"

# Print configuration
echo ""
echo "=========================================="
echo "RS Height Estimation - Inference"
echo "=========================================="
echo "Checkpoint:     $CHECKPOINT"
echo "Dataset:        $DATASET_NAME"
echo "Dataset path:   $FULL_DATASET_PATH"
echo "Split:          $SPLIT"
echo "Batch size:     $BATCH_SIZE"
echo "Output dir:     $OUTPUT_DIR"
echo "Save preds:     $(if [[ -n "$SAVE_PREDICTIONS" ]]; then echo "Yes"; else echo "No"; fi)"
echo "=========================================="
echo ""
echo "Command: $CMD"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
$CMD 2>&1 | tee "$OUTPUT_DIR/infer_${DATASET_NAME}_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Inference completed! Results saved to: $OUTPUT_DIR"
