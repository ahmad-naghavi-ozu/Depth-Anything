#!/bin/bash
################################################################################
# Training script for Remote Sensing Height Estimation
# Usage: ./rs_height_train.sh [OPTIONS]
################################################################################

# Default configuration
CONDA_ENV="depth_anything_rs"
DATASET_ROOT="/home/asfand/Ahmad/datasets"
MODEL="zoedepth"
DATASET="dfc2023s"
MIDAS_MODEL="dinov2_base"
BATCH_SIZE=6
EPOCHS=100
WORKERS=4
EARLY_STOP_PATIENCE=10
DISTRIBUTED="True"
GPU_IDS="0,1,2,3"

# Additional arguments (empty by default, user can override)
EXTRA_ARGS=""

# Help function
show_help() {
    cat << EOF
Usage: ./rs_height_train.sh [OPTIONS]

Training script for remote sensing height estimation using ZoeDepth.

OPTIONS:
    -d, --dataset NAME         Dataset name (default: dfc2023s)
                              Options: dfc2023s, dfc2023mini, dfc2019_crp512_bin, huawei_contest
    --dataset-root PATH        Dataset root directory (default: /home/asfand/Ahmad/datasets)
    -b, --batch-size N         Batch size per GPU (default: 6)
    -e, --epochs N             Number of training epochs (default: 100)
    -w, --workers N            Number of data loading workers (default: 4)
    -p, --patience N           Early stopping patience (default: 10)
    --distributed BOOL         Use distributed training (default: True)
    --gpu-ids IDS              GPU IDs to use (default: 0,1,2,3)
    -r, --resume PATH          Resume from checkpoint path
    -h, --help                 Show this help message

EXAMPLES:
    # Basic training on DFC2023S with 4 GPUs
    ./rs_height_train.sh

    # Single GPU training on DFC2023mini
    ./rs_height_train.sh --dataset dfc2023mini --distributed False --gpu-ids 0 --batch-size 2

    # Training on DFC2019_crp512_bin dataset
    ./rs_height_train.sh --dataset DFC2019_crp512_bin --epochs 30 --batch-size 4

    # Resume training from checkpoint
    ./rs_height_train.sh --resume ./checkpoints/rs_height_zoedepth/DFC2023S/model_latest.pt

    # Training with custom dataset location
    ./rs_height_train.sh --dataset huawei_contest --dataset-root /custom/path --epochs 30

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -p|--patience)
            EARLY_STOP_PATIENCE="$2"
            shift 2
            ;;
        --distributed)
            DISTRIBUTED="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        -r|--resume)
            EXTRA_ARGS="$EXTRA_ARGS --resume $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Note: Dataset path validation is skipped because dataset names may use different casing
# than the actual directory names. The Python script will validate the actual path
# from the config file.

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/bin/activate "$CONDA_ENV" || {
    echo "Error: Failed to activate conda environment '$CONDA_ENV'"
    exit 1
}

# Set GPU visibility
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# Construct command
CMD="python rs_height_train.py \
    --model $MODEL \
    --dataset $DATASET \
    --midas_model_type $MIDAS_MODEL \
    --bs $BATCH_SIZE \
    --epochs $EPOCHS \
    --workers $WORKERS \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --distributed $DISTRIBUTED \
    $EXTRA_ARGS"

# Print configuration
echo ""
echo "=========================================="
echo "RS Height Estimation - Training"
echo "=========================================="
echo "Dataset:        $DATASET"
echo "Batch size:     $BATCH_SIZE"
echo "Epochs:         $EPOCHS"
echo "Workers:        $WORKERS"
echo "Patience:       $EARLY_STOP_PATIENCE"
echo "Distributed:    $DISTRIBUTED"
echo "GPU IDs:        $GPU_IDS"
echo "=========================================="
echo ""
echo "Command: $CMD"
echo ""

# Run training
$CMD 2>&1 | tee "logs/rs_height_train_${DATASET}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Training completed! Check logs/ directory for details."
