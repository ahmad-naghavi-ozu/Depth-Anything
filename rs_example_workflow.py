#!/usr/bin/env python3

"""
Example workflow for Remote Sensing Height Estimation
This script demonstrates the complete pipeline from dataset validation to inference
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False
    except FileNotFoundError:
        print("❌ COMMAND NOT FOUND")
        print(f"Make sure the script exists: {cmd[1]}")
        return False


def main():
    parser = argparse.ArgumentParser(description='RS Height Estimation Workflow Example')
    parser.add_argument('--dataset-root', type=str, required=True,
                       help='Root directory of RS dataset')
    parser.add_argument('--output-dir', type=str, default='./rs_workflow_output',
                       help='Output directory for results')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training step (for testing inference only)')
    parser.add_argument('--sample-image', type=str, default=None,
                       help='Single image for inference test')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🛰️  Remote Sensing Height Estimation Workflow")
    print(f"Dataset: {dataset_root}")
    print(f"Output: {output_dir}")
    
    # Step 1: Validate dataset structure
    if not run_command([
        'python', 'rs_utils.py', 'validate', 
        '--dataset-root', str(dataset_root)
    ], "Validating dataset structure"):
        print("\n❌ Dataset validation failed. Please fix your dataset structure.")
        print("Expected structure:")
        print("  [DATASET]/")
        print("  ├── train/")  
        print("  │   ├── dsm/")
        print("  │   └── rgb/")
        print("  ├── valid/")
        print("  │   ├── dsm/")
        print("  │   └── rgb/")
        print("  └── test/")
        print("      ├── dsm/")
        print("      └── rgb/")
        return False
    
    # Step 2: Analyze dataset statistics
    run_command([
        'python', 'rs_utils.py', 'analyze',
        '--dataset-root', str(dataset_root)
    ], "Analyzing dataset statistics")
    
    # Step 3: Training (optional)
    if not args.skip_training:
        training_output = output_dir / "training"
        training_output.mkdir(exist_ok=True)
        
        if run_command([
            'python', 'train_rs.py',
            '-m', 'zoedepth',
            '-d', 'remote_sensing',
            '--rs-root', str(dataset_root),
            '--validate-data',
            '--height-scale-factor', '1.0',
            '--max-height', '200'
        ], "Training RS height estimation model"):
            print("✅ Training completed successfully!")
        else:
            print("⚠️  Training failed, but continuing with inference using pre-trained weights...")
    else:
        print("⏭️  Skipping training step")
    
    # Step 4: Inference
    inference_output = output_dir / "inference"
    inference_output.mkdir(exist_ok=True)
    
    # Determine what to use for inference
    if args.sample_image:
        inference_input = args.sample_image
    else:
        # Use a test image from the dataset
        test_rgb_dir = dataset_root / "test" / "rgb"
        if test_rgb_dir.exists():
            # Find first image file
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                files = list(test_rgb_dir.glob(ext))
                if files:
                    inference_input = str(files[0])
                    break
            else:
                # Try uppercase extensions
                for ext in ['*.PNG', '*.JPG', '*.JPEG', '*.TIF', '*.TIFF']:
                    files = list(test_rgb_dir.glob(ext))
                    if files:
                        inference_input = str(files[0])
                        break
                else:
                    print("⚠️  No test images found, using train directory")
                    train_rgb_dir = dataset_root / "train" / "rgb"
                    if train_rgb_dir.exists():
                        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                            files = list(train_rgb_dir.glob(ext))
                            if files:
                                inference_input = str(files[0])
                                break
                        else:
                            print("❌ No images found for inference")
                            return False
        else:
            print("❌ No test directory found")
            return False
    
    print(f"Using image for inference: {inference_input}")
    
    if run_command([
        'python', 'run_rs.py',
        '--img-path', inference_input,
        '--outdir', str(inference_output),
        '--save-raw',
        '--height-scale', '1.0'
    ], f"Running inference on {inference_input}"):
        print("✅ Inference completed successfully!")
        print(f"Results saved to: {inference_output}")
        
        # List output files
        output_files = list(inference_output.glob("*"))
        if output_files:
            print("\nGenerated files:")
            for f in output_files:
                print(f"  📁 {f.name}")
    else:
        print("❌ Inference failed")
        return False
    
    # Step 5: Summary and next steps
    print(f"\n{'='*60}")
    print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"📁 All outputs saved to: {output_dir}")
    print("\nNext steps:")
    print("1. 📊 Review the inference results in the output directory")
    print("2. 📏 Check the height values make sense for your application")
    print("3. 🔧 Adjust height_scale_factor if needed")
    print("4. 🚀 Scale up to process your full dataset")
    print("\nFor batch processing:")
    print(f"  python run_rs.py --img-path {dataset_root}/test/rgb/ --outdir ./batch_results/")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)