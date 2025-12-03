#!/usr/bin/env python3
"""
Master Pipeline Runner - 300 Images Model
Runs both PyTorch and INT8 TFLite pipelines
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*80)
    print(f"  {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED (exit code: {e.returncode})")
        return False

def main():
    print("="*80)
    print("  YOLO11 Complete Pipeline - 300 Images Model")
    print("="*80)
    print("\nThis script will:")
    print("  1. Run PyTorch (.pt) model pipeline → pipeline_output_300pt/")
    print("  2. Export PyTorch to INT8 TFLite → models/tflite/")
    print("  3. Run INT8 TFLite pipeline → pipeline_output_300INT8tflite/")
    print("="*80)
    
    # Check if we're in the YOLO directory
    if not Path('models/pytorch/best.pt').exists():
        print("\n❌ Error: Run this script from the YOLO/ directory")
        print("   Current directory should contain: models/pytorch/best.pt")
        return 1
    
    # Step 1: Run PyTorch pipeline
    if not run_command(
        ['python3', 'run_pipeline_300pt.py'],
        'Step 1: PyTorch Model Pipeline'
    ):
        print("\n⚠️  PyTorch pipeline failed, but continuing...")
    
    # Step 2: Export to INT8 TFLite
    int8_model = Path('models/tflite/best_yolo11_300_int8.tflite')
    
    if int8_model.exists():
        print(f"\n✅ INT8 model already exists: {int8_model}")
        print("   Skipping export step (delete file to re-export)")
    else:
        if not run_command(
            ['python3', 'export_int8_300pt.py'],
            'Step 2: Export to INT8 TFLite'
        ):
            print("\n❌ INT8 export failed!")
            print("\nℹ️  You can still use the existing INT8 model if available:")
            print("   models/tflite/best_yolo_int8.tflite")
            
            # Ask if user wants to continue with existing model
            response = input("\nContinue with existing INT8 model? (y/n): ")
            if response.lower() != 'y':
                return 1
    
    # Step 3: Run INT8 TFLite pipeline
    if not run_command(
        ['python3', 'run_pipeline_300INT8tflite.py'],
        'Step 3: INT8 TFLite Pipeline'
    ):
        print("\n❌ INT8 TFLite pipeline failed!")
        return 1
    
    # Summary
    print("\n" + "="*80)
    print("  🎉 ALL PIPELINES COMPLETE!")
    print("="*80)
    
    print("\n📊 Results:")
    
    if Path('pipeline_output_300pt').exists():
        pt_images = len(list(Path('pipeline_output_300pt').glob('*_detected.jpg')))
        print(f"  ✅ PyTorch Pipeline: {pt_images} images → pipeline_output_300pt/")
    
    if Path('pipeline_output_300INT8tflite').exists():
        int8_images = len(list(Path('pipeline_output_300INT8tflite').glob('*_detected.jpg')))
        print(f"  ✅ INT8 TFLite Pipeline: {int8_images} images → pipeline_output_300INT8tflite/")
    
    if int8_model.exists():
        print(f"\n📦 INT8 Model for QIDK NPU:")
        print(f"  {int8_model}")
        print(f"  Size: {int8_model.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  • Compare outputs in pipeline_output_300pt/ vs pipeline_output_300INT8tflite/")
    print("  • Deploy INT8 model to QIDK device for NPU testing")
    print("  • Check pipeline_summary.json in each output folder for details")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
