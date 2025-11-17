#!/usr/bin/env python3
"""
Prepare calibration data for QNN INT8 quantization
Converts JPG images to raw tensor format expected by QNN
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def preprocess_image_to_raw(image_path, output_path, input_size=(1024, 1024)):
    """Convert image to raw tensor format for QNN"""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load: {image_path}")
        return False
    
    # Resize
    img_resized = cv2.resize(img, input_size)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Convert to CHW format
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension: (1, 3, 1024, 1024)
    img_input = np.expand_dims(img_chw, axis=0)
    
    # Save as raw
    img_input.tofile(str(output_path))
    return True

def main():
    print("="*70)
    print("🔄 Preparing Calibration Data for QNN Quantization")
    print("="*70)
    
    # Read calibration list
    with open('calibration_list.txt', 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"\nFound {len(image_paths)} images in calibration list")
    
    # Create output directory
    raw_dir = Path('calibration_raw')
    raw_dir.mkdir(exist_ok=True)
    
    # Create new calibration list with raw files
    raw_list_path = 'calibration_list_raw.txt'
    
    print(f"Converting images to raw format...")
    raw_paths = []
    success_count = 0
    
    for img_path_str in tqdm(image_paths):
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"  Warning: {img_path} not found, skipping")
            continue
        
        # Create output path
        raw_filename = img_path.stem + '.raw'
        raw_path = raw_dir / raw_filename
        
        # Convert and save
        if preprocess_image_to_raw(img_path, raw_path):
            raw_paths.append(str(raw_path))
            success_count += 1
    
    # Save raw file list
    with open(raw_list_path, 'w') as f:
        for path in raw_paths:
            f.write(path + '\n')
    
    print(f"\n✅ Conversion complete!")
    print(f"   Converted: {success_count}/{len(image_paths)} images")
    print(f"   Output directory: {raw_dir}")
    print(f"   Calibration list: {raw_list_path}")
    print(f"\nNow run quantization with: --input_list {raw_list_path}")

if __name__ == "__main__":
    main()
