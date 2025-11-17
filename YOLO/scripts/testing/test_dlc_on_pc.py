#!/usr/bin/env python3
"""
Test DLC Model on PC (CPU/GPU)
================================

Tests DLC files on PC without requiring NPU hardware.
Useful for validation before deploying to QIDK.

Usage:
    python test_dlc_on_pc.py <dlc_file> [--image <path>]

Example:
    python test_dlc_on_pc.py runs/detect/train/weights/best_yolo_int8.dlc
    python test_dlc_on_pc.py best_yolo_fp32.dlc --image /path/to/test.jpg

Requirements:
    - SNPE SDK installed and in PATH
    - SNPE Python bindings available
"""

import sys
import os
import argparse
import numpy as np
import cv2
from pathlib import Path

# Try to import SNPE
try:
    from snpe import modeltools
    SNPE_AVAILABLE = True
except ImportError:
    SNPE_AVAILABLE = False
    print("⚠️  SNPE Python bindings not found!")
    print("   Falling back to snpe-net-run command-line tool")


def preprocess_image(image_path, img_size=1024):
    """Preprocess image to raw format for DLC"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    orig_shape = img.shape[:2]
    
    # Resize
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0.0, 1.0]
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # HWC to CHW
    img_chw = np.transpose(img_float, (2, 0, 1))
    
    return img_chw, orig_shape


def postprocess_outputs(raw_output, orig_shape, conf_threshold=0.25, iou_threshold=0.45, img_size=1024):
    """Postprocess DLC output to get detections"""
    # raw_output shape: (1, 9, 21504) or (9, 21504)
    if raw_output.ndim == 3:
        raw_output = raw_output[0]  # Remove batch dimension
    
    # Extract components
    x_center = raw_output[0, :]
    y_center = raw_output[1, :]
    width = raw_output[2, :]
    height = raw_output[3, :]
    class_probs = raw_output[4:, :]  # (5, 21504)
    
    # Convert xywh to xyxy
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Get max class probability and class ID
    class_scores = np.max(class_probs, axis=0)
    class_ids = np.argmax(class_probs, axis=0)
    
    # Filter by confidence
    mask = class_scores > conf_threshold
    boxes_xyxy = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1)
    scores_filtered = class_scores[mask]
    class_ids_filtered = class_ids[mask]
    
    # NMS
    if len(boxes_xyxy) == 0:
        return np.array([]), np.array([]), np.array([])
    
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        scores_filtered.tolist(),
        conf_threshold,
        iou_threshold
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        final_boxes = boxes_xyxy[indices]
        final_scores = scores_filtered[indices]
        final_class_ids = class_ids_filtered[indices]
        
        # Scale to original image size
        orig_h, orig_w = orig_shape
        scale_x = orig_w / img_size
        scale_y = orig_h / img_size
        
        final_boxes[:, [0, 2]] *= scale_x
        final_boxes[:, [1, 3]] *= scale_y
        
        return final_boxes, final_scores, final_class_ids
    else:
        return np.array([]), np.array([]), np.array([])


def test_with_snpe_python(dlc_path, image_path, runtime='cpu'):
    """Test using SNPE Python API"""
    print(f"Testing with SNPE Python API (runtime: {runtime})")
    # TODO: Implement when SNPE Python bindings are properly installed
    print("⚠️  SNPE Python API not fully implemented in this version")
    print("   Using command-line fallback...")
    return test_with_command_line(dlc_path, image_path, runtime)


def test_with_command_line(dlc_path, image_path, runtime='cpu'):
    """Test using snpe-net-run command-line tool"""
    import subprocess
    import time
    
    print(f"\n{'='*80}")
    print(f"Testing DLC Model on PC ({runtime.upper()})")
    print(f"{'='*80}\n")
    
    dlc_path = Path(dlc_path)
    image_path = Path(image_path)
    
    # Create temp directory
    temp_dir = Path("/tmp/dlc_test")
    temp_dir.mkdir(exist_ok=True)
    
    # Preprocess image
    print(f"[1/4] Preprocessing image: {image_path.name}")
    img_chw, orig_shape = preprocess_image(image_path)
    print(f"  Original shape: {orig_shape[0]}x{orig_shape[1]}")
    print(f"  Input tensor shape: {img_chw.shape}")
    
    # Save as raw
    input_raw = temp_dir / "input.raw"
    img_chw.tofile(str(input_raw))
    print(f"  Saved to: {input_raw}")
    
    # Run inference
    print(f"\n[2/4] Running inference with snpe-net-run...")
    output_dir = temp_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    runtime_flag = f"--use_{runtime}"
    
    cmd = [
        "snpe-net-run",
        "--container", str(dlc_path),
        "--input_raw", str(input_raw),
        runtime_flag,
        "--output_dir", str(output_dir)
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        inference_time = (time.time() - start_time) * 1000
        print(f"  ✓ Inference complete in {inference_time:.1f} ms")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error running snpe-net-run:")
        print(f"    {e.stderr}")
        return None
    except FileNotFoundError:
        print(f"  ✗ snpe-net-run not found!")
        print(f"    Make sure SNPE SDK is installed and in PATH")
        print(f"    Run: export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH")
        return None
    
    # Load output
    print(f"\n[3/4] Loading output tensor...")
    output_raw = output_dir / "Result_0" / "output0.raw"
    
    if not output_raw.exists():
        print(f"  ✗ Output file not found: {output_raw}")
        print(f"    Available files:")
        for f in output_dir.rglob("*"):
            if f.is_file():
                print(f"      {f}")
        return None
    
    raw_output = np.fromfile(str(output_raw), dtype=np.float32)
    raw_output = raw_output.reshape(1, 9, 21504)
    print(f"  Output shape: {raw_output.shape}")
    
    # Postprocess
    print(f"\n[4/4] Postprocessing (NMS + decoding)...")
    boxes, scores, class_ids = postprocess_outputs(raw_output, orig_shape)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"RESULTS - {len(boxes)} Detections")
    print(f"{'='*80}\n")
    
    class_names = ['worker', 'truck', 'bike', 'bulldozer', 'car']
    
    if len(boxes) == 0:
        print("  No detections found!")
    else:
        for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(cls_id)]
            print(f"  {i+1}. {class_name:12s} (conf: {score:.3f}) @ [{x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}]")
    
    print(f"\n{'='*80}")
    print(f"✓ PC Testing Complete")
    print(f"  Runtime: {runtime.upper()}")
    print(f"  Inference: {inference_time:.1f} ms")
    print(f"  Detections: {len(boxes)}")
    print(f"{'='*80}\n")
    
    return {
        'boxes': boxes,
        'scores': scores,
        'class_ids': class_ids,
        'inference_time_ms': inference_time,
        'runtime': runtime
    }


def main():
    parser = argparse.ArgumentParser(description="Test DLC model on PC")
    parser.add_argument("dlc", type=str, help="Path to DLC file")
    parser.add_argument("--image", type=str, 
                       default="/home/harshyy/Desktop/20250103_104457.jpg",
                       help="Path to test image")
    parser.add_argument("--runtime", choices=['cpu', 'gpu'], default='cpu',
                       help="Execution runtime (cpu or gpu)")
    args = parser.parse_args()
    
    # Check if DLC exists
    if not Path(args.dlc).exists():
        print(f"✗ DLC file not found: {args.dlc}")
        sys.exit(1)
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"✗ Image file not found: {args.image}")
        sys.exit(1)
    
    # Run test
    if SNPE_AVAILABLE:
        results = test_with_snpe_python(args.dlc, args.image, args.runtime)
    else:
        results = test_with_command_line(args.dlc, args.image, args.runtime)
    
    if results is None:
        print("\n✗ Testing failed!")
        sys.exit(1)
    
    print("Tip: Compare results with ONNX validation:")
    print("  python validate_onnx_for_npu.py")
    print("  Expected: ~11 detections (9 workers, 2 trucks)")


if __name__ == "__main__":
    main()
