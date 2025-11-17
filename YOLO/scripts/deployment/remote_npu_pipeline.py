#!/usr/bin/env python3
"""
Remote NPU Pipeline - Run YOLO on QIDK NPU from PC
===================================================

This script orchestrates the complete detection pipeline:
1. Preprocessing on PC (faster, easier)
2. Inference on QIDK NPU (optimized hardware)
3. Postprocessing on PC (with visualization)

Usage:
    python remote_npu_pipeline.py --image <path> --dlc <dlc_path>

Example:
    python remote_npu_pipeline.py \
        --image /home/harshyy/Desktop/20250103_104457.jpg \
        --dlc runs/detect/train/weights/best_yolo_int8.dlc

Requirements:
    - QIDK connected via USB (adb devices should show it)
    - SNPE runtime on QIDK (/data/local/tmp/yolo_npu/)
    - DLC model on QIDK
"""

import argparse
import subprocess
import time
import sys
import numpy as np
import cv2
import json
from pathlib import Path


CLASS_NAMES = ['worker', 'truck', 'bike', 'bulldozer', 'car']


def check_adb_connection():
    """Check if QIDK is connected via ADB"""
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, check=True)
        devices = [line.split('\t')[0] for line in result.stdout.split('\n')[1:] if '\tdevice' in line]
        
        if not devices:
            print("✗ No ADB devices found!")
            print("  Connect QIDK via USB and enable USB debugging")
            return None
        
        print(f"✓ Found {len(devices)} ADB device(s):")
        for i, dev in enumerate(devices, 1):
            print(f"  {i}. {dev}")
        
        return devices[0] if len(devices) == 1 else devices
    
    except FileNotFoundError:
        print("✗ ADB not found! Install Android Debug Bridge:")
        print("  sudo apt install adb")
        return None
    except subprocess.CalledProcessError as e:
        print(f"✗ ADB error: {e}")
        return None


def preprocess_image(image_path, img_size=1024):
    """Preprocess image to raw format"""
    print(f"[1/6] Preprocessing image...")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    orig_shape = img.shape[:2]
    print(f"  Original size: {orig_shape[1]}x{orig_shape[0]}")
    
    # Resize
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # HWC to CHW
    img_chw = np.transpose(img_float, (2, 0, 1))
    
    print(f"  Preprocessed shape: {img_chw.shape}")
    return img, img_chw, orig_shape


def push_to_device(device, local_path, remote_path):
    """Push file to QIDK via ADB"""
    cmd = ['adb', '-s', device, 'push', str(local_path), remote_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to push {local_path}: {e.stderr.decode()}")
        return False


def pull_from_device(device, remote_path, local_path):
    """Pull file from QIDK via ADB"""
    cmd = ['adb', '-s', device, 'pull', remote_path, str(local_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to pull {remote_path}: {e.stderr.decode()}")
        return False


def run_inference_on_qidk(device, dlc_name, input_name, runtime='dsp'):
    """Run inference on QIDK NPU"""
    print(f"[4/6] Running inference on QIDK NPU...")
    
    # Build command
    cmd = [
        'adb', '-s', device, 'shell',
        f'cd /data/local/tmp/yolo_npu && '
        f'LD_LIBRARY_PATH=/data/local/tmp/yolo_npu:$LD_LIBRARY_PATH '
        f'./snpe-net-run '
        f'--container {dlc_name} '
        f'--input_raw {input_name} '
        f'--use_{runtime} '
        f'--output_dir output/'
    ]
    
    print(f"  Runtime: {runtime.upper()}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        inference_time = (time.time() - start_time) * 1000
        print(f"  ✓ Inference complete: {inference_time:.1f} ms")
        return inference_time
    except subprocess.TimeoutExpired:
        print(f"  ✗ Inference timeout (>30s)")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Inference failed:")
        print(f"    {e.stderr}")
        return None


def postprocess_outputs(raw_output, orig_shape, conf_threshold=0.25, iou_threshold=0.45, img_size=1024):
    """Postprocess NPU output to get detections"""
    print(f"[5/6] Postprocessing detections...")
    
    # Reshape if needed
    if raw_output.ndim == 1:
        raw_output = raw_output.reshape(1, 9, 21504)
    elif raw_output.ndim == 2:
        raw_output = raw_output.reshape(1, 9, 21504)
    
    # Extract components
    x_center = raw_output[0, 0, :]
    y_center = raw_output[0, 1, :]
    width = raw_output[0, 2, :]
    height = raw_output[0, 3, :]
    class_probs = raw_output[0, 4:, :]
    
    # Convert xywh to xyxy
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Get max class
    class_scores = np.max(class_probs, axis=0)
    class_ids = np.argmax(class_probs, axis=0)
    
    # Filter by confidence
    mask = class_scores > conf_threshold
    boxes_xyxy = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1)
    scores_filtered = class_scores[mask]
    class_ids_filtered = class_ids[mask]
    
    if len(boxes_xyxy) == 0:
        print(f"  No detections above threshold {conf_threshold}")
        return np.array([]), np.array([]), np.array([])
    
    # NMS
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
        
        # Scale to original image
        orig_h, orig_w = orig_shape
        scale_x = orig_w / img_size
        scale_y = orig_h / img_size
        
        final_boxes[:, [0, 2]] *= scale_x
        final_boxes[:, [1, 3]] *= scale_y
        
        print(f"  ✓ Found {len(final_boxes)} detections after NMS")
        return final_boxes, final_scores, final_class_ids
    
    return np.array([]), np.array([]), np.array([])


def visualize_results(image, boxes, scores, class_ids, output_path='remote_npu_result.jpg'):
    """Draw bounding boxes and save"""
    print(f"[6/6] Visualizing results...")
    
    img_vis = image.copy()
    
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES[int(cls_id)]
        
        # Color coding
        if 'worker' in class_name:
            color = (0, 255, 0)  # Green
        elif 'truck' in class_name:
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 165, 255)  # Orange
        
        # Draw box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_vis, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, img_vis)
    print(f"  ✓ Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Remote NPU Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--dlc", type=str, required=True, help="Path to DLC file")
    parser.add_argument("--device", type=str, default=None, help="ADB device serial (auto-detect if omitted)")
    parser.add_argument("--runtime", choices=['dsp', 'gpu', 'cpu'], default='dsp',
                       help="QIDK runtime (dsp=NPU, gpu=GPU, cpu=CPU)")
    parser.add_argument("--output", type=str, default="remote_npu_result.jpg",
                       help="Output visualization path")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Remote NPU Pipeline - QIDK Inference from PC")
    print(f"{'='*80}\n")
    
    # Check ADB connection
    devices = check_adb_connection()
    if devices is None:
        sys.exit(1)
    
    device = args.device if args.device else (devices[0] if isinstance(devices, list) else devices)
    print(f"Using device: {device}\n")
    
    # Check files exist
    image_path = Path(args.image)
    dlc_path = Path(args.dlc)
    
    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        sys.exit(1)
    
    if not dlc_path.exists():
        print(f"✗ DLC not found: {dlc_path}")
        sys.exit(1)
    
    # Preprocess on PC
    original_image, input_tensor, orig_shape = preprocess_image(image_path)
    
    # Save as raw
    temp_input = Path("/tmp/remote_npu_input.raw")
    input_tensor.tofile(str(temp_input))
    print(f"  Saved to: {temp_input}")
    
    # Setup device directory
    print(f"\n[2/6] Setting up QIDK device...")
    subprocess.run(['adb', '-s', device, 'shell', 'mkdir', '-p', '/data/local/tmp/yolo_npu'],
                  capture_output=True)
    print(f"  ✓ Created /data/local/tmp/yolo_npu/")
    
    # Push files
    print(f"\n[3/6] Pushing files to QIDK...")
    
    # Push DLC (if not already there)
    dlc_name = dlc_path.name
    print(f"  Pushing DLC: {dlc_name}")
    if not push_to_device(device, dlc_path, f'/data/local/tmp/yolo_npu/{dlc_name}'):
        sys.exit(1)
    
    # Push input
    input_name = temp_input.name
    print(f"  Pushing input: {input_name}")
    if not push_to_device(device, temp_input, f'/data/local/tmp/yolo_npu/{input_name}'):
        sys.exit(1)
    
    print(f"  ✓ Files pushed")
    
    # Run inference on QIDK
    inference_time = run_inference_on_qidk(device, dlc_name, input_name, args.runtime)
    if inference_time is None:
        sys.exit(1)
    
    # Pull results
    print(f"\n  Pulling output tensor...")
    temp_output = Path("/tmp/remote_npu_output.raw")
    if not pull_from_device(device, '/data/local/tmp/yolo_npu/output/Result_0/output0.raw', temp_output):
        print(f"  ✗ Failed to pull output")
        sys.exit(1)
    
    # Load output
    raw_output = np.fromfile(str(temp_output), dtype=np.float32)
    print(f"  ✓ Output loaded: {raw_output.shape}")
    
    # Postprocess on PC
    boxes, scores, class_ids = postprocess_outputs(raw_output, orig_shape)
    
    # Visualize
    visualize_results(original_image, boxes, scores, class_ids, args.output)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"RESULTS - Remote NPU Inference")
    print(f"{'='*80}\n")
    
    print(f"Device: {device}")
    print(f"Runtime: {args.runtime.upper()}")
    print(f"Inference time: {inference_time:.1f} ms")
    print(f"Total detections: {len(boxes)}\n")
    
    if len(boxes) > 0:
        print("Detections:")
        for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids), 1):
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES[int(cls_id)]
            print(f"  {i}. {class_name:12s} (conf: {score:.3f}) @ [{x1:4d}, {y1:4d}, {x2:4d}, {y2:4d}]")
    
    print(f"\n{'='*80}")
    print(f"✓ Remote Pipeline Complete")
    print(f"  Output saved: {args.output}")
    print(f"{'='*80}\n")
    
    # Save JSON report
    json_output = Path(args.output).with_suffix('.json')
    report = {
        'device': device,
        'runtime': args.runtime,
        'inference_time_ms': float(inference_time),
        'total_detections': len(boxes),
        'detections': [
            {
                'class': CLASS_NAMES[int(cls_id)],
                'confidence': float(score),
                'bbox': [int(x) for x in box]
            }
            for box, score, cls_id in zip(boxes, scores, class_ids)
        ]
    }
    
    with open(json_output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"JSON report saved: {json_output}")


if __name__ == "__main__":
    main()
