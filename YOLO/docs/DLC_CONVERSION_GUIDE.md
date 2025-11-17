# ONNX to DLC Conversion Guide for QIDK NPU Deployment
# =====================================================

This guide provides step-by-step commands to convert the validated ONNX model
to a quantized DLC file ready for QIDK NPU deployment.

## Prerequisites

Ensure you have Qualcomm SNPE or QNN SDK installed. Example setup:

```bash
# Set SNPE_ROOT to your installation path
export SNPE_ROOT=/opt/qcom/aistack/snpe-2.x.x.xxxx
export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH
```

For QNN:
```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn-2.x.x
export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
```

Verify installation:
```bash
snpe-onnx-to-dlc --help
# or for QNN:
qnn-onnx-converter --help
```

---

## Model Information

**Source Model**: `runs/detect/train/weights/best.pt` (PyTorch YOLOv11)
**ONNX Model**: `runs/detect/train/weights/best_simplified.onnx`
**Input**: `images` - shape [1, 3, 1024, 1024], dtype float32
**Output**: `output0` - shape [1, 9, 21504], dtype float32

**Validated Output**: 11 detections (9 workers + 2 trucks) on test image

---

## Step 1: Convert ONNX to DLC (FP32)

Navigate to YOLO directory:
```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO
```

### For SNPE:
```bash
snpe-onnx-to-dlc \
  --input_network runs/detect/train/weights/best_simplified.onnx \
  --output_path runs/detect/train/weights/best_yolo_fp32.dlc \
  --input_dim images 1,3,1024,1024
```

### For QNN (alternative):
```bash
qnn-onnx-converter \
  --input_network runs/detect/train/weights/best_simplified.onnx \
  --output_path runs/detect/train/weights/best_yolo_fp32.dlc \
  --input_dim images 1,3,1024,1024
```

**Expected output**: 
- File: `runs/detect/train/weights/best_yolo_fp32.dlc`
- This is a floating-point DLC (not optimized for NPU yet)

---

## Step 2: Inspect DLC (Optional but Recommended)

Check the converted DLC for any unsupported layers:

```bash
snpe-dlc-info --input_dlc runs/detect/train/weights/best_yolo_fp32.dlc
```

Look for:
- ✅ All layers supported
- ⚠️  Any "custom" or "unsupported" operations
- ✅ Input/output dimensions match expected

If you see unsupported operations, you may need to:
1. Simplify the ONNX graph further
2. Use CPU fallback for those layers
3. Adjust export parameters

For YOLOv11, all standard operations (Conv, Add, Concat, Sigmoid, etc.) should be supported.

---

## Step 3: Quantize to INT8 (NPU-Optimized)

This step converts the FP32 DLC to INT8 using calibration data for optimal NPU performance.

**Calibration data prepared**: 150 images in `calibration_list.txt`

### Run Quantization:

```bash
snpe-dlc-quantize \
  --input_dlc runs/detect/train/weights/best_yolo_fp32.dlc \
  --output_dlc runs/detect/train/weights/best_yolo_int8.dlc \
  --input_list calibration_list.txt \
  --enable_htp \
  --use_enhanced_quantizer
```

**Important flags explained**:
- `--input_list`: List of calibration images (absolute or relative paths)
- `--enable_htp`: Enable Hexagon Tensor Processor (NPU) optimizations
- `--use_enhanced_quantizer`: Better quantization accuracy
- `--optimizations cle`: (Optional) Cross-layer equalization for better accuracy

**Expected output**:
- File: `runs/detect/train/weights/best_yolo_int8.dlc`
- Size: ~2-3 MB (vs 10 MB for FP32)
- Optimized for NPU execution

**Quantization time**: ~5-15 minutes depending on calibration dataset size

---

## Step 4: Verify Quantized Model

Check the quantized DLC:

```bash
snpe-dlc-info --input_dlc runs/detect/train/weights/best_yolo_int8.dlc
```

Expected:
- Layer types: Mostly INT8 quantized
- Encoding: Fixed point / INT8
- Runtime: HTP (Hexagon) compatible

---

## Step 5: Test DLC on Device (QIDK)

### 5.1 Push DLC to Device

```bash
# Create directory on device
adb shell mkdir -p /data/local/tmp/yolo_npu

# Push DLC
adb push runs/detect/train/weights/best_yolo_int8.dlc /data/local/tmp/yolo_npu/

# Push test image (preprocessed to .raw format - see note below)
adb push test_image_1024x1024.raw /data/local/tmp/yolo_npu/
```

**Note**: SNPE runtime expects raw binary input, not JPEG. You need to preprocess:

```python
# Create raw input file from image
import cv2
import numpy as np

img = cv2.imread('/home/harshyy/Desktop/20250103_104457.jpg')
img = cv2.resize(img, (1024, 1024))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  # CHW
img.tofile('test_image_1024x1024.raw')
```

### 5.2 Run on NPU

```bash
# SSH or adb shell into device
adb shell

# Navigate to SNPE directory on device
cd /data/local/tmp/yolo_npu

# Run inference
snpe-net-run \
  --container best_yolo_int8.dlc \
  --input_raw test_image_1024x1024.raw \
  --use_dsp \
  --output_dir output/
```

**Runtime flags**:
- `--use_dsp`: Use NPU/Hexagon DSP
- `--use_gpu`: Use GPU (alternative)
- `--use_cpu`: Use CPU (fallback)

**Output**: Raw tensor file `output/Result_0/output0.raw` (shape [1, 9, 21504])

### 5.3 Retrieve and Decode Results

```bash
# Pull output from device
adb pull /data/local/tmp/yolo_npu/output/Result_0/output0.raw .

# Decode on host PC
python decode_npu_output.py output0.raw
```

---

## Step 6: Create NPU Output Decoder Script

Save this as `decode_npu_output.py`:

```python
#!/usr/bin/env python3
import sys
import numpy as np
from validate_onnx_for_npu import postprocess_outputs, visualize_detections, CLASS_NAMES
import cv2

def decode_npu_output(raw_file, image_path, output_viz='npu_result.jpg'):
    # Load raw output tensor
    raw_output = np.fromfile(raw_file, dtype=np.float32)
    raw_output = raw_output.reshape(1, 9, 21504)
    
    # Load original image
    img = cv2.imread(image_path)
    orig_shape = img.shape[:2]
    
    # Postprocess (same as ONNX validation)
    boxes, scores, class_ids = postprocess_outputs(
        raw_output, orig_shape, 
        conf_threshold=0.25, 
        iou_threshold=0.45, 
        img_size=1024
    )
    
    print(f"NPU Detections: {len(boxes)}")
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        print(f"  {i+1}. {CLASS_NAMES[int(cls_id)]} ({score:.3f}) @ [{x1},{y1},{x2},{y2}]")
    
    # Visualize
    visualize_detections(img, boxes, scores, class_ids, output_viz)
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python decode_npu_output.py <output0.raw> <original_image.jpg>")
        sys.exit(1)
    
    decode_npu_output(sys.argv[1], sys.argv[2])
```

Usage:
```bash
python decode_npu_output.py output0.raw /home/harshyy/Desktop/20250103_104457.jpg
```

---

## Performance Expectations

### FP32 DLC (CPU/GPU):
- Inference: ~200-300 ms
- Accuracy: Same as ONNX (11 detections)

### INT8 DLC (NPU):
- Inference: ~50-100 ms (2-4x faster)
- Accuracy: ~95-99% of FP32 (might get 10-12 detections vs 11)
- Power: ~50% of FP32

---

## Troubleshooting

### Issue: "Unsupported layer" during conversion
**Solution**: 
- Simplify ONNX further
- Check SNPE version supports all ops
- Use CPU fallback for specific layers

### Issue: Quantization accuracy loss
**Solution**:
- Increase calibration dataset (200+ images)
- Use per-channel quantization: `--use_per_channel_quantization`
- Try mixed precision: Keep some layers in FP16

### Issue: NPU execution fails
**Solution**:
- Check device has NPU support: `snpe-platform-validator`
- Try GPU first: `--use_gpu`
- Verify input tensor format (CHW, float32, normalized)

---

## Summary

✅ **Files Created**:
- `runs/detect/train/weights/best_simplified.onnx` - Optimized ONNX
- `calibration_list.txt` - 150 calibration images
- `validate_onnx_for_npu.py` - ONNX validation script
- `onnx_npu_validation.jpg` - Validation visualization
- `onnx_npu_validation.json` - Validation results

📋 **Next Steps**:
1. Set up SNPE/QNN SDK environment
2. Run Step 1: ONNX → DLC (FP32)
3. Run Step 2: Inspect DLC
4. Run Step 3: Quantize to INT8
5. Run Step 4-6: Deploy and test on QIDK

🎯 **Expected Result**:
- INT8 DLC running on NPU at ~50-100ms
- Same detection quality (9-11 objects)
- Ready for production deployment

---

**Questions or Issues?**
- Check SNPE SDK documentation for your specific version
- Ensure calibration images are representative of real deployment
- Compare NPU outputs with ONNX validation results
