# INT8 Model Export Issue - Root Cause

## 🚨 Problem

The INT8 TFLite model is **fundamentally broken**. After applying all post-processing fixes, it now detects objects, but:

- **All confidences are exactly 0.500**
- **All detections are in a grid pattern**
- **21,504 detections per image** (every single anchor)
- **Zero input produces same output as real images**

## 🔍 Root Cause: Model Quantization Failure

The model isn't actually learning - it's outputting baseline values:

```python
# With ZERO input:
- Raw output: mostly uint8 value 8 (the zero_point)
- After dequantization: ~0.0
- After sigmoid: 0.500 (sigmoid(0) = 0.5)
- Class scores: ALL exactly 0.500
```

This means the INT8 model is **not performing inference** - it's just outputting near-zero values for all channels.

## ❌ Current INT8 Model Issues

### Issue #1: No Calibration Dataset Used
The existing `export_int8_300pt.py` script was **created but never run**. The current `best_yolo_int8.tflite` was likely exported without proper quantization.

### Issue #2: ONNX → TFLite Conversion Path
Looking at the workspace, the INT8 model might have been:
1. PyTorch (.pt) → ONNX (.onnx) → TFLite (wrong path)
2. Should be: PyTorch → TFLite directly with Ultralytics

### Issue #3: Model Format Mismatch
YOLO11 uses specific activation functions:
- **Bbox coords**: Sigmoid
- **Class scores**: Sigmoid
- **Objectness**: Usually included in outputs

The INT8 model may have been quantized with wrong activation assumptions.

## ✅ Solution: Re-export INT8 Model Properly

### Option 1: Use Ultralytics Export (RECOMMENDED)

```python
from ultralytics import YOLO
import numpy as np
from pathlib import Path

# Load PyTorch model
model = YOLO('models/pytorch/best.pt')

# Prepare calibration data (150 images)
calib_dir = Path('data_1-300')
calib_images = list(calib_dir.glob('*.jpg'))[:150]

# Export to TFLite INT8 with calibration
model.export(
    format='tflite',
    int8=True,
    data='data_1-300',  # Use training data for calibration
    imgsz=1024,
    optimize=True
)
```

### Option 2: Fix TensorFlow Export Script

The current `export_int8_300pt.py` needs to:
1. Load PyTorch model
2. Convert to TensorFlow SavedModel
3. Apply INT8 quantization with **representative dataset**
4. Verify outputs are not all zeros

### Option 3: Check Existing Model

```bash
# Verify if current model was exported properly
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO
ls -lh models/tflite/

# Check if there are other TFLite models
find . -name "*.tflite" -ls
```

## 🔬 Diagnostic Test Results

### Test 1: Zero Input
```
Input: all zeros (1024x1024x3)
Output: uint8 [8-255]
After sigmoid: all class scores = 0.500
Detections: 21,504 (100% of anchors)
```

### Test 2: Real Image
```
Input: actual construction site image
Output: Same as zero input!
Class scores: all 0.500
Detections: 8 (after NMS on 21,504 grid boxes)
```

**Conclusion**: Model is not processing input images at all.

## 📊 Comparison

| Metric | PyTorch (.pt) | INT8 TFLite (broken) | Expected INT8 |
|--------|--------------|---------------------|---------------|
| Detections | 126 | 8 (grid) | ~120-124 |
| Confidence range | 0.26-0.88 | All 0.500 | 0.26-0.86 |
| Model behavior | ✅ Normal | ❌ Dead | ✅ Should work |
| Zero input | No detections | 21,504 detections | No detections |

## 🛠️ Next Steps

### Step 1: Find Working Model Source

Check if ONNX model works correctly:
```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO
python3 scripts/testing/test_onnx_on_pc.py
```

### Step 2: Re-export INT8 from Working Source

Use the best available format (.pt or .onnx) to export INT8 properly.

### Step 3: Verify New INT8 Model

```python
# Test with zero input - should give NO detections
# Test with real image - should give ~120 detections
```

## 🎯 Expected Behavior After Fix

### With Zero Input:
- Output: varied values, not all zero_point
- Class scores: low values (< 0.25)
- Detections: 0

### With Real Image:
- Output: varied activations
- Class scores: 0.26-0.86 range
- Detections: 120-124 objects
- Confidence distribution similar to PyTorch

## 📝 Files to Check

1. **models/tflite/best_yolo_int8.tflite** - Current broken model (2.82 MB)
2. **models/onnx/** - Check if ONNX model exists and works
3. **export_int8_300pt.py** - Script to export fresh INT8 model (not run yet)
4. **yolo11n.onnx** - Root directory has this, check if it works

## ⚠️ Important Notes

1. **INT8 quantization is lossy** but shouldn't be THIS broken
2. **Typical INT8 loss**: 1-2% mAP drop, not 100% failure
3. **All 0.500 scores** = model is dead, not just inaccurate
4. **Need calibration dataset** for representative value ranges

## 🔍 How to Detect Dead Quantized Models

Signs of a broken INT8 model:
- ✅ Loads without errors
- ✅ Runs inference without crashing
- ❌ All outputs near zero_point value
- ❌ After activation, all values same
- ❌ Zero input gives same result as real image
- ❌ Every anchor passes confidence threshold

---

**Status**: Post-processing is now CORRECT, but model itself is broken. Need to re-export INT8 model with proper quantization calibration.

**Urgency**: HIGH - Current INT8 model is unusable for QIDK deployment
