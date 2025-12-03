# YOLO11 300-Image Model Pipeline

Complete pipeline for running the YOLO11 model trained on 300 manually annotated images, with both PyTorch and INT8 TFLite versions optimized for QIDK NPU deployment.

## 📋 Overview

This pipeline provides:

1. **PyTorch (.pt) Inference Pipeline**
   - Uses `models/pytorch/best.pt` (trained on 300 images)
   - Outputs to `pipeline_output_300pt/`
   - Full precision FP32 detection

2. **INT8 TFLite Quantized Pipeline**
   - Quantized INT8 model for QIDK NPU
   - Uses 150 calibration images from `data_1-300`
   - Outputs to `pipeline_output_300INT8tflite/`
   - Optimized for mobile/edge deployment

## 🚀 Quick Start

### Run Complete Pipeline (All Steps)

```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO
python3 run_complete_pipeline.py
```

This will:
1. Run PyTorch model on test images
2. Export PyTorch → INT8 TFLite (with calibration)
3. Run INT8 TFLite model on test images

### Run Individual Pipelines

**PyTorch Pipeline:**
```bash
python3 run_pipeline_300pt.py
```

**INT8 TFLite Export:**
```bash
python3 export_int8_300pt.py
```

**INT8 TFLite Pipeline:**
```bash
python3 run_pipeline_300INT8tflite.py
```

## 📁 Output Structure

```
YOLO/
├── pipeline_output_300pt/              # PyTorch model outputs
│   ├── *_detected.jpg                  # Annotated images
│   ├── *_detections.json               # Detection data per image
│   └── pipeline_summary.json           # Overall summary
│
├── pipeline_output_300INT8tflite/      # INT8 TFLite outputs
│   ├── *_detected.jpg                  # Annotated images
│   ├── *_detections.json               # Detection data per image
│   └── pipeline_summary.json           # Overall summary
│
└── models/tflite/
    └── best_yolo11_300_int8.tflite    # Quantized model (~3 MB)
```

## 📊 Model Information

### Training Details
- **Dataset**: 300 manually annotated construction site images
- **Classes**: worker, truck, bike, bulldozer, car
- **Architecture**: YOLOv11n (nano variant)
- **Input Size**: 1024×1024
- **Epochs**: 150-200 with early stopping

### Model Variants

| Variant | Format | Precision | Size | Use Case | Output Folder |
|---------|--------|-----------|------|----------|---------------|
| **PyTorch** | .pt | FP32 | ~11 MB | Training, PC inference | `pipeline_output_300pt/` |
| **INT8 TFLite** | .tflite | INT8 | ~3 MB | QIDK NPU, mobile | `pipeline_output_300INT8tflite/` |

## 🔧 Detailed Usage

### 1. PyTorch Pipeline (`run_pipeline_300pt.py`)

Runs the original PyTorch model on test images.

**Features:**
- Full FP32 precision
- Direct Ultralytics YOLO inference
- Confidence threshold: 0.25
- NMS IoU threshold: 0.45

**Output:**
- Annotated images with bounding boxes
- JSON files with detection coordinates and confidence
- Summary JSON with all results

**Example output JSON:**
```json
{
  "image": "test_image.jpg",
  "model": "best.pt (300 images trained)",
  "total_detections": 3,
  "detections": [
    {
      "class": "worker",
      "confidence": 0.847,
      "bbox": [245.2, 120.5, 310.8, 280.3],
      "center": [278.0, 200.4]
    }
  ]
}
```

### 2. INT8 Export (`export_int8_300pt.py`)

Exports PyTorch model to INT8 quantized TFLite.

**Process:**
1. Loads `models/pytorch/best.pt`
2. Exports to TFLite FP32 (intermediate)
3. Quantizes to INT8 using calibration data
4. Saves to `models/tflite/best_yolo11_300_int8.tflite`

**Calibration:**
- Uses 150 images from `data_1-300` or `data/train/images`
- Representative dataset for accurate quantization
- INT8 for both inputs and outputs (`uint8`)

**Verification:**
- Checks input/output dtypes
- Confirms INT8 quantization
- Reports model size

### 3. INT8 TFLite Pipeline (`run_pipeline_300INT8tflite.py`)

Runs the quantized INT8 model on test images.

**Features:**
- Custom INT8 inference (no Ultralytics)
- Preprocessing: uint8 [0, 255]
- Post-processing: YOLO output parsing + NMS
- Bbox visualization

**Performance:**
- Faster inference on NPU/mobile
- Lower memory footprint (~3 MB vs 11 MB)
- Minimal accuracy loss (~1-2% typically)

**Output:**
- Same format as PyTorch pipeline
- Includes quantization metadata in JSON

## 🎯 Output Comparison

Both pipelines produce:

1. **Visual Outputs** (`*_detected.jpg`):
   - Bounding boxes drawn on images
   - Class labels + confidence scores
   - Color-coded by class

2. **Detection Data** (`*_detections.json`):
   - Per-image detection results
   - Bounding box coordinates (x1, y1, x2, y2)
   - Center points for distance calculations
   - Confidence scores

3. **Summary** (`pipeline_summary.json`):
   - Model metadata
   - Training dataset info
   - Quantization details (INT8 only)
   - All results aggregated

## 📈 Expected Results

### Detection Quality

| Metric | PyTorch (FP32) | INT8 TFLite |
|--------|----------------|-------------|
| **Accuracy** | 100% (baseline) | ~98-99% |
| **Speed (QIDK NPU)** | N/A (PC only) | 20-80ms |
| **Model Size** | 11 MB | ~3 MB |
| **Memory Usage** | Higher | Lower |

### Typical Detection Thresholds

- **Confidence**: 0.25 (filters weak detections)
- **NMS IoU**: 0.45 (removes overlapping boxes)

## 🛠️ Troubleshooting

### INT8 Export Fails

**Symptom:** `export_int8_300pt.py` crashes during quantization

**Solutions:**
1. Check calibration images exist:
   ```bash
   ls data_1-300/*.jpg | wc -l  # Should show images
   ```

2. Reduce calibration dataset size (edit `num_images=150` to `50`)

3. Use existing INT8 model:
   ```bash
   cp models/tflite/best_yolo_int8.tflite models/tflite/best_yolo11_300_int8.tflite
   ```

### No Test Images Found

**Symptom:** `No test images found in test_images`

**Solution:**
```bash
# Add test images to YOLO/test_images/
cp /path/to/images/*.jpg test_images/
```

### TensorFlow Import Error

**Symptom:** `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**
```bash
# Activate environment with TensorFlow
conda activate pipeline  # or yolo11, or install tensorflow
pip install tensorflow
```

### Model Not Found

**Symptom:** `PyTorch model not found: models/pytorch/best.pt`

**Solution:**
```bash
# Ensure you're in YOLO directory
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO

# Check model exists
ls models/pytorch/best.pt
```

## 🔬 Technical Details

### PyTorch Model

- **Framework**: Ultralytics YOLO v11
- **Backend**: PyTorch 2.x
- **Inference**: CPU/CUDA
- **Preprocessing**: Automatic (via Ultralytics)
- **Postprocessing**: Built-in NMS

### INT8 TFLite Model

- **Framework**: TensorFlow Lite
- **Quantization**: Full INT8 (inputs + weights + outputs)
- **Calibration**: Post-training quantization with representative dataset
- **Inference**: TFLite interpreter (CPU/GPU/NPU)
- **Preprocessing**: Manual uint8 conversion
- **Postprocessing**: Custom YOLO parser + NMS

### YOLO Output Format

Raw output shape: `[1, 9, 21504]`
- **Batch**: 1
- **Channels**: 9 (4 bbox + 5 classes)
- **Anchors**: 21504 (across 3 detection scales)

Detection decoding:
1. Transpose to `[21504, 9]`
2. Extract bbox coords: `[x_center, y_center, width, height]`
3. Extract class scores: `[worker, truck, bike, bulldozer, car]`
4. Filter by confidence threshold
5. Convert bbox format: center → corners
6. Apply Non-Maximum Suppression

## 📦 Deployment

### For QIDK NPU

1. **Get INT8 Model:**
   ```bash
   # After running pipeline
   ls models/tflite/best_yolo11_300_int8.tflite
   ```

2. **Transfer to Device:**
   ```bash
   adb push models/tflite/best_yolo11_300_int8.tflite /data/local/tmp/
   ```

3. **Run Inference:**
   - Use TFLite NPU delegate (Hexagon)
   - Or convert to QNN/DLC format for native NPU

### For Android App

1. Copy INT8 model to app assets:
   ```bash
   cp models/tflite/best_yolo11_300_int8.tflite \
      ../VisionSolution1-ObjectDetection-YoloNas/app/src/main/assets/
   ```

2. Update app to load INT8 model instead of FP32

3. Rebuild and deploy

## 📝 Files Created

| File | Purpose |
|------|---------|
| `run_pipeline_300pt.py` | PyTorch inference pipeline |
| `export_int8_300pt.py` | INT8 TFLite export script |
| `run_pipeline_300INT8tflite.py` | INT8 TFLite inference pipeline |
| `run_complete_pipeline.py` | Master script (runs all 3) |
| `PIPELINE_300_README.md` | This documentation |

## 🎓 Next Steps

1. **Compare Outputs:**
   ```bash
   # View annotated images side-by-side
   eog pipeline_output_300pt/*_detected.jpg &
   eog pipeline_output_300INT8tflite/*_detected.jpg &
   ```

2. **Check Detection Differences:**
   ```bash
   # Compare JSON outputs
   diff pipeline_output_300pt/pipeline_summary.json \
        pipeline_output_300INT8tflite/pipeline_summary.json
   ```

3. **Deploy to QIDK:**
   - See `docs/QIDK_NPU_TESTING_GUIDE.md`
   - Use INT8 model for best NPU performance

4. **Integrate with Depth Pipeline:**
   - Use detection outputs with `../Pipeline/integrated_pipeline_*.py`
   - Combine detection + depth for 3D localization

## 📞 Support

- **Training Issues**: See `scripts/training/train.py`
- **Conversion Issues**: See `docs/ALL_DLC_CONVERSION_PATHS.md`
- **NPU Deployment**: See `docs/QIDK_NPU_TESTING_GUIDE.md`
- **TFLite Issues**: See `docs/FIX_TFLITE_NPU.md`

---

**Created**: December 2025  
**Model**: YOLO11n trained on 300 construction site images  
**Purpose**: Construction worker safety detection
