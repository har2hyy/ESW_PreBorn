# Integrated Pipeline - TFLite Version

## Overview

This pipeline integrates YOLOv11 TFLite worker detection with Depth Anything V2 depth estimation for comprehensive spatial analysis.

## Current Status

✅ **Working**: TFLite FLOAT32 model  
⚠️  **Issue**: TFLite INT8 model has quantization problems

### INT8 Quantization Issue

The INT8 quantized model (`best_int8.tflite`) has a critical bug where:
- **Bbox channels (0-3)**: Properly quantized ✓
- **Class channels (4-8)**: All stuck at zero_point value (5) ✗

This results in:
- All class predictions dequantize to 0
- After sigmoid(0) = 0.5, all confidences are below detection threshold
- **Result**: No detections

**Diagnosis performed:**
```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO
conda run -n pipeline python -c "
# Analysis showed:
# - 10,700/18,900 values (56%) at zero_point (normal for sparse output)
# - BUT all class prediction channels uniformly at zero_point (abnormal)
# - Bbox channels have proper variation
"
```

**Solution**: Use FLOAT32 model until INT8 export is fixed.

## File: `integrated_pipeline_tflite_int8.py`

Despite the name, this currently uses **FLOAT32** model due to INT8 issues.

### Features

1. **Object Detection**: YOLOv11 TFLite (FLOAT32)
   - 12 detections in test image (11 workers, 1 truck)
   - Confidence threshold: 0.51
   - NMS threshold: 0.45

2. **Depth Estimation**: Depth Anything V2
   - Encoder: vitb
   - Output: Normalized depth map (0-255)
   - Device: CUDA (GPU accelerated)

3. **Distance Calculation**:
   - Euclidean distances between object centers
   - Depth differences
   - Scaled depth for 3D spatial awareness

4. **Visualizations**:
   - YOLO detections with bounding boxes
   - Colored depth map
   - Combined analysis view
   - Distance matrix heatmap
   - JSON report with all metrics

### Performance

**Test Image**: `/home/harshyy/Desktop/20250103_104457.jpg` (1920x1080)

| Component | Time (ms) | Device |
|-----------|-----------|--------|
| YOLO TFLite FLOAT32 | ~58ms | CPU |
| Depth Anything V2 | ~2,290ms | CUDA |
| **Total** | ~2,350ms | Mixed |

### Model Comparison

| Model | Size | Inference | Detections | Status |
|-------|------|-----------|------------|--------|
| best_int8.tflite | 2.8 MB | ~47ms | 0 ❌ | Broken quantization |
| best_float16.tflite | 5.1 MB | ~55ms | Not tested | Should work |
| **best_float32.tflite** | 10 MB | ~58ms | 12 ✅ | **Working** |
| best.pt (PyTorch) | ~6 MB | ~30ms | 6-12 ✅ | Working (GPU) |

## Usage

### Command Line

```bash
# Activate environment
conda activate pipeline

# Run pipeline
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/Pipeline
python integrated_pipeline_tflite_int8.py
```

### Configuration

Edit the `main()` function to customize:

```python
# Model paths
YOLO_MODEL = '../YOLO/runs/detect/train/weights/best_saved_model/best_float32.tflite'
YOLO_CLASSES = '../YOLO/classes.txt'

# Input/output
TEST_IMAGE = '/path/to/your/image.jpg'
OUTPUT_DIR = '/path/to/output/directory'

# Pipeline settings
pipeline = IntegratedPipelineTFLiteINT8(
    yolo_model_path=YOLO_MODEL,
    yolo_classes_path=YOLO_CLASSES,
    depth_encoder='vitb',          # 'vits', 'vitb', 'vitl', or 'vitg'
    yolo_conf_threshold=0.51,      # Detection confidence threshold
    yolo_nms_threshold=0.45,       # Non-maximum suppression threshold  
    depth_scale_factor=3.0         # Depth scaling for distance calculations
)
```

## Output Files

For input `image.jpg`, generates:

1. **`image_yolo_detections_int8.jpg`**: Annotated image with bounding boxes
2. **`image_depth_map_int8.png`**: Colored depth visualization
3. **`image_combined_analysis_int8.jpg`**: Side-by-side YOLO + depth
4. **`image_distance_matrix_int8.png`**: Heatmap of pairwise distances
5. **`image_analysis_report_int8.json`**: Complete numerical data

### Sample Output (Test Image)

```
Detected Objects:
  1. worker (confidence: 0.58, avg depth: 100.2)
  2. worker (confidence: 0.58, avg depth: 105.7)
  ...
  12. worker (confidence: 0.51, avg depth: 129.7)

Closest objects:
  1. worker <-> worker: 21.0px apart, depth diff: 4.6
  2. worker <-> worker: 22.0px apart, depth diff: 0.1
  3. worker <-> worker: 28.0px apart, depth diff: 5.4
```

## Fixing the INT8 Model

To properly export INT8 model:

1. **Prepare calibration dataset** (required for INT8 quantization)
2. **Re-export with calibration**:
   ```python
   from ultralytics import YOLO
   
   model = YOLO('best.pt')
   model.export(
       format='tflite',
       int8=True,
       data='construction_data.yaml',  # Calibration data
       imgsz=320
   )
   ```
3. **Verify output** with test inference

## Comparison with Other Pipelines

| Pipeline | Model Format | Advantages | Disadvantages |
|----------|--------------|------------|---------------|
| `integrated_pipeline_pytorch.py` | PyTorch (.pt) | Fastest GPU, best accuracy | Requires PyTorch/CUDA |
| `integrated_detection_depth_pipeline.py` | TFLite FLOAT32 | Tested, reliable | Larger size (10MB) |
| **`integrated_pipeline_tflite_int8.py`** | TFLite FLOAT32 | CPU-friendly, portable | Same as above (INT8 broken) |

## Dependencies

```bash
conda activate pipeline
# Already installed:
# - tensorflow
# - torch (for Depth Anything V2)
# - opencv-python
# - numpy
# - matplotlib
```

## Troubleshooting

**Q: No detections found**  
A: Check if using INT8 model - switch to FLOAT32

**Q: CUDA out of memory**  
A: Use smaller depth encoder: `depth_encoder='vits'`

**Q: Slow inference**  
A: Normal on CPU. YOLO ~60ms, Depth ~2.3s per image.

## Future Work

- [ ] Fix INT8 quantization with proper calibration dataset
- [ ] Add FLOAT16 model testing
- [ ] Optimize depth inference (try vits encoder)
- [ ] Add batch processing support
- [ ] Implement real-time video processing

## Author

Pipeline Integration System  
Date: November 2025
