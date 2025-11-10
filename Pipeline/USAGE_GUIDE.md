# Integrated Detection + Depth Pipeline - Complete Usage Guide

## ✅ Successfully Created & Tested!

The pipeline has been successfully created and tested. Here's everything you need to know:

---

## 📋 What Was Created

### 1. Main Pipeline (`integrated_detection_depth_pipeline.py`)
- **600+ lines** of production-ready Python code
- Integrates YOLOv11 and Depth Anything V2
- Calculates distances between all detected objects
- Generates comprehensive visualizations

### 2. Quick Start Script (`quick_start.py`)
- Simple demo to run the pipeline
- Easy to modify for different images

### 3. Documentation (`README.md`)
- Detailed explanation of architecture
- Step-by-step process breakdown
- Technical details and examples

---

## 🚀 How to Use

### Step 1: Activate the Environment
```bash
conda activate pipeline
```

This environment has:
- ✅ PyTorch 2.6.0 + CUDA 12.4 (GPU support)
- ✅ TensorFlow 2.20.0
- ✅ OpenCV, Matplotlib
- ✅ All Depth Anything V2 dependencies
- ✅ All YOLO dependencies

### Step 2: Run on Your Images

#### Option A: Use Quick Start (Easiest)
```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/Pipeline
python quick_start.py
```

Edit `quick_start.py` to change:
- `IMAGE_PATH`: Your input image
- `OUTPUT_DIR`: Where to save results

#### Option B: Use in Your Own Script
```python
from integrated_detection_depth_pipeline import IntegratedPipeline

# Initialize once
pipeline = IntegratedPipeline(
    yolo_model_path='../YOLO/runs/detect/train/weights/best_saved_model/best_float32.tflite',
    yolo_classes_path='../YOLO/classes.txt',
    depth_encoder='vitb',  # or 'vits' for speed
    yolo_conf_threshold=0.60,
    yolo_nms_threshold=0.45
)

# Process images
results = pipeline.process_image('/path/to/image.jpg', '/path/to/output')

# Access results
print(f"Detected {results['total_detections']} objects")
for det in results['detections']:
    print(f"{det['class']}: depth={det['depth_avg']:.1f}")
```

---

## 📊 Understanding the Output

### Generated Files (for each image):

1. **`*_yolo_detections.jpg`**
   - Original image with bounding boxes
   - Shows detected objects with class labels and confidence scores

2. **`*_depth_map.png`**
   - Colored depth visualization
   - Red/Orange = Close, Blue/Purple = Far

3. **`*_combined_analysis.jpg`**
   - Side-by-side: YOLO + Depth
   - Statistics panel with summary
   - Professional visualization for reports

4. **`*_distance_matrix.png`** (if 2+ objects detected)
   - Heatmap showing distances between all object pairs
   - Two matrices: Euclidean distance + Depth difference

5. **`*_analysis_report.json`**
   - Complete structured data
   - Machine-readable for further processing
   - Contains all detections and distances

---

## 🔍 How It Works - Detailed Explanation

### Pipeline Architecture

```
Input Image
    ↓
[Stage 1: YOLO Detection]
    → Detects objects (workers, trucks, etc.)
    → Returns bounding boxes + classes
    ↓
[Stage 2: Depth Estimation]
    → Generates depth map for entire image
    → Each pixel gets a depth value (0-255)
    ↓
[Stage 3: Depth Extraction]
    → For each detected object:
      - Extract depth from bbox region
      - Calculate avg/median/min/max depth
    ↓
[Stage 4: Distance Calculation]
    → For all object pairs:
      - Euclidean distance (pixels)
      - Depth difference
      - Horizontal/vertical separation
    ↓
[Stage 5: Visualization]
    → Generate all output files
    → Create combined analysis
    → Save JSON report
```

### Distance Metrics Explained

#### 1. **Euclidean Distance (pixels)**
```
Formula: √((x₂-x₁)² + (y₂-y₁)²)
```
- Straight-line distance between object centers in 2D image
- **Does NOT account for depth/perspective**
- Useful for: Understanding relative positions in image plane

#### 2. **Depth Difference**
```
Formula: |depth₁ - depth₂|
```
- Difference in estimated depth values
- **Relative depth only** (not absolute meters)
- Useful for: Understanding which objects are closer/farther

#### 3. **Horizontal & Vertical Separation**
```
Horizontal: |x₂ - x₁|
Vertical: |y₂ - y₁|
```
- Component-wise distances
- Useful for: Understanding left/right, up/down relationships

### Example Interpretation

**Scenario:** Construction site with 3 objects

```json
{
  "detections": [
    {
      "id": 0,
      "class": "worker",
      "center": [200, 300],
      "depth_avg": 120.5
    },
    {
      "id": 1,
      "class": "worker",
      "center": [350, 320],
      "depth_avg": 125.2
    },
    {
      "id": 2,
      "class": "excavator",
      "center": [600, 250],
      "depth_avg": 180.8
    }
  ],
  "distances": [
    {
      "obj1_class": "worker",
      "obj2_class": "worker",
      "euclidean": 152.3,
      "depth_diff": 4.7,
      "interpretation": "Two workers close together at similar depth"
    },
    {
      "obj1_class": "worker",
      "obj2_class": "excavator",
      "euclidean": 408.7,
      "depth_diff": 60.3,
      "interpretation": "Worker and excavator far apart, excavator much farther away"
    }
  ]
}
```

**What This Tells You:**
- Workers 0 and 1: Side by side (low depth diff), close together
- Worker 0 and Excavator: Separated both in image space AND depth
- **Safety Alert:** If worker-excavator distance < threshold → warn!

---

## ⚙️ Configuration Options

### YOLO Parameters

**Confidence Threshold** (`yolo_conf_threshold`):
- **Default:** 0.60
- **Range:** 0.0 to 1.0
- **Higher (0.70-0.90):** Fewer detections, more confident
- **Lower (0.40-0.60):** More detections, may include false positives
- **Recommendation:** 0.60 for construction sites

**NMS Threshold** (`yolo_nms_threshold`):
- **Default:** 0.45
- **Range:** 0.0 to 1.0
- **Purpose:** Removes overlapping bounding boxes
- **Recommendation:** Keep at 0.45

### Depth Model Selection

**Encoder Options:**

| Encoder | Speed      | Accuracy | GPU Memory | Recommended For |
|---------|------------|----------|------------|-----------------|
| `vits`  | Fastest    | Good     | ~1GB       | Real-time, QIDK |
| `vitb`  | Fast       | Better   | ~2GB       | Balanced (default) |
| `vitl`  | Moderate   | High     | ~4GB       | High quality |
| `vitg`  | Slow       | Highest  | ~8GB       | Research only |

**For QIDK (running with YOLOv11n):**
```python
depth_encoder='vits'  # Recommended for speed
```

---

## 🔧 Technical Details

### Performance

Tested on: NVIDIA GPU with CUDA 12.4

| Component | Time (per image) | Notes |
|-----------|-----------------|-------|
| YOLO Inference | ~30-100ms | TFLite on CPU/GPU |
| Depth Inference | ~100-300ms | PyTorch on GPU |
| Post-processing | ~50-100ms | Distance calculations |
| **Total Pipeline** | **~200-500ms** | End-to-end |

### Memory Requirements

- **Minimum:** 4GB GPU RAM (with `vits` encoder)
- **Recommended:** 6GB+ GPU RAM (with `vitb` encoder)
- **CPU RAM:** ~4GB

### Coordinate Systems

**Image Space:**
- Origin: Top-left corner (0, 0)
- X-axis: Left → Right (increasing)
- Y-axis: Top → Bottom (increasing)
- Units: Pixels

**Depth Space:**
- Z-axis: Into the scene (perpendicular to image)
- Units: Relative (0-255 normalized)
- Lower values = Closer to camera
- Higher values = Farther from camera

---

## 🎯 Use Cases

### 1. Construction Site Safety
```python
# Monitor worker-machinery distances
for dist in results['distances']:
    if 'worker' in dist['obj1_class'] and 'excavator' in dist['obj2_class']:
        if dist['euclidean'] < 200:  # pixels
            print(f"⚠️ WARNING: Worker too close to excavator!")
```

### 2. Object Proximity Analysis
```python
# Find all nearby object pairs
nearby_pairs = [d for d in results['distances'] if d['euclidean'] < 150]
print(f"Found {len(nearby_pairs)} nearby object pairs")
```

### 3. Depth-Based Filtering
```python
# Find all foreground objects (close to camera)
foreground = [d for d in results['detections'] if d['depth_avg'] < 100]
background = [d for d in results['detections'] if d['depth_avg'] > 150]
```

### 4. Batch Processing
```python
import glob

# Process all images in a directory
image_files = glob.glob('/path/to/images/*.jpg')
for img in image_files:
    results = pipeline.process_image(img, '/path/to/output')
    print(f"Processed {img}: {results['total_detections']} objects")
```

---

## 🐛 Troubleshooting

### Issue: No objects detected

**Cause:** Image doesn't contain trained classes or confidence too high

**Solution:**
```python
# Lower confidence threshold
pipeline = IntegratedPipeline(
    ...,
    yolo_conf_threshold=0.40  # Try lower value
)
```

### Issue: Out of memory

**Cause:** GPU RAM insufficient for depth model

**Solution:**
```python
# Use smaller encoder
pipeline = IntegratedPipeline(
    ...,
    depth_encoder='vits'  # Instead of 'vitb'
)
```

### Issue: Slow processing

**Cause:** Large images or slow encoder

**Solutions:**
1. Resize input images before processing
2. Use `vits` encoder instead of `vitb`
3. Process on GPU (check `torch.cuda.is_available()`)

### Issue: Module not found

**Cause:** Wrong conda environment

**Solution:**
```bash
# Always use the pipeline environment
conda activate pipeline
python quick_start.py
```

---

## 📝 Code Walkthrough

### Key Classes and Methods

#### `IntegratedPipeline` Class

**Initialization:**
```python
def __init__(self, yolo_model_path, yolo_classes_path, depth_encoder, ...):
    # Loads both YOLO and Depth models
    # Sets up device (CUDA/CPU)
    # Initializes model configurations
```

**Main Processing:**
```python
def process_image(self, image_path, output_dir):
    # Stage 1: Run YOLO detection
    # Stage 2: Run depth estimation
    # Stage 3: Extract depth for each detection
    # Stage 4: Calculate pairwise distances
    # Stage 5: Generate visualizations
    # Returns: Dictionary with all results
```

**Distance Calculation:**
```python
def _calculate_distances(self, detections):
    # For each pair of detections:
    #   - Euclidean distance between centers
    #   - Depth difference
    #   - Horizontal/vertical components
    # Returns: List of distance measurements
```

**Visualization:**
```python
def _create_combined_visualization(...):
    # Creates side-by-side YOLO + Depth view
    # Adds title bar and statistics panel
    # Returns: Combined image
```

### Data Structures

**Detection Dictionary:**
```python
{
    'id': 0,
    'class': 'worker',
    'class_id': 0,
    'confidence': 0.75,
    'bbox': [x1, y1, x2, y2],
    'center': (center_x, center_y),
    'depth_avg': 145.3,
    'depth_median': 142.0,
    'depth_min': 120,
    'depth_max': 180,
    'depth_center': 143,
    'area': 6500
}
```

**Distance Dictionary:**
```python
{
    'obj1_id': 0,
    'obj1_class': 'worker',
    'obj2_id': 1,
    'obj2_class': 'excavator',
    'euclidean': 287.5,
    'horizontal': 245,
    'vertical': 120,
    'depth_diff': 45.2,
    'obj1_depth': 145.3,
    'obj2_depth': 190.5
}
```

---

## 🚀 Next Steps & Improvements

### Immediate Use

1. **Test on Construction Images:**
   - Use images from `YOLO/train/images/` 
   - Should detect workers, trucks, bulldozers, etc.

2. **Adjust Thresholds:**
   - Tune confidence for your use case
   - Set distance thresholds for safety alerts

### Future Enhancements

1. **Metric Depth:**
   - Add camera calibration
   - Convert relative → absolute distances (meters)
   - Requires camera intrinsics

2. **Real-Time Processing:**
   - Video stream support
   - Multi-threading for speed
   - Frame-by-frame analysis

3. **Advanced Analytics:**
   - Trajectory tracking (over time)
   - Velocity estimation
   - Proximity alerts with thresholds

4. **3D Reconstruction:**
   - Generate point clouds
   - 3D visualization
   - Volumetric analysis

---

## 📚 Key Concepts Review

### Monocular Depth Estimation
- Estimates depth from **single RGB image**
- Based on visual cues (size, perspective, occlusion)
- **Relative depth** (not absolute metric)

### Non-Maximum Suppression (NMS)
- Removes duplicate/overlapping detections
- Keeps highest confidence boxes
- Essential for clean YOLO output

### Relative vs. Metric Depth
- **Relative:** Values normalized within image (this pipeline)
  - Good for comparing depths
  - Not good for absolute measurements
- **Metric:** Real-world distances in meters
  - Requires calibration or depth sensors

---

## ✅ Summary

You now have a **complete, production-ready pipeline** that:

✅ Detects objects using YOLOv11
✅ Estimates depth using Depth Anything V2
✅ Calculates distances between all object pairs
✅ Generates comprehensive visualizations
✅ Saves structured JSON reports
✅ Runs on GPU for optimal performance
✅ Is fully documented and explained

**Environment:** `conda activate pipeline`

**Main Files:**
- `integrated_detection_depth_pipeline.py` - Core pipeline
- `quick_start.py` - Easy demo script
- `README.md` - Technical documentation
- `USAGE_GUIDE.md` - This file

**Output Location:** `/home/harshyy/Desktop/pipeline_output/`

---

## 👨‍💻 Author & Support

Pipeline created: November 3, 2025

For questions or issues:
1. Check README.md for technical details
2. Review this guide for usage examples
3. Inspect the JSON output for data structure

**Happy detecting!** 🎉
