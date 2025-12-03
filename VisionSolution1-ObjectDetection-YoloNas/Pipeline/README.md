# Integrated Detection + Depth Pipeline

## 🎯 Overview

This pipeline combines **YOLOv11 object detection** with **Depth Anything V2 depth estimation** to provide comprehensive spatial analysis of images. It detects objects, estimates their depths, and calculates distances between all detected items.

---

## 🏗️ Architecture

```
Input Image
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 1: YOLO Object Detection                 │
│  - Detects objects in image                     │
│  - Returns: bounding boxes, classes, confidence │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 2: Depth Estimation                      │
│  - Generates depth map using Depth Anything V2  │
│  - Returns: normalized depth values (0-255)     │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 3: Depth Extraction                      │
│  - Extract depth for each detected object       │
│  - Calculate avg/median/min/max depth per bbox  │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 4: Distance Calculation                  │
│  - Compute pairwise distances between objects   │
│  - Calculate: Euclidean, depth diff, H/V sep    │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Stage 5: Visualization & Output                │
│  - Generate annotated images                    │
│  - Create distance matrices                     │
│  - Save comprehensive JSON report               │
└─────────────────────────────────────────────────┘
    ↓
Output Files
```

---

## 🔧 How It Works

### Stage 1: Object Detection (YOLOv11)

**Purpose:** Identify and localize objects in the image

**Process:**
1. Load pre-trained YOLOv11 TFLite model
2. Preprocess image (resize to 128×128, normalize to [0,1])
3. Run inference through neural network
4. Post-process outputs:
   - Extract bounding boxes (x1, y1, x2, y2)
   - Filter by confidence threshold (default: 0.60)
   - Apply Non-Maximum Suppression (NMS) to remove overlaps
5. Return detected objects with:
   - Class name (e.g., "Worker", "Excavator")
   - Confidence score (0.0 to 1.0)
   - Bounding box coordinates

**Output Example:**
```
Detection 1: Worker (0.75) at [245, 120, 310, 280]
Detection 2: Excavator (0.82) at [450, 80, 680, 350]
```

---

### Stage 2: Depth Estimation (Depth Anything V2)

**Purpose:** Estimate relative depth for every pixel in the image

**Process:**
1. Load Depth Anything V2 model (Vision Transformer encoder)
2. Preprocess image:
   - Resize to 518×518
   - Normalize according to model requirements
3. Run inference through depth network
4. Post-process depth map:
   - Raw depth values (relative, not metric)
   - Normalize to 0-255 range: `(depth - min) / (max - min) × 255`
   - Lower values = closer to camera
   - Higher values = farther from camera

**Key Points:**
- **Relative Depth:** Values are relative within the image (not absolute meters)
- **Monocular:** Uses single RGB image (no stereo/LiDAR needed)
- **Dense:** Every pixel gets a depth value

**Output Example:**
```
Depth Map: 1080×1920 array
  - Min: 0
  - Max: 255
  - Mean: 127.3
```

---

### Stage 3: Depth Extraction for Detected Objects

**Purpose:** Associate depth information with each detected object

**Process:**
1. For each bounding box from YOLO:
   ```python
   bbox_depth = depth_map[y1:y2, x1:x2]  # Extract region
   ```
2. Calculate depth statistics:
   - **Average Depth:** Mean of all pixels in bbox
   - **Median Depth:** Median value (robust to outliers)
   - **Min/Max Depth:** Range of depths within object
   - **Center Depth:** Depth at bbox center point

**Why Multiple Metrics?**
- **Average:** Overall depth of object
- **Median:** More robust if object has depth variation
- **Center:** Quick approximation (single pixel lookup)

**Output Example:**
```json
{
  "id": 0,
  "class": "Worker",
  "bbox": [245, 120, 310, 280],
  "center": [277, 200],
  "depth_avg": 145.3,
  "depth_median": 142.0,
  "depth_min": 120,
  "depth_max": 180,
  "depth_center": 143
}
```

---

### Stage 4: Distance Calculation

**Purpose:** Compute spatial relationships between all pairs of detected objects

**Distance Metrics Calculated:**

#### 1. **Euclidean Distance (pixels)**
Straight-line distance between object centers in image space:
```
distance = √((x₂ - x₁)² + (y₂ - y₁)²)
```

**Interpretation:** Pixel separation in 2D image plane

#### 2. **Depth Difference**
Difference in estimated depth values:
```
depth_diff = |depth₁ - depth₂|
```

**Interpretation:** 
- Large value → Objects at different depths (fore/background)
- Small value → Objects at similar depth

#### 3. **Horizontal/Vertical Separation**
Component-wise distances:
```
horizontal = |x₂ - x₁|
vertical = |y₂ - y₁|
```

**Use Case:** Understanding relative positioning (left/right, up/down)

**Pairwise Calculation:**
For N objects, compute all N×(N-1)/2 pairs:
```
N=3 objects → 3 pairs: (0,1), (0,2), (1,2)
N=5 objects → 10 pairs
N=10 objects → 45 pairs
```

**Output Example:**
```json
{
  "obj1_id": 0,
  "obj1_class": "Worker",
  "obj2_id": 1,
  "obj2_class": "Excavator",
  "euclidean": 287.5,
  "horizontal": 245,
  "vertical": 120,
  "depth_diff": 45.2,
  "obj1_depth": 145.3,
  "obj2_depth": 190.5
}
```

---

### Stage 5: Visualization & Reporting

**Purpose:** Generate comprehensive visual and data outputs

#### Outputs Generated:

1. **YOLO Detection Image** (`*_yolo_detections.jpg`)
   - Original image with bounding boxes
   - Class labels and confidence scores
   - Color-coded by class

2. **Depth Map** (`*_depth_map.png`)
   - Colored depth visualization
   - Spectral colormap: Red/Orange (close) → Blue/Purple (far)

3. **Combined Analysis** (`*_combined_analysis.jpg`)
   - Side-by-side YOLO + Depth
   - Statistics panel with summary
   - Title and metadata

4. **Distance Matrix** (`*_distance_matrix.png`)
   - Heatmap visualization
   - Two matrices:
     - Euclidean distances (pixels)
     - Depth differences
   - Color-coded: Green (close) → Red (far)

5. **JSON Report** (`*_analysis_report.json`)
   - Complete structured data
   - All detections with depth info
   - All pairwise distances
   - Machine-readable for further processing

---

## 📊 Understanding the Output

### Distance Interpretation

**Euclidean Distance (Pixels):**
- Measures 2D separation in image
- **Does NOT account for perspective/depth**
- Two objects can be close in pixels but far in 3D space

**Depth Difference:**
- Measures separation along camera axis
- Complements Euclidean distance
- **Relative values only** (not absolute meters)

**Combined Interpretation:**
```
High Euclidean + Low Depth Diff → Side by side at same depth
Low Euclidean + High Depth Diff → Aligned but fore/background
High Euclidean + High Depth Diff → Far apart in all dimensions
```

### Example Scenario:

```
Image: Construction site
Detections:
  1. Worker A - Center: (200, 300), Depth: 120
  2. Worker B - Center: (350, 320), Depth: 125
  3. Excavator - Center: (600, 250), Depth: 180

Distances:
  Worker A ↔ Worker B:
    - Euclidean: 152 pixels
    - Depth diff: 5
    - Interpretation: Close together, same depth plane

  Worker A ↔ Excavator:
    - Euclidean: 408 pixels
    - Depth diff: 60
    - Interpretation: Far apart, excavator much farther away

  Worker B ↔ Excavator:
    - Euclidean: 261 pixels
    - Depth diff: 55
    - Interpretation: Moderate pixel distance, significant depth separation
```

---

## 🚀 Usage

### Basic Usage:

```python
from integrated_detection_depth_pipeline import IntegratedPipeline

# Initialize pipeline
pipeline = IntegratedPipeline(
    yolo_model_path='path/to/yolo_model.tflite',
    yolo_classes_path='path/to/classes.txt',
    depth_encoder='vitb',  # or 'vits' for speed
    yolo_conf_threshold=0.60,
    yolo_nms_threshold=0.45
)

# Process an image
results = pipeline.process_image(
    image_path='/path/to/image.jpg',
    output_dir='/path/to/output'
)

# Access results
print(f"Found {results['total_detections']} objects")
for detection in results['detections']:
    print(f"{detection['class']}: depth={detection['depth_avg']:.1f}")

for distance in results['distances']:
    print(f"{distance['obj1_class']} ↔ {distance['obj2_class']}: "
          f"{distance['euclidean']:.1f}px apart")
```

### Command Line:

```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/Pipeline
conda activate depth  # or yolo11
python integrated_detection_depth_pipeline.py
```

---

## ⚙️ Configuration Options

### YOLO Parameters:

- `yolo_conf_threshold` (0.0-1.0): Minimum confidence for detection
  - **Higher** = fewer, more confident detections
  - **Lower** = more detections, some may be false positives
  - Recommended: 0.60 for construction site images

- `yolo_nms_threshold` (0.0-1.0): IoU threshold for NMS
  - Controls overlap tolerance
  - Recommended: 0.45

### Depth Parameters:

- `depth_encoder`: Model size/accuracy tradeoff
  - `'vits'`: Fastest, lowest memory, good accuracy
  - `'vitb'`: Balanced (recommended)
  - `'vitl'`: Slower, higher accuracy
  - `'vitg'`: Slowest, highest accuracy, requires more GPU memory

---

## 📁 File Structure

```
Pipeline/
├── integrated_detection_depth_pipeline.py  # Main pipeline code
└── README.md                               # This file

Dependencies:
├── YOLO/
│   ├── tflite_yolo.py                     # YOLO inference class
│   ├── classes.txt                        # Class names
│   └── runs/detect/train/weights/         # Trained model
│
└── Depth-Anything-V2/
    ├── depth_anything_v2/                 # Model architecture
    └── checkpoints/                       # Model weights
        ├── depth_anything_v2_vitb.pth
        └── depth_anything_v2_vits.pth
```

---

## 🔬 Technical Details

### YOLO Model:
- **Architecture:** YOLOv11n (nano variant)
- **Input:** 128×128×3 RGB image
- **Output:** [1, 9, 21504] tensor
  - 9 channels = 4 bbox coords + 5 class scores
  - 21504 anchors across 3 scales (8, 16, 32 stride)
- **Post-processing:** Sigmoid activation, NMS

### Depth Model:
- **Architecture:** Vision Transformer (DPT head)
- **Input:** 518×518×3 RGB image
- **Output:** Dense depth map (same resolution as input)
- **Training:** Supervised on large-scale depth datasets
- **Inference:** Single forward pass, ~100-300ms on GPU

### Performance:
- **YOLO Inference:** ~50-100ms (TFLite on CPU/GPU)
- **Depth Inference:** ~100-300ms (PyTorch on GPU)
- **Total Pipeline:** ~200-500ms per image
- **Memory:** ~2-4GB GPU RAM (depending on encoder)

---

## 🎓 Key Concepts

### 1. **Monocular Depth Estimation**
- Predicts depth from single RGB image
- **Relative depth** (not absolute metric)
- Based on visual cues: size, perspective, occlusion, texture

### 2. **Relative vs. Metric Depth**
- **Relative:** Values are normalized within image (this pipeline)
  - Good for: comparing depths, spatial relationships
  - Not good for: absolute measurements in meters
- **Metric:** Real-world distances in meters
  - Requires: calibration, stereo cameras, or LiDAR

### 3. **Coordinate Systems**
- **Image Space:** (x, y) pixel coordinates, origin top-left
- **Depth Space:** z-axis perpendicular to image plane
- **3D World Space:** Not computed (would need camera calibration)

### 4. **Non-Maximum Suppression (NMS)**
- Removes duplicate/overlapping detections
- Process:
  1. Sort boxes by confidence
  2. Keep highest confidence box
  3. Remove boxes with IoU > threshold
  4. Repeat for remaining boxes

---

## 🐛 Troubleshooting

### "Module not found" errors:
```bash
# Ensure you're in correct conda environment
conda activate depth  # or yolo11

# Install missing packages
pip install opencv-python torch matplotlib
```

### "YOLO model not found":
- Update `YOLO_MODEL` path in script
- Check if model training completed successfully
- Verify path: `YOLO/runs/detect/train/weights/best_saved_model/best_float32.tflite`

### "Depth weights not found":
- Download weights (see main README)
- Place in: `Depth-Anything-V2/checkpoints/`
- Filename format: `depth_anything_v2_{encoder}.pth`

### Out of memory:
- Use smaller encoder: `depth_encoder='vits'`
- Reduce image size
- Close other GPU applications

---

## 🔮 Future Enhancements

1. **Metric Depth Estimation:**
   - Camera calibration integration
   - Convert relative → absolute distances (meters)

2. **3D Reconstruction:**
   - Generate 3D point clouds
   - Visualize in 3D space

3. **Temporal Analysis:**
   - Video processing
   - Track objects across frames
   - Measure velocities and trajectories

4. **Risk Assessment:**
   - Detect proximity violations (e.g., worker too close to machinery)
   - Alert generation based on distance thresholds

5. **Real-time Processing:**
   - Optimize for live video streams
   - Multi-threading for parallel processing

---

## 📚 References

- **YOLOv11:** Ultralytics YOLO documentation
- **Depth Anything V2:** [GitHub Repository](https://github.com/DepthAnything/Depth-Anything-V2)
- **Computer Vision:** Multiple View Geometry in Computer Vision (Hartley & Zisserman)

---

## 👤 Author

Pipeline Integration System
November 2025

---

## 📄 License

This pipeline integrates:
- YOLOv11 (AGPL-3.0)
- Depth Anything V2 (Apache-2.0)

Ensure compliance with both licenses for your use case.
