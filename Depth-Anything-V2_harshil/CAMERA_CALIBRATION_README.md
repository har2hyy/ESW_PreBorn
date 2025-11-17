# Camera Calibration & Metric Depth Estimation

This folder now includes **camera calibration** and **metric depth conversion** capabilities, allowing you to convert relative depth maps (pixels) to actual real-world distances (meters).

## 🆕 New Features Added from `depth_perception/`

The following files have been integrated from the `depth_perception` folder:

1. **`calibrate_camera.py`** - Camera calibration using checkerboard patterns
2. **`pixel_to_3d.py`** - Convert pixel coordinates to 3D world coordinates
3. **`align_depth_scale.py`** - Scale relative depth maps to metric depth
4. **`run_tflite_depth.py`** - Enhanced TFLite depth estimation with calibration support
5. **`camera_intrinsics.json`** - Pre-calibrated camera parameters

---

## 📊 What's the Difference?

### ❌ Old Depth-Anything-V2 (Pixel-based)
- Outputs **relative depth** (arbitrary units)
- No real-world distance measurements
- Can't determine if two workers are actually 2 meters apart
- Depth values are just for visualization

### ✅ New Camera-Optimized Version (Metric depth)
- Outputs **metric depth** (meters)
- Calculates **actual distances** between objects
- Can measure safety distances accurately
- Uses **camera calibration** for precise 3D coordinates
- Accounts for **lens distortion**

---

## 🚀 Quick Start

### 1. Camera Calibration (One-time setup)

If you have calibration videos with a checkerboard pattern:

```bash
python calibrate_camera.py \
    --videos calibration_video_1.mp4 calibration_video_2.mp4 \
    --pattern-cols 7 --pattern-rows 9 \
    --square-size 0.02 \
    --output camera_intrinsics.json \
    --debug-images
```

**Note**: Pattern size is the number of **internal corners**, not squares!
- A 8×10 checkerboard has 7×9 internal corners
- `square-size` is in meters (0.02 = 2cm)

### 2. Run Depth Estimation with Calibration

```bash
python run_tflite_depth.py \
    --model checkpoints/depth_anything_v2.tflite \
    --image input_image.jpg \
    --intrinsics camera_intrinsics.json \
    --output depth_result.png
```

This will generate:
- `depth_result.png` - Colorized depth visualization
- `depth_result_raw.png` - Raw 16-bit depth data
- Camera calibration overlay with FOV information

### 3. Convert Depth Map to Metric Depth

If you have a known reference distance in your image:

```bash
python align_depth_scale.py \
    --raw depth_result_raw.png \
    --roi 100 200 300 300 \
    --known-distance 5.0 \
    --output depth_metric.png
```

Where:
- `--roi x y w h` - Region of interest at known distance
- `--known-distance` - Actual distance in meters to that region

### 4. Use Pixel-to-3D Converter

```python
from pixel_to_3d import PixelTo3DConverter
import cv2
import numpy as np

# Initialize converter
converter = PixelTo3DConverter('camera_intrinsics.json')

# Load depth map
depth_map = cv2.imread('depth_metric.png', cv2.IMREAD_UNCHANGED)

# Convert pixel coordinates to 3D
u, v = 320, 240  # Pixel coordinates
depth = depth_map[v, u]  # Get depth at that pixel
x, y, z = converter.pixel_to_3d(u, v, depth)

print(f"3D position: ({x:.2f}, {y:.2f}, {z:.2f}) meters")

# Measure distance between two workers
worker1_3d = converter.pixel_to_3d(320, 240, depth_map[240, 320])
worker2_3d = converter.pixel_to_3d(450, 280, depth_map[280, 450])

distance = converter.distance_between_points(worker1_3d, worker2_3d)
horizontal_dist = converter.horizontal_distance(worker1_3d, worker2_3d)

print(f"Distance between workers: {distance:.2f}m")
print(f"Horizontal distance: {horizontal_dist:.2f}m")
```

---

## 📐 Camera Intrinsics Explained

The `camera_intrinsics.json` file contains:

```json
{
  "focal_length": {
    "fx": 1309.86,  // Focal length in X direction (pixels)
    "fy": 1315.80   // Focal length in Y direction (pixels)
  },
  "principal_point": {
    "cx": 944.08,   // Optical center X (pixels)
    "cy": 542.44    // Optical center Y (pixels)
  },
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "distortion_coefficients": [
    -0.440,  // k1 - radial distortion
    0.636,   // k2
    0.004,   // p1 - tangential distortion
    -0.005,  // p2
    -0.464   // k3
  ],
  "reprojection_error_pixels": 0.17  // Quality metric (lower is better)
}
```

### What These Parameters Mean:

- **Focal Length (fx, fy)**: How "zoomed in" the camera is
- **Principal Point (cx, cy)**: Center of the image sensor (usually near center)
- **Distortion Coefficients**: Corrects lens distortion (barrel/pincushion effects)
- **Reprojection Error**: Quality of calibration
  - < 0.5 pixels = Excellent ✨
  - < 1.0 pixels = Good ✓
  - < 2.0 pixels = Acceptable ⚠️
  - \> 2.0 pixels = Poor - Recalibrate ❌

---

## 🎯 Use Cases for Construction Safety

### Example 1: Measure Worker-to-Worker Distance

```python
from pixel_to_3d import PixelTo3DConverter
import cv2

# Load YOLO detections
detections = yolo_model.detect('construction_site.jpg')

# Load depth map
depth_map = cv2.imread('depth_metric.tiff', cv2.IMREAD_UNCHANGED)

# Initialize converter
converter = PixelTo3DConverter('camera_intrinsics.json')

# For each pair of workers
for worker1, worker2 in worker_pairs:
    # Get 3D positions
    x1, y1 = worker1.center
    x2, y2 = worker2.center
    
    pos1 = converter.pixel_to_3d(x1, y1, depth_map[y1, x1])
    pos2 = converter.pixel_to_3d(x2, y2, depth_map[y2, x2])
    
    # Calculate distance
    distance = converter.horizontal_distance(pos1, pos2)
    
    # Check safety
    if distance < 2.0:  # Less than 2 meters
        print(f"⚠️  Safety violation! Workers {distance:.2f}m apart")
```

### Example 2: Measure Object Sizes

```python
# Get bounding box 3D coordinates
bbox = (100, 150, 200, 300)  # x1, y1, x2, y2
bbox_3d = converter.bbox_to_3d(bbox, depth_map)

print(f"Object center: {bbox_3d['center_3d']}")
print(f"Object depth: {bbox_3d['mean_depth']:.2f}m")

# Calculate physical size
corners = bbox_3d['corners_3d']
width = converter.distance_between_points(corners[0], corners[1])
height = converter.distance_between_points(corners[0], corners[2])
print(f"Object size: {width:.2f}m × {height:.2f}m")
```

### Example 3: Create Point Cloud for 3D Visualization

```python
# Load RGB image and depth map
rgb = cv2.imread('construction_site.jpg')
depth = cv2.imread('depth_metric.tiff', cv2.IMREAD_UNCHANGED)

# Create point cloud
points = converter.create_point_cloud(depth, rgb)

# Save as PLY file (can be opened in MeshLab, CloudCompare, etc.)
converter.save_point_cloud(points, 'construction_site_3d.ply')
```

---

## 🔄 Workflow Integration

### Complete Pipeline for Construction Safety

1. **Calibrate Camera** (once per camera)
   ```bash
   python calibrate_camera.py --videos calib*.mp4 --pattern-cols 7 --pattern-rows 9 --square-size 0.02
   ```

2. **Capture Construction Site Image**
   - Use the same camera that was calibrated

3. **Run Depth Estimation**
   ```bash
   python run_tflite_depth.py --model depth_anything_v2.tflite --image site.jpg --intrinsics camera_intrinsics.json
   ```

4. **Align to Metric Depth** (if needed)
   ```bash
   python align_depth_scale.py --raw depth_result_raw.png --roi 100 200 300 300 --known-distance 5.0
   ```

5. **Run YOLO Detection**
   ```bash
   cd ../YOLO
   python run_optimal_detection.py --image ../Depth-Anything-V2/site.jpg
   ```

6. **Calculate Safety Distances**
   - Use `pixel_to_3d.py` to convert YOLO bboxes to 3D
   - Measure distances between workers
   - Alert if safety thresholds violated

---

## 📚 API Reference

### PixelTo3DConverter Class

```python
converter = PixelTo3DConverter('camera_intrinsics.json')
```

#### Methods:

**`pixel_to_3d(u, v, depth)`**
- Convert pixel coordinates to 3D world coordinates
- Returns: `(X, Y, Z)` in meters

**`bbox_to_3d(bbox, depth_map)`**
- Convert bounding box to 3D
- Returns: Dict with `center_3d`, `corners_3d`, `mean_depth`

**`distance_between_points(point1, point2)`**
- Calculate Euclidean distance between two 3D points
- Returns: Distance in meters

**`horizontal_distance(point1, point2)`**
- Calculate ground-level distance (ignores Y axis)
- Returns: Horizontal distance in meters

**`undistort_image(image)`**
- Remove lens distortion from image
- Returns: Undistorted image

**`get_field_of_view()`**
- Get camera field of view
- Returns: Dict with `horizontal_fov_degrees`, `vertical_fov_degrees`

**`create_point_cloud(depth_map, rgb_image=None)`**
- Create 3D point cloud from depth map
- Returns: Numpy array (N, 3) or (N, 6) with XYZ or XYZRGB

**`save_point_cloud(points, output_path)`**
- Save point cloud to PLY file
- Args: points array, output .ply path

---

## 🎓 Technical Details

### Coordinate Systems

- **Image coordinates**: (u, v) in pixels, origin at top-left
- **Camera coordinates**: (X, Y, Z) in meters, origin at camera
  - X: Right
  - Y: Down
  - Z: Forward (into the scene)
- **World coordinates**: After undistortion and transformation

### Depth Map Formats

- **Relative depth**: Values 0-1 (from Depth-Anything-V2)
- **16-bit PNG**: Scaled to 0-65535 for storage
- **Metric depth**: Actual distances in meters after alignment
- **Float32 TIFF**: High-precision metric depth storage

### Calibration Quality

The reprojection error tells you how accurate your calibration is:

```
Error = sqrt(Σ(projected_point - actual_point)²) / N
```

To improve calibration:
- Use more frames (50-100)
- Vary checkerboard angles and distances
- Ensure checkerboard is flat and well-lit
- Use smaller frame skip to get more samples

---

## ⚠️ Important Notes

1. **Depth-Anything outputs relative depth**, not metric depth
   - Use `align_depth_scale.py` to convert to meters
   - Or use a reference object at known distance

2. **Camera must stay the same**
   - Recalibrate if you change camera, lens, or zoom
   - Calibration is specific to one camera configuration

3. **Metric depth accuracy**
   - Depends on calibration quality
   - Typical error: ±10-20cm for objects 5-10m away
   - Better for closer objects, worse for farther

4. **Lens distortion**
   - Always undistort images before measuring
   - Use `converter.undistort_image()` or `cv2.undistort()`

---

## 🔧 Troubleshooting

### "Checkerboard not detected"
- Ensure pattern size is correct (internal corners, not squares)
- Check lighting - avoid glare and shadows
- Try different `--frame-skip` values
- Ensure checkerboard is clearly visible

### "Reprojection error too high"
- Collect more calibration frames
- Ensure checkerboard is flat
- Vary angles and distances more
- Check if pattern size is correct

### "Depth values seem incorrect"
- Check if you aligned to metric depth with `align_depth_scale.py`
- Verify reference distance is accurate
- Ensure depth map is not inverted (far = high values)

### "Safety distances don't make sense"
- Verify camera intrinsics match the image
- Check if depth map is in correct units (meters)
- Ensure you're using `horizontal_distance()` for ground-level checks

---

## 📖 Further Reading

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Depth-Anything-V2 Paper](https://arxiv.org/abs/2406.09414)
- [Camera Pinhole Model](https://en.wikipedia.org/wiki/Pinhole_camera_model)

---

## ✅ Summary

You now have a **complete depth estimation system** that:

- ✅ Calibrates cameras with checkerboard patterns
- ✅ Estimates depth from single images
- ✅ Converts relative depth to metric depth
- ✅ Transforms pixel coordinates to 3D world coordinates
- ✅ Measures real-world distances between objects
- ✅ Corrects for lens distortion
- ✅ Creates 3D point clouds

This enables **accurate safety monitoring** for construction sites with real distance measurements! 🏗️👷‍♂️
