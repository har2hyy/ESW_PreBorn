# How to Get ACTUAL Real-World Distances

## Current Situation

You have **TWO calibration systems working together**:

### 1. Camera Calibration ✅ (Already Done)
- **File**: `app/src/main/assets/camera_intrinsics.json`
- **What it does**: Converts 2D pixels → 3D coordinates
- **Status**: ✅ Calibrated on 2025-11-16 with checkerboard pattern

### 2. Depth Scale ❌ (Needs Calibration)
- **Current value**: 2.0 (arbitrary default)
- **What it does**: Scales depth model output → real-world meters
- **Status**: ❌ NOT calibrated - just a placeholder

---

## The Complete Pipeline

```
Camera Frame
    ↓
[1] Object Detection (YOLO)
    → Detects worker/vehicle bounding boxes
    ↓
[2] Depth Estimation (DepthAnythingV2)
    → Outputs normalized depth (0.0 to 1.0)
    ↓
[3] Depth Scaling ⚠️ NEEDS CALIBRATION
    → Multiplies by depthScale (currently 2.0)
    → normalized_depth × depthScale = depth_value
    ↓
[4] Camera Calibration ✅ ALREADY CALIBRATED
    → Uses fx, fy, cx, cy to convert:
    → (pixel_x, pixel_y, depth_value) → (X, Y, Z) in real space
    ↓
[5] 3D Distance Calculation
    → distance = √[(X₂-X₁)² + (Y₂-Y₁)² + (Z₂-Z₁)²]
```

**The bottleneck**: Step [3] - depth scale is uncalibrated!

---

## How to Calibrate Depth Scale for REAL Meters

### Method 1: Single-Point Calibration (Quick & Simple)

1. **Place a reference object at a known distance**
   ```bash
   # Example: Put a box exactly 3.00 meters from the QIDK camera
   # Use a measuring tape for accuracy
   ```

2. **Launch the app and point at the object**
   - Note the displayed depth value (e.g., "1.45")
   
3. **Calculate correction factor**
   ```
   Correction Factor = Actual Distance / Displayed Distance
   Example: 3.00m / 1.45 = 2.07
   ```

4. **Update depth scale in code**
   ```java
   // File: app/src/main/java/com/qc/objectdetectionYoloNas/DepthPipelineManager.java
   private float depthScaleMeters = 2.07f; // Update from 2.0f
   ```

5. **Rebuild and test**
   ```bash
   ./gradlew installDebug
   ```
   Object at 3.00m should now show ~3.00

6. **Verify with multiple distances**
   - Test at 1m, 2m, 5m, 10m
   - If accurate across all ranges → ✅ calibrated!
   - If not → See Method 2

---

### Method 2: Multi-Point Calibration (Accurate)

If single-point doesn't work across all distances, depth might be non-linear.

1. **Collect calibration data**
   ```
   Place objects at: 1m, 2m, 3m, 5m, 7m, 10m
   Record displayed depths for each
   ```

2. **Calculate average scale factor**
   ```
   Scale₁ = 1.0m / displayed₁
   Scale₂ = 2.0m / displayed₂
   Scale₃ = 3.0m / displayed₃
   ...
   
   Final Scale = Average(Scale₁, Scale₂, Scale₃, ...)
   ```

3. **Apply best-fit scale**
   Use the average as your depth scale.

---

### Method 3: Runtime Calibration Slider (User-Friendly)

**You already have this!** The depth scale slider (0.1 to 10.0).

**How to use it**:

1. Place a reference object at known distance (e.g., 3.0m)

2. Launch app and point at object

3. Adjust the "Depth Scale Factor" slider in real-time

4. When displayed depth matches actual distance (3.0) → **calibrated!**

5. Test with objects at other distances to verify

6. Note the final slider value and hardcode it in `DepthPipelineManager.java`:
   ```java
   private float depthScaleMeters = 3.45f; // Your calibrated value
   ```

---

## Why Camera Calibration Alone Isn't Enough

### What Camera Calibration Provides:
- **Geometric correctness**: Accurate angular relationships
- **Perspective projection**: Correct field of view
- **Distortion correction**: Straight lines stay straight

### What It DOESN'T Provide:
- ❌ **Absolute depth scale** - The model outputs relative depth (0-1)
- ❌ **Metric units** - Camera matrix is in pixels, not meters

### The Math:

```python
# Camera calibration formula
X_world = (pixel_x - cx) × Z_depth / fx
Y_world = (pixel_y - cy) × Z_depth / fy
Z_world = Z_depth

# Notice: Z_depth is an INPUT, not calculated by calibration!
# Depth model provides Z_depth, but in arbitrary units (0-1)
# Depth scale converts those units to meters
```

---

## Example: Complete Calibration Workflow

### Starting State:
- Depth scale: 2.0 (uncalibrated)
- Object at 5.0m shows: "2.30"

### Calibration Steps:

1. **Calculate scale factor**
   ```
   5.0m / 2.30 = 2.17
   ```

2. **Update depth scale**
   ```
   Old: 2.0
   New: 2.0 × 2.17 = 4.34
   ```

3. **Rebuild app**
   ```bash
   cd VisionSolution1-ObjectDetection-YoloNas
   # Edit DepthPipelineManager.java: depthScaleMeters = 4.34f
   ./gradlew installDebug
   ```

4. **Verify**
   - Object at 5.0m now shows: "5.0" ✅
   - Object at 2.0m now shows: "2.0" ✅
   - Object at 10.0m now shows: "10.0" ✅

### Result:
✅ **Actual real-world meter values!**

---

## Testing Your Calibration

### Create Test Markers:

```bash
# Place tape markers at known distances:
1m  - "1 METER"
2m  - "2 METERS"
3m  - "3 METERS"
5m  - "5 METERS"
10m - "10 METERS"
```

### Point camera at each marker and verify:
- Displayed depth should match marker distance
- If error > 10%, recalibrate
- If error < 5%, you're golden! ✅

---

## Why This Works

### The Full Math:

```python
# Step 1: Depth model output (relative)
normalized_depth = 0.5  # Model says halfway through scene

# Step 2: Convert to metric depth (YOUR CALIBRATION)
depth_meters = normalized_depth × depthScaleMeters
             = 0.5 × 4.34
             = 2.17 meters

# Step 3: Camera calibration converts to 3D (ALREADY CALIBRATED)
X = (pixel_x - cx) × depth_meters / fx
Y = (pixel_y - cy) × depth_meters / fy
Z = depth_meters

# Result: Real-world 3D coordinates in METERS
Point3D(X=0.45m, Y=-0.23m, Z=2.17m)
```

### The Key Insight:
- **Camera calibration** handles the **geometry** (angles, perspective)
- **Depth scale** handles the **metric units** (arbitrary → meters)
- **Both together** = Real-world 3D positions in meters!

---

## Current Status

✅ Camera intrinsics calibrated (fx, fy, cx, cy)  
✅ Depth model functional (DepthAnythingV2 on NPU)  
✅ 3D projection code implemented (`pixelToPoint`)  
✅ Slider UI for runtime calibration (0.1 to 10.0)  
❌ Depth scale NOT calibrated (still using default 2.0)

**Next step**: Run calibration procedure to find your real depth scale!

---

## Quick Start: 5-Minute Calibration

```bash
# 1. Place object at exactly 3.0 meters (use measuring tape)

# 2. Connect QIDK and launch app
./gradlew installDebug

# 3. Point camera at object, note displayed depth
# Example: Shows "1.40"

# 4. Calculate new scale
# 3.0 / 1.40 = 2.14
# Current scale: 2.0
# New scale: 2.0 × 2.14 = 4.28

# 5. Update DepthPipelineManager.java
# private float depthScaleMeters = 4.28f;

# 6. Rebuild
./gradlew installDebug

# 7. Verify: Object should now show "~3.0"
```

---

## Summary

### Q: What does camera calibration give us?
**A**: Accurate geometric projection (pixels → 3D angles & ratios)

### Q: What's missing?
**A**: Depth scale (relative depth → absolute meters)

### Q: How to fix it?
**A**: Calibrate depth scale using known-distance objects

### Q: Will it give actual real-world meters?
**A**: ✅ YES! After calibrating depth scale, all distances will be in true meters

### Q: Is the camera calibration wasted?
**A**: ❌ NO! It's essential for accurate 3D positioning and distance calculations

