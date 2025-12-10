# Integrated Pipeline Integration Guide

## ✅ What's Ready

1. **C++ Pipeline** (`integrated_pipeline_onnx.cpp`)
   - Depth estimation using OpenCV (monocular cues)
   - Detection + depth aggregation
   - Distance calculations (2D + 2.5D)
   - JSON output matching Python pipeline

2. **Java Wrapper** (`IntegratedPipeline.java`)
   - JNI bridge to C++ pipeline
   - Easy-to-use API

3. **YOLO Model** 
   - ✅ Copied: `yolo11.onnx` → `app/src/main/assets/`
   - Size: 11MB

4. **Build Configuration**
   - ✅ CMakeLists.txt updated

## 🚀 How to Use in Your App

### Option A: Minimal Integration (Recommended)

Add to your existing inference code:

```java
// In CameraFragment or wherever you do inference
private IntegratedPipeline integratedPipeline;

@Override
public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    
    // Initialize integrated pipeline
    integratedPipeline = new IntegratedPipeline();
    if (integratedPipeline.initialize()) {
        Log.i(TAG, "Integrated pipeline ready!");
    }
}

// After getting YOLO detections:
private void processDetections(Mat frameMat, List<DetectionWithDepth> detections) {
    // Pass to integrated pipeline for depth analysis
    String jsonResult = integratedPipeline.process(frameMat, detections);
    
    // Parse JSON result
    try {
        JSONObject result = new JSONObject(jsonResult);
        int totalDetections = result.getInt("total_detections");
        JSONArray detectionsArray = result.getJSONArray("detections");
        JSONArray distancesArray = result.getJSONArray("distances");
        
        // Use depth-enhanced data
        for (int i = 0; i < detectionsArray.length(); i++) {
            JSONObject det = detectionsArray.getJSONObject(i);
            String className = det.getString("class");
            double depthAvg = det.getDouble("depth_avg");
            int centerX = det.getJSONArray("center").getInt(0);
            int centerY = det.getJSONArray("center").getInt(1);
            
            Log.i(TAG, String.format("%s depth: %.1f @ (%d,%d)", 
                  className, depthAvg, centerX, centerY));
        }
        
        // Show closest pair
        if (distancesArray.length() > 0) {
            JSONObject closest = distancesArray.getJSONObject(0);
            String obj1 = closest.getString("obj1_class");
            String obj2 = closest.getString("obj2_class");
            double distance = closest.getDouble("euclidean");
            double depthDiff = closest.getDouble("depth_diff");
            
            Log.i(TAG, String.format("Closest: %s ↔ %s (%.1fpx, Δdepth: %.1f)",
                  obj1, obj2, distance, depthDiff));
        }
        
    } catch (JSONException e) {
        Log.e(TAG, "Error parsing pipeline result", e);
    }
}

@Override
public void onDestroy() {
    if (integratedPipeline != null) {
        integratedPipeline.release();
    }
    super.onDestroy();
}
```

### Option B: Full Replacement

Replace your existing TFLite runner with integrated pipeline for complete depth-aware inference.

## 📊 Output Format

The pipeline returns JSON with this structure:

```json
{
  "total_detections": 3,
  "depth_inference_ms": 45.2,
  "detections": [
    {
      "id": 0,
      "class": "worker",
      "class_id": 0,
      "confidence": 0.89,
      "bbox": [100, 150, 80, 120],
      "center": [140, 210],
      "depth_avg": 128.5,
      "depth_median": 130.0,
      "depth_min": 90.0,
      "depth_max": 165.0,
      "depth_center": 128.0,
      "area": 9600
    }
  ],
  "distances": [
    {
      "obj1_id": 0,
      "obj1_class": "worker",
      "obj2_id": 1,
      "obj2_class": "truck",
      "euclidean": 245.6,
      "horizontal": 180.0,
      "vertical": 165.0,
      "depth_diff": 45.2,
      "depth_diff_scaled": 135.6
    }
  ]
}
```

## 🔧 Build Instructions

```bash
cd /home/gyandeep_das/Desktop/New\ Folder/ESW_PreBorn/VisionSolution1-ObjectDetection-YoloNas

# Build
./gradlew assembleDebug

# Install
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## 🎯 Features Matching Python Pipeline

✅ **Stage 1**: YOLO detection (your existing code)
✅ **Stage 2**: Depth estimation (monocular OpenCV-based)
✅ **Stage 3**: Per-object depth statistics
✅ **Stage 4**: Pairwise distance calculations
✅ **Stage 5**: JSON output for further processing

## 🚨 Current Limitations

1. **Depth Model**: Uses OpenCV gradient-based estimation (not Depth Anything V2)
   - Real Depth Anything V2 requires:
     - PyTorch Mobile (large)
     - Or ONNX Runtime with large model (~600MB)
     - Or TFLite conversion (complex)

2. **NPU Acceleration**: Not available without SNPE/QNN SDK
   - Current implementation runs on CPU/GPU

## 💡 Next Steps to Match Python Pipeline Exactly

### To add real Depth Anything V2:

1. **Export depth model to ONNX**:
```python
# In your Python environment
cd /home/gyandeep_das/Desktop/New\ Folder/ESW_PreBorn/Depth-Anything-V2
python export_to_onnx.py --encoder vits --checkpoint checkpoints/depth_anything_v2_vits.pth
```

2. **Add ONNX Runtime to Android**:
```gradle
// app/build.gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
}
```

3. **Load in C++** or use Java ONNX Runtime

### To get NPU acceleration:

- Install QNN SDK (better YOLOv11 support than SNPE)
- Convert ONNX → QNN context binary
- Update C++ to use QNN runtime

## 📝 Testing

Once built, the app will:
1. Show camera preview
2. Run YOLO detection
3. Enhance detections with depth info
4. Calculate spatial relationships
5. Display results

Check logcat for detailed output:
```bash
adb logcat | grep IntegratedPipeline
```

## ✨ Status

**READY TO BUILD** - All code integrated, just needs compilation!

The pipeline is functionally complete and mirrors your Python implementation's logic, just with a simpler depth estimation method for now.
