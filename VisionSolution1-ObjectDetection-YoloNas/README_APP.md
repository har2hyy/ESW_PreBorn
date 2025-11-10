# VisionSolution1-ObjectDetection-YoloNas

Android application for real-time object detection using YOLO on Qualcomm platforms.

## Overview

This is an Android Gradle project that implements YOLOv11 object detection optimized for Qualcomm devices using:
- **YOLO Model**: YOLOv11n converted to ONNX format
- **TFLite Models**: Float16 and Float32 quantized models
- **Android SDK**: Gradle-based build system

## Project Structure

```
VisionSolution1-ObjectDetection-YoloNas/
├── app/                          # Main Android application module
│   ├── src/main/                 # Source code and resources
│   ├── build.gradle              # App-level build configuration
│   └── google-services.json      # Firebase configuration
├── yolo11n.onnx                  # YOLO model in ONNX format (10.6 MB)
├── yolo11n_saved_model/          # TFLite converted models
│   ├── yolo11n_float16.tflite   # Float16 quantized model
│   └── yolo11n_float32.tflite   # Float32 quantized model
├── GenerateDLC.ipynb             # Jupyter notebook for DLC generation
├── build.gradle                  # Project-level build configuration
├── settings.gradle               # Project settings
├── gradlew                       # Gradle wrapper (Unix)
├── gradlew.bat                   # Gradle wrapper (Windows)
└── .gitattributes                # Git LFS configuration
```

## Model Information

- **Model**: YOLOv11n (nano variant)
- **Format**: ONNX, TFLite (Float16/Float32)
- **Classes**: Custom construction site classes
  - worker
  - truck
  - bike
  - bulldozer
  - car

## Git LFS Files

The following files are tracked with Git LFS (as per `.gitattributes`):
- OpenCV native libraries
- SNPE release files
- App assets
- JNI libraries
- ONNX models

## Build Instructions

### Prerequisites
- Android Studio (latest version)
- JDK 11 or higher
- Android SDK (API level as specified in app/build.gradle)
- Git LFS (for large files)

### Building the App

1. **Clone the repository** (if not already cloned):
   ```bash
   git lfs install
   git clone <repository-url>
   ```

2. **Open in Android Studio**:
   - Open Android Studio
   - File → Open → Select this directory

3. **Sync Gradle**:
   ```bash
   ./gradlew build
   ```

4. **Run on device**:
   - Connect Android device via USB
   - Click "Run" in Android Studio

### Command Line Build

```bash
# Clean build
./gradlew clean

# Build debug APK
./gradlew assembleDebug

# Build release APK
./gradlew assembleRelease

# Install on connected device
./gradlew installDebug
```

## Related Components

This Android app is part of the larger ESW_PreBorn project:
- **Pipeline/**: PyTorch-based detection + depth pipeline (Python)
- **YOLO/**: Model training and TFLite conversion scripts
- **Depth-Anything-V2/**: Depth estimation model (not used in Android app)

## Model Training

The YOLO model used in this app was trained using:
```
../YOLO/train.py
```

See `../YOLO/README.md` for training details.

## Notes

- The app is optimized for Qualcomm Snapdragon devices
- TFLite models are preferred for mobile deployment
- Original PyTorch model is in `../YOLO/runs/detect/train/weights/best.pt`
- ONNX export was done using `../YOLO/export_int8.py`

## Output Files Location

Built APKs can be found at:
```
app/build/outputs/apk/debug/app-debug.apk
app/build/outputs/apk/release/app-release.apk
```

## License

See LICENSE file in the root of ESW_PreBorn project.
