# QNN/DLC Models

This folder contains models in Qualcomm Neural Network (QNN) format for NPU deployment on QIDK devices.

## Files

- **best_yolo.cpp** - QNN model C++ source (FP32, 1.5 MB)
- **best_yolo.bin** - QNN model weights binary (FP32, 11 MB) ⚠️ Not found - see notes
- **best_yolo_fp16.cpp** - FP16 precision model (if created)
- **best_yolo_net.json** - Network configuration JSON
- **best_yolo_fp16_net.json** - FP16 network configuration

## ⚠️ Current Status

**YOLO11 Incompatibility Issue:**
- YOLO11 C2f module causes "Unable to broadcast shapes" error in QNN 2.40.0
- FP32 conversion: ✅ Success (best_yolo.cpp + .bin created)
- INT8 quantization: ❌ Failed (broadcast shape mismatch)

**See**: `../../docs/ALL_DLC_CONVERSION_PATHS.md` for 8 alternative paths to DLC

## Recommended Solutions

### ⭐ Solution 1: Use YOLOv8 (90% Success Rate)
```bash
# Retrain with YOLOv8 architecture
cd ../../scripts/training
pip install ultralytics==8.0.196

python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Use v8, not v11
model.train(
    data='../../data/construction_data.yaml',
    epochs=100,
    imgsz=1024,
    batch=8,
    device='cpu',
    name='yolov8_construction'
)
model.export(format='onnx', simplify=True, opset=12)
"

# Then convert to DLC
cd ../conversion
./run_qnn_converter.sh
```

### Solution 2: Use TFLite with Hexagon Delegate
See: `../tflite/README.md` and `../../docs/FIX_TFLITE_NPU.md`

## QNN Model Usage (When Available)

### Compile to Context Binary
```bash
# Activate QNN environment
source ~/snpe-sdk/2.40.0.251030/bin/envsetup.sh

# Generate context binary for HTP (NPU)
qnn-model-lib-generator \
  -c best_yolo.cpp \
  -b best_yolo.bin \
  -o libYolo11Model.so \
  -t aarch64-android

# Or use context binary generator
qnn-context-binary-generator \
  --model best_yolo.cpp \
  --backend libQnnHtp.so \
  --binary_file best_yolo.bin \
  --output_dir ./context_binaries
```

### Test on PC
```bash
cd ../../scripts/testing
python test_dlc_on_pc.py
```

### Deploy to QIDK Device
```bash
# Push model to device
adb push libYolo11Model.so /data/local/tmp/
adb push ../../calibration/calibration_raw/image_001.raw /data/local/tmp/

# Run inference on NPU
adb shell "cd /data/local/tmp && \
  qnn-net-run \
    --model libYolo11Model.so \
    --backend libQnnHtp.so \
    --input_list input.txt \
    --output_dir output"

# See full guide: ../../docs/QIDK_NPU_TESTING_GUIDE.md
```

## Conversion from ONNX

### FP32 Conversion (Working)
```bash
cd ../../scripts/conversion

# Activate environment
conda activate dlc
source ~/snpe-sdk/2.40.0.251030/bin/envsetup.sh

# Convert ONNX to QNN FP32
qnn-onnx-converter \
  --input_network ../../models/onnx/best_simplified.onnx \
  --output_path best_yolo.cpp

# Output: best_yolo.cpp + best_yolo.bin
```

### INT8 Quantization (Currently Failing for YOLO11)
```bash
# Requires calibration data
qnn-onnx-converter \
  --input_network ../../models/onnx/best_simplified.onnx \
  --output_path best_yolo_int8.cpp \
  --input_list ../../calibration/calibration_list_raw.txt \
  --param_quantizer tf \
  --act_quantizer tf \
  --weight_bw 8 \
  --act_bw 8 \
  --use_per_channel_quantization

# ❌ Fails with broadcast shape error
# Solution: Use YOLOv8 instead (see above)
```

## Model Specifications

### Input
- **Format**: Raw binary (.raw files)
- **Shape**: [1, 3, 1024, 1024]
- **Type**: float32 (FP32) or uint8 (INT8)
- **Layout**: NCHW (batch, channels, height, width)
- **Preprocessing**: BGR→RGB, normalize [0,1], transpose to CHW

### Output
- **Shape**: [1, 9, 21504]
- **Format**: 4 bbox coords + 5 class scores per anchor
- **Type**: float32 (FP32) or uint8 (INT8)

## Performance Expectations

| Backend | Precision | Inference Time | Notes |
|---------|-----------|----------------|-------|
| CPU (x86_64) | FP32 | 800-1200ms | PC testing only |
| HTP (NPU) | FP32 | 150-300ms | Qualcomm HTP |
| HTP (NPU) | INT8 | 20-80ms | Best performance |
| GPU (Adreno) | FP16 | 100-200ms | Fallback option |

## Calibration Data

Calibration images for quantization are located at:
- `../../calibration/calibration_raw/` - 150 preprocessed .raw files
- `../../calibration/calibration_list_raw.txt` - File list

Generate calibration data:
```bash
cd ../../calibration
python prepare_calibration_data.py
```

## Known Issues

1. **YOLO11 Architecture Incompatibility**
   - Error: "getBroadcastedTensorShape: Unable to broadcast shapes"
   - Location: /model.2/m.0/Add (C2f module)
   - Status: Unsupported in QNN 2.40.0
   - Workaround: Use YOLOv8 or TFLite path

2. **Missing .bin File**
   - If best_yolo.bin is missing, re-run conversion
   - Both .cpp and .bin required for compilation

3. **Library Conflicts**
   - Use clean conda environment 'dlc'
   - Source QNN environment before conversion

## Documentation

- **Installation**: `../../docs/INSTALL_SNPE_SDK.md`
- **Conversion Guide**: `../../docs/DLC_CONVERSION_GUIDE.md`
- **All Paths to DLC**: `../../docs/ALL_DLC_CONVERSION_PATHS.md`
- **QIDK Testing**: `../../docs/QIDK_NPU_TESTING_GUIDE.md`
- **Complete Workflow**: `../../docs/COMPLETE_WORKFLOW.md`

## Next Steps

1. **Try YOLOv8 conversion** (recommended)
2. **Or use TFLite + Hexagon Delegate** (alternative)
3. **Contact Qualcomm** for YOLO11 support timeline
4. **Update to QNN 2.50+** when available (may support YOLO11)
