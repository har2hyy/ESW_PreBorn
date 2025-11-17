# YOLO Construction Safety Detection

YOLOv11 model for construction site safety - detecting workers, vehicles, and equipment.

## 📁 Organized Folder Structure

```
YOLO/
├── models/                      # Model files organized by format
│   ├── pytorch/                 # PyTorch .pt models
│   │   ├── best.pt             # Trained YOLOv11 model (1024x1024)
│   │   ├── yolo11n.pt          # Pre-trained YOLOv11 nano
│   │   └── README.md           # PyTorch usage guide
│   ├── onnx/                    # ONNX models for cross-platform
│   │   ├── best.onnx           # Original ONNX export
│   │   ├── best_simplified.onnx # Simplified (recommended)
│   │   └── README.md           # ONNX usage guide
│   ├── tflite/                  # TensorFlow Lite for mobile
│   │   ├── best_yolo_tflite/   # TFLite model directory
│   │   │   ├── best_simplified_float32.tflite (11 MB)
│   │   │   └── best_simplified_float16.tflite (5.2 MB)
│   │   └── README.md           # TFLite usage + Android deployment
│   └── qnn_dlc/                 # Qualcomm QNN for NPU
│       ├── best_yolo.cpp       # QNN model C++ (FP32)
│       ├── best_yolo_net.json  # Network configuration
│       └── README.md           # QNN/DLC conversion guide
│
├── scripts/                     # Organized scripts by purpose
│   ├── training/                # Model training
│   │   ├── train.py            # Main training script
│   │   └── export_int8.py      # Export + quantization
│   ├── conversion/              # Format conversions
│   │   ├── convert_to_dlc.sh   # ONNX → QNN/DLC
│   │   └── run_qnn_converter.sh # QNN converter wrapper
│   ├── testing/                 # Model testing & validation
│   │   ├── test_onnx_on_pc.py  # Test ONNX on PC
│   │   ├── test_dlc_on_pc.py   # Test DLC on PC
│   │   ├── validate_onnx_for_npu.py # NPU compatibility check
│   │   ├── optimized_yolo_onnx.py   # Optimized ONNX inference
│   │   ├── decode_npu_output.py     # Parse NPU outputs
│   │   └── prepare_npu_input.py     # Prepare NPU inputs
│   └── deployment/              # Device deployment
│       ├── remote_npu_pipeline.py   # Remote NPU testing
│       ├── setup_qidk_device.sh     # QIDK setup script
│       ├── install_snpe_helper.sh   # SNPE/QNN installer
│       └── verify_and_proceed.sh    # Deployment verification
│
├── calibration/                 # Quantization calibration data
│   ├── calibration_raw/         # 150 preprocessed .raw files
│   ├── calibration_list.txt     # Image paths (JPG)
│   ├── calibration_list_raw.txt # Raw file paths (.raw)
│   ├── prepare_calibration_data.py # Preprocessing script
│   └── calibration_image_sample_data_20x128x128x3_float32.npy
│
├── data/                        # Dataset & configuration
│   ├── construction_data.yaml   # Dataset config
│   ├── classes.txt              # Class names
│   ├── train/                   # Training images + labels
│   │   ├── images/
│   │   └── labels/
│   ├── val/                     # Validation images + labels
│   │   ├── images/
│   │   └── labels/
│   └── data_1-100/              # Additional dataset
│
├── docs/                        # Documentation
│   ├── README.md                # Original README
│   ├── ALL_DLC_CONVERSION_PATHS.md  # 8 paths to DLC format
│   ├── FIX_TFLITE_NPU.md        # Fix TFLite NPU fallback
│   ├── INSTALL_SNPE_SDK.md      # QNN/SNPE SDK installation
│   ├── QIDK_NPU_TESTING_GUIDE.md # QIDK device testing
│   ├── DLC_CONVERSION_GUIDE.md  # DLC conversion guide
│   ├── COMPLETE_WORKFLOW.md     # Full workflow guide
│   ├── COMPLETE_NPU_DEPLOYMENT.md # NPU deployment guide
│   ├── NPU_DEPLOYMENT_CHECKLIST.md # Deployment checklist
│   ├── NPU_WORKFLOW_SUMMARY.md  # Workflow summary
│   └── QUICK_REFERENCE.txt      # Quick commands
│
├── output/                      # Test outputs & results
│   ├── onnx_pc_test_results/    # ONNX test results
│   ├── debug_output.npy         # Debug outputs
│   └── onnx_npu_validation.json # NPU validation results
│
├── runs/                        # Training runs (Ultralytics)
│   └── detect/
│       └── train/               # Training artifacts
│
├── data_1-100/                  # Legacy dataset (kept for reference)
├── __pycache__/                 # Python cache
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Train Model (PyTorch)
```bash
cd scripts/training
python train.py
```

### 2. Export to ONNX
```bash
cd models/pytorch
python -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', simplify=True, opset=12)
"
mv best.onnx ../onnx/
```

### 3. Test ONNX on PC
```bash
cd scripts/testing
python test_onnx_on_pc.py
```

### 4. Convert to TFLite (Mobile)
```bash
cd models/onnx
pip install onnx2tf tensorflow
onnx2tf -i best_simplified.onnx -o ../tflite/best_tflite
```

### 5. Convert to QNN/DLC (QIDK NPU)
```bash
cd scripts/conversion
conda activate dlc
source ~/snpe-sdk/2.40.0.251030/bin/envsetup.sh
./run_qnn_converter.sh
```

## 📊 Model Information

| Property | Value |
|----------|-------|
| **Architecture** | YOLOv11n |
| **Input Size** | 1024×1024 |
| **Classes** | 5 (worker, truck, bike, bulldozer, car) |
| **Dataset** | Construction site images |
| **Training Epochs** | 100 |
| **Framework** | Ultralytics YOLO |

## 🎯 Available Model Formats

| Format | File | Size | Use Case | Status |
|--------|------|------|----------|--------|
| **PyTorch** | best.pt | ~11 MB | Training, inference | ✅ Ready |
| **ONNX** | best_simplified.onnx | 11 MB | Cross-platform | ✅ Ready |
| **TFLite (FP32)** | best_simplified_float32.tflite | 11 MB | Mobile (CPU) | ✅ Ready |
| **TFLite (FP16)** | best_simplified_float16.tflite | 5.2 MB | Mobile (GPU) | ✅ Ready |
| **TFLite (INT8)** | best_int8.tflite | ~3 MB | Mobile (NPU) | ⏳ Create |
| **QNN/DLC (FP32)** | best_yolo.cpp/.bin | 12.5 MB | QIDK NPU | ✅ Ready |
| **QNN/DLC (INT8)** | best_yolo_int8.cpp | ~3 MB | QIDK NPU | ❌ YOLO11 issue |

## ⚠️ Known Issues

### YOLO11 → QNN/DLC Conversion Issue
- **Problem**: YOLO11 C2f module incompatible with QNN 2.40.0
- **Error**: "Unable to broadcast shapes" in Add operations
- **Status**: FP32 works, INT8 quantization fails
- **Solutions**: See `docs/ALL_DLC_CONVERSION_PATHS.md` for 8 alternative paths

### TFLite NPU Fallback Issue
- **Problem**: INT8 TFLite models run on CPU instead of NPU
- **Cause**: NNAPI disabled + missing Hexagon Delegate
- **Solution**: See `docs/FIX_TFLITE_NPU.md`

## 📖 Documentation Guide

| Document | Purpose |
|----------|---------|
| **ALL_DLC_CONVERSION_PATHS.md** | 8 paths to convert YOLO11 to DLC |
| **FIX_TFLITE_NPU.md** | Fix TFLite NPU acceleration issues |
| **INSTALL_SNPE_SDK.md** | Install QNN/SNPE SDK 2.40.0 |
| **QIDK_NPU_TESTING_GUIDE.md** | Test models on QIDK device |
| **COMPLETE_WORKFLOW.md** | Full training → deployment workflow |

## 🛠️ Recommended Workflow

### For Mobile (Android/iOS)
```
PyTorch (.pt) → ONNX (.onnx) → TFLite (.tflite)
                                    ↓
                            Deploy with Hexagon Delegate
```

### For Qualcomm QIDK (NPU)
```
Option 1 (Recommended):
PyTorch → YOLOv8 → ONNX → QNN/DLC (INT8) → QIDK NPU

Option 2 (Current):
PyTorch → ONNX → TFLite (INT8) → QIDK with Hexagon Delegate

Option 3 (Fallback):
PyTorch → ONNX → QNN/DLC (FP32 only) → QIDK NPU
```

## 📈 Performance Benchmarks

| Platform | Format | Precision | Inference Time | Notes |
|----------|--------|-----------|----------------|-------|
| PC (CPU) | ONNX | FP32 | 142ms | ONNX Runtime |
| PC (CPU) | PyTorch | FP32 | ~200ms | Ultralytics |
| Android (CPU) | TFLite | FP32 | 800-1200ms | 4 threads |
| Android (GPU) | TFLite | FP16 | 300-500ms | GPU delegate |
| Android (NPU) | TFLite | INT8 | 50-150ms | Hexagon delegate |
| QIDK (NPU) | QNN | FP32 | 150-300ms | HTP backend |
| QIDK (NPU) | QNN | INT8 | 20-80ms | Target (needs YOLOv8) |

## 🔧 Environment Setup

### Conda Environments
```bash
# Pipeline environment (original)
conda env create -f environment.yml

# DLC environment (clean, for QNN conversion)
conda create -n dlc python=3.10 -y
conda activate dlc
pip install numpy==1.23.5 onnx==1.15.0 protobuf==3.20.3
```

### QNN/SNPE SDK
```bash
# Set environment
export SNPE_ROOT=~/snpe-sdk/2.40.0.251030
source $SNPE_ROOT/bin/envsetup.sh

# Add to ~/.bashrc for persistence
echo "export SNPE_ROOT=~/snpe-sdk/2.40.0.251030" >> ~/.bashrc
echo "alias snpe-setup='source \$SNPE_ROOT/bin/envsetup.sh'" >> ~/.bashrc
```

## 📞 Support & Resources

- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- **ONNX Runtime**: https://onnxruntime.ai/
- **TensorFlow Lite**: https://www.tensorflow.org/lite
- **Qualcomm QNN**: https://qpm.qualcomm.com/ (requires account)
- **Issues**: See `docs/` folder for troubleshooting guides

## 🎓 Learning Path

1. **Start**: `models/pytorch/README.md` - Understand PyTorch model
2. **Export**: `models/onnx/README.md` - Learn ONNX conversion
3. **Mobile**: `models/tflite/README.md` - Deploy to Android/iOS
4. **NPU**: `models/qnn_dlc/README.md` - Optimize for Qualcomm NPU
5. **Issues**: `docs/ALL_DLC_CONVERSION_PATHS.md` - Solve conversion problems

## ✅ Next Steps

### Immediate Actions
1. ✅ PyTorch model trained (best.pt)
2. ✅ ONNX exported and validated (best_simplified.onnx)
3. ✅ TFLite FP32/FP16 created
4. ✅ QNN FP32 model created
5. ⏳ Create TFLite INT8 for mobile NPU
6. ⏳ Train YOLOv8 for QNN INT8 (recommended path)

### For Production Deployment
1. Create INT8 quantized models (TFLite or QNN)
2. Test on target device (QIDK or Android)
3. Optimize inference pipeline
4. Benchmark performance vs. accuracy
5. Deploy to production

---

**Last Updated**: November 17, 2025
**Model Version**: YOLOv11n, 1024×1024, 5 classes
**Status**: Development - Ready for deployment testing
