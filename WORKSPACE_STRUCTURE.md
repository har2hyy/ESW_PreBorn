# Workspace Structure - Production Ready
**Last Updated:** November 16, 2025

## Clean Production Structure

```
ESW_PreBorn/
├── YOLO/                               # YOLO Detection & NPU Deployment
│   ├── optimized_yolo_onnx.py         # ✅ Main ONNX detector (imgsz=1024)
│   ├── validate_onnx_for_npu.py       # ✅ ONNX validation (11 detections confirmed)
│   ├── prepare_npu_input.py           # ✅ Convert images to .raw for NPU
│   ├── decode_npu_output.py           # ✅ Decode NPU output tensors
│   ├── train.py                       # ✅ Training script
│   ├── export_int8.py                 # ✅ INT8 quantization
│   ├── classes.txt                    # Class names
│   ├── calibration_list.txt           # 150 calibration images
│   ├── DLC_CONVERSION_GUIDE.md        # 📖 Complete DLC conversion guide
│   ├── NPU_DEPLOYMENT_CHECKLIST.md    # 📖 Quick-start checklist
│   └── runs/detect/train/weights/
│       ├── best.pt                    # PyTorch weights (imgsz=1024)
│       ├── best.onnx                  # ✅ ONNX model (validated)
│       └── best_simplified.onnx       # ✅ Optimized for SNPE
│
├── Pipeline/                           # Integrated Detection + Depth Pipelines
│   ├── integrated_pipeline_onnx.py    # ✅ Production ONNX pipeline
│   ├── integrated_pipeline_pytorch.py # ✅ PyTorch reference pipeline
│   ├── DEPTH_ANALYSIS_COMPLETE_GUIDE.md
│   ├── IMPLEMENTATION_SUGGESTIONS.md
│   ├── OUTPUT_FILES_GUIDE.md
│   ├── USAGE_GUIDE.md
│   └── README.md
│
├── Depth-Anything-V2/                  # Depth Estimation Module
│   ├── depth_anything_v2/
│   ├── checkpoints/
│   │   ├── depth_anything_v2_vits.pth
│   │   └── depth_anything_v2_vitb.pth
│   └── ...
│
├── CLEANUP_COMPLETE.md                 # 📋 Cleanup summary
├── REDUNDANT_FILES_ANALYSIS.md         # 📋 Analysis of deleted files
└── WORKSPACE_STRUCTURE.md              # 📋 This file
```

## Quick Reference

### 🎯 For ONNX Validation
```bash
cd YOLO
python validate_onnx_for_npu.py
```
**Output:** 11 detections (9 workers, 2 trucks) ✅

### 🚀 For Full Pipeline
```bash
cd Pipeline
python integrated_pipeline_onnx.py
```
**Output:** Detection + depth + distance analysis

### �� For NPU Deployment
```bash
cd YOLO
# Step 1: Prepare input
python prepare_npu_input.py /path/to/image.jpg input.raw

# Step 2: Convert ONNX → DLC (see DLC_CONVERSION_GUIDE.md)
snpe-onnx-to-dlc --input_network best_simplified.onnx ...

# Step 3: Decode NPU output
python decode_npu_output.py output0.raw /path/to/image.jpg
```

## File Counts

### Before Cleanup
- YOLO: 22 Python files
- Pipeline: 7 Python files
- **Total:** 29 files

### After Cleanup
- YOLO: 6 production Python files ✅
- Pipeline: 2 production Python files ✅
- **Total:** 8 files
- **Removed:** 15 redundant files
- **Renamed:** 1 file (tflite → onnx)

## Production Workflow

1. **Training** → `train.py` (completed)
2. **ONNX Export** → `best.onnx` (validated)
3. **Simplification** → `best_simplified.onnx` (ready)
4. **Validation** → `validate_onnx_for_npu.py` (11 detections ✅)
5. **DLC Conversion** → Follow `DLC_CONVERSION_GUIDE.md`
6. **NPU Deployment** → `prepare_npu_input.py` + `decode_npu_output.py`

## Model Specifications

- **Architecture:** YOLOv11n
- **Training Resolution:** 1024×1024
- **Classes:** 5 (worker, truck, bike, bulldozer, car)
- **ONNX Input:** [1, 3, 1024, 1024] float32
- **ONNX Output:** [1, 9, 21504] float32
- **Target Device:** Qualcomm QIDK NPU

## Next Steps

✅ **Completed:**
- Model trained and validated
- ONNX export successful
- Simplified for SNPE compatibility
- Calibration data prepared
- Validation confirmed (11 detections)
- Workspace cleaned and organized

⏳ **Pending:**
- DLC conversion (requires SNPE SDK)
- INT8 quantization
- On-device testing

**Ready for NPU deployment!** 🚀
