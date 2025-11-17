# NPU Deployment Quick Start
# ============================

This checklist guides you through the complete NPU deployment workflow.

## ✅ Pre-Deployment Checklist

### 1. Files Ready
- [x] `runs/detect/train/weights/best_simplified.onnx` - Optimized ONNX model
- [x] `calibration_list.txt` - 150 calibration images
- [x] `validate_onnx_for_npu.py` - ONNX validation (11 detections confirmed)
- [x] `prepare_npu_input.py` - Raw input converter
- [x] `decode_npu_output.py` - NPU output decoder

### 2. Environment Setup
- [ ] SNPE/QNN SDK installed
- [ ] Environment variables configured (`SNPE_ROOT`, `PATH`, `LD_LIBRARY_PATH`)
- [ ] Test: `snpe-onnx-to-dlc --help` works

### 3. Device Ready
- [ ] QIDK device connected via ADB
- [ ] Test: `adb devices` shows device
- [ ] SNPE runtime libraries on device

---

## 🚀 Quick Start Commands

### Step 1: Convert ONNX → DLC (FP32)
```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO

snpe-onnx-to-dlc \
  --input_network runs/detect/train/weights/best_simplified.onnx \
  --output_path runs/detect/train/weights/best_yolo_fp32.dlc \
  --input_dim images 1,3,1024,1024
```

### Step 2: Inspect DLC
```bash
snpe-dlc-info --input_dlc runs/detect/train/weights/best_yolo_fp32.dlc
```

### Step 3: Quantize to INT8
```bash
snpe-dlc-quantize \
  --input_dlc runs/detect/train/weights/best_yolo_fp32.dlc \
  --output_dlc runs/detect/train/weights/best_yolo_int8.dlc \
  --input_list calibration_list.txt \
  --enable_htp \
  --use_enhanced_quantizer
```

### Step 4: Prepare Test Input
```bash
python prepare_npu_input.py /home/harshyy/Desktop/20250103_104457.jpg test_input.raw
```

### Step 5: Deploy to Device
```bash
# Push files
adb shell mkdir -p /data/local/tmp/yolo_npu
adb push runs/detect/train/weights/best_yolo_int8.dlc /data/local/tmp/yolo_npu/
adb push test_input.raw /data/local/tmp/yolo_npu/

# Run on NPU
adb shell "cd /data/local/tmp/yolo_npu && snpe-net-run \
  --container best_yolo_int8.dlc \
  --input_raw test_input.raw \
  --use_dsp \
  --output_dir output/"

# Retrieve results
adb pull /data/local/tmp/yolo_npu/output/Result_0/output0.raw .
```

### Step 6: Decode Results
```bash
python decode_npu_output.py output0.raw /home/harshyy/Desktop/20250103_104457.jpg
```

**Expected Output**: 9-11 detections (same as ONNX validation)

---

## 📊 Validation Results

### ONNX Baseline (CPU)
- Model: `best_simplified.onnx`
- Detections: 11 (9 workers + 2 trucks)
- Inference: ~252 ms
- Status: ✅ Validated

### INT8 DLC Target (NPU)
- Model: `best_yolo_int8.dlc`
- Expected Detections: 9-11 (±1 due to quantization)
- Expected Inference: ~50-100 ms (2-4x faster)
- Expected Accuracy: 95-99% of FP32

---

## 🔧 Troubleshooting

### Issue: `snpe-onnx-to-dlc: command not found`
**Solution**: Set SNPE environment variables
```bash
export SNPE_ROOT=/opt/qcom/aistack/snpe-2.x.x.xxxx
export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH
```

### Issue: Unsupported ONNX operations
**Solution**: Check SNPE compatibility
```bash
snpe-dlc-info --input_dlc best_yolo_fp32.dlc
```
Look for any "custom" layers. If found, re-export ONNX with simpler ops.

### Issue: Low accuracy after quantization
**Solution**: 
- Increase calibration dataset (200+ images)
- Try per-channel quantization: `--use_per_channel_quantization`
- Check calibration images are representative

### Issue: NPU execution fails
**Solution**:
```bash
# Validate device NPU
adb shell snpe-platform-validator

# Try GPU fallback
snpe-net-run ... --use_gpu
```

---

## 📁 Output Files

After successful deployment:

```
YOLO/
├── runs/detect/train/weights/
│   ├── best_simplified.onnx      # Optimized ONNX (10.2 MB)
│   ├── best_yolo_fp32.dlc        # FP32 DLC (~10 MB)
│   └── best_yolo_int8.dlc        # INT8 DLC (~2-3 MB) ⭐
├── test_input.raw                # Preprocessed input (12 MB)
├── output0.raw                   # NPU output (753 KB)
├── npu_result.jpg                # Visualization
└── npu_result.json               # Detection results
```

---

## 🎯 Success Criteria

- [x] ONNX model validated: 11 detections
- [ ] FP32 DLC created successfully
- [ ] INT8 DLC quantized with 150 calibration images
- [ ] NPU execution completes without errors
- [ ] Detection count: 9-11 objects (within ±1 of ONNX)
- [ ] Inference time: <100 ms on NPU
- [ ] Visual validation: Correct bboxes on workers/trucks

---

## 📞 Support

For detailed instructions, see:
- `DLC_CONVERSION_GUIDE.md` - Complete step-by-step guide
- `validate_onnx_for_npu.py` - Reference ONNX implementation
- Qualcomm SNPE documentation: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk

---

**Last Updated**: After ONNX validation (11 detections confirmed)
**Next Action**: Install SNPE SDK and run Step 1 (ONNX → DLC)
