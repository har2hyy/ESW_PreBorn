# 🚀 QIDK NPU Testing Guide - Step by Step

## Prerequisites Checklist
- [ ] SNPE SDK installed and verified
- [ ] DLC files created (best_yolo_fp32.dlc and best_yolo_int8.dlc)
- [ ] DLC tested on PC (confirms model works)
- [ ] QIDK device powered on
- [ ] USB cable connected to PC
- [ ] ADB installed on PC

---

## 🔌 Step 1: Connect QIDK Device

### 1.1 Enable USB Debugging on QIDK
```
Settings → About Device → Tap "Build Number" 7 times → Back
Settings → Developer Options → Enable "USB Debugging"
```

### 1.2 Connect via USB
- Connect QIDK to PC using USB cable
- Ensure device is in normal mode (not fastboot/recovery)

### 1.3 Verify ADB Connection
```bash
# Check if ADB is installed
adb version
# Should show: Android Debug Bridge version 1.0.xx

# List connected devices
adb devices
# Should show:
# List of devices attached
# XXXXXXXX    device

# If shows "unauthorized", check QIDK screen and approve USB debugging
```

### 1.4 Troubleshooting ADB
```bash
# If device not detected:
sudo apt-get update
sudo apt-get install android-tools-adb android-tools-fastboot

# Restart ADB server
adb kill-server
adb start-server
adb devices

# Check USB connection
lsusb | grep -i qualcomm
```

---

## 🛠️ Step 2: Setup QIDK Device (One-Time)

### 2.1 Run Automated Setup
```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO

# Run setup script
./setup_qidk_device.sh
```

**What this script does:**
1. ✅ Checks ADB connection
2. ✅ Creates `/data/local/tmp/yolo_npu/` directory on device
3. ✅ Pushes SNPE runtime libraries (libSNPE.so, libhta.so, etc.)
4. ✅ Pushes snpe-net-run binary
5. ✅ Sets executable permissions

### 2.2 Manual Setup (if script fails)

```bash
# Create directories on device
adb shell "mkdir -p /data/local/tmp/yolo_npu"

# Push SNPE libraries
adb push $SNPE_ROOT/lib/aarch64-android/libSNPE.so /data/local/tmp/yolo_npu/
adb push $SNPE_ROOT/lib/aarch64-android/libhta.so /data/local/tmp/yolo_npu/
adb push $SNPE_ROOT/lib/dsp/libsnpehtpv75_skel.so /data/local/tmp/yolo_npu/

# Push snpe-net-run binary
adb push $SNPE_ROOT/bin/aarch64-android/snpe-net-run /data/local/tmp/yolo_npu/

# Set permissions
adb shell "chmod +x /data/local/tmp/yolo_npu/snpe-net-run"
adb shell "chmod 644 /data/local/tmp/yolo_npu/*.so"
```

### 2.3 Verify Setup
```bash
# Check files on device
adb shell "ls -lh /data/local/tmp/yolo_npu/"
# Should show: snpe-net-run, libSNPE.so, libhta.so, etc.

# Test snpe-net-run
adb shell "cd /data/local/tmp/yolo_npu && LD_LIBRARY_PATH=. ./snpe-net-run --help"
# Should show snpe-net-run help message
```

---

## 🎯 Step 3: Run Inference on NPU

### Option A: Automated Remote Pipeline (Recommended ⭐)

```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO

# Run full pipeline from PC
./remote_npu_pipeline.py \
    --image /path/to/test/image.jpg \
    --dlc best_yolo_int8.dlc \
    --runtime dsp

# Example with actual file:
./remote_npu_pipeline.py \
    --image val/images/IMG_20240916_112859.jpg \
    --dlc best_yolo_int8.dlc \
    --runtime dsp
```

**What this does:**
1. ✅ Preprocesses image on PC (resize to 1024x1024)
2. ✅ Pushes DLC and input.raw to QIDK via ADB
3. ✅ Runs inference on QIDK NPU using DSP runtime
4. ✅ Pulls output.raw from QIDK
5. ✅ Postprocesses on PC (NMS, visualization)
6. ✅ Saves `remote_npu_result.jpg` and `remote_npu_result.json`

**Runtime Options:**
- `--runtime dsp` → NPU/HTP (50-100ms) ⚡ **RECOMMENDED**
- `--runtime gpu` → GPU on QIDK (150-200ms)
- `--runtime cpu` → CPU on QIDK (300-500ms)

### Option B: Manual Step-by-Step

#### 3.1 Prepare Input on PC
```bash
# Create input.raw from image using Python
python3 << EOF
import cv2
import numpy as np

# Load and preprocess
img = cv2.imread('val/images/IMG_20240916_112859.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (1024, 1024))
img_normalized = img_resized.astype(np.float32) / 255.0
img_chw = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW

# Save as raw
img_chw.tofile('input.raw')
print(f"✅ Created input.raw: {img_chw.shape} = {img_chw.nbytes} bytes")
EOF
```

#### 3.2 Push Files to Device
```bash
# Push DLC model
adb push best_yolo_int8.dlc /data/local/tmp/yolo_npu/

# Push input
adb push input.raw /data/local/tmp/yolo_npu/
```

#### 3.3 Run Inference on Device
```bash
# Run on NPU (DSP runtime)
adb shell "cd /data/local/tmp/yolo_npu && \
    LD_LIBRARY_PATH=. ./snpe-net-run \
    --container best_yolo_int8.dlc \
    --input_raw input.raw \
    --use_dsp"

# Should show:
# Processing DNN Input(s):
# input.raw
# Successfully executed!
```

#### 3.4 Pull Results from Device
```bash
# Pull output tensor
adb pull /data/local/tmp/yolo_npu/output/Result_0/output0.raw ./output0.raw
adb pull /data/local/tmp/yolo_npu/output/Result_0/output1.raw ./output1.raw
adb pull /data/local/tmp/yolo_npu/output/Result_0/output2.raw ./output2.raw
```

#### 3.5 Postprocess on PC
```bash
python3 << 'EOF'
import numpy as np
import cv2

# Load outputs
out0 = np.fromfile('output0.raw', dtype=np.float32).reshape(1, 80, 80, 64)
out1 = np.fromfile('output1.raw', dtype=np.float32).reshape(1, 40, 40, 64)
out2 = np.fromfile('output2.raw', dtype=np.float32).reshape(1, 20, 20, 64)

print(f"✅ Loaded outputs: {out0.shape}, {out1.shape}, {out2.shape}")

# Continue with postprocessing (NMS, visualization)
# See remote_npu_pipeline.py for full implementation
EOF
```

---

## 📊 Step 4: Validate Results

### 4.1 Expected Results
- **Detections**: 9-11 objects (workers, trucks, bikes, bulldozers, cars)
- **Inference Time**: 
  - NPU (DSP): **50-100ms** ⚡
  - GPU: 150-200ms
  - CPU: 300-500ms
- **Accuracy**: Should match ONNX/DLC PC results (~95-99%)

### 4.2 Check Output Files
```bash
# Automated pipeline output
ls -lh remote_npu_result.*
# remote_npu_result.jpg  - Visualization with bounding boxes
# remote_npu_result.json - Detection data (bbox, conf, class)

# View results
cat remote_npu_result.json
# [{"class": "worker", "confidence": 0.92, "bbox": [x1, y1, x2, y2]}, ...]
```

### 4.3 Performance Monitoring
```bash
# Monitor device during inference
adb shell "top -n 1 | grep snpe"

# Check thermal status
adb shell "cat /sys/class/thermal/thermal_zone0/temp"

# Check battery (if applicable)
adb shell "dumpsys battery"
```

---

## 🔍 Step 5: Troubleshooting

### Issue: "adb: device not found"
**Solution:**
```bash
adb kill-server
adb start-server
adb devices

# Check USB connection
lsusb | grep Qualcomm

# Try different USB port or cable
```

### Issue: "snpe-net-run: not found"
**Solution:**
```bash
# Re-run setup
./setup_qidk_device.sh

# Or manually push binary
adb push $SNPE_ROOT/bin/aarch64-android/snpe-net-run /data/local/tmp/yolo_npu/
adb shell "chmod +x /data/local/tmp/yolo_npu/snpe-net-run"
```

### Issue: "error while loading shared libraries"
**Solution:**
```bash
# Always set LD_LIBRARY_PATH when running snpe-net-run
adb shell "cd /data/local/tmp/yolo_npu && LD_LIBRARY_PATH=. ./snpe-net-run ..."

# Or push all required .so files
adb push $SNPE_ROOT/lib/aarch64-android/*.so /data/local/tmp/yolo_npu/
```

### Issue: "Failed to load network"
**Solution:**
```bash
# Verify DLC is valid
snpe-dlc-info -i best_yolo_int8.dlc

# Re-push DLC to device
adb push best_yolo_int8.dlc /data/local/tmp/yolo_npu/

# Check file integrity
adb shell "ls -lh /data/local/tmp/yolo_npu/best_yolo_int8.dlc"
```

### Issue: "DSP runtime not available"
**Solution:**
```bash
# Check available runtimes on device
adb shell "cd /data/local/tmp/yolo_npu && LD_LIBRARY_PATH=. ./snpe-net-run --help"

# Try GPU runtime instead
./remote_npu_pipeline.py --image test.jpg --dlc best_yolo_int8.dlc --runtime gpu

# Check if HTP libraries are present
adb shell "ls -l /data/local/tmp/yolo_npu/libhta*"
```

### Issue: Low accuracy or wrong detections
**Solution:**
```bash
# Test with FP32 DLC first
./remote_npu_pipeline.py --image test.jpg --dlc best_yolo_fp32.dlc --runtime dsp

# Compare with PC results
./test_dlc_on_pc.py --dlc best_yolo_int8.dlc --image test.jpg

# Check calibration dataset quality
head -20 calibration_list.txt
```

### Issue: Slow inference (>200ms on NPU)
**Solution:**
```bash
# Ensure using DSP runtime (not CPU)
./remote_npu_pipeline.py --runtime dsp  # NOT cpu or gpu

# Check device temperature
adb shell "cat /sys/class/thermal/thermal_zone*/temp"

# Reboot device if thermal throttling
adb reboot

# Monitor CPU frequency
adb shell "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"
```

---

## 📈 Step 6: Batch Processing (Production)

### Process Multiple Images
```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO

# Create batch processing script
cat > batch_npu_inference.sh << 'EOF'
#!/bin/bash
IMAGE_DIR="val/images"
OUTPUT_DIR="npu_batch_results"
mkdir -p $OUTPUT_DIR

for img in $IMAGE_DIR/*.jpg; do
    filename=$(basename "$img" .jpg)
    echo "Processing $filename..."
    
    ./remote_npu_pipeline.py \
        --image "$img" \
        --dlc best_yolo_int8.dlc \
        --runtime dsp \
        --output "$OUTPUT_DIR/${filename}_result.jpg"
done

echo "✅ Batch processing complete! Results in $OUTPUT_DIR/"
EOF

chmod +x batch_npu_inference.sh
./batch_npu_inference.sh
```

### Performance Logging
```bash
# Log inference times
./remote_npu_pipeline.py --image test.jpg --dlc best_yolo_int8.dlc --runtime dsp 2>&1 | tee npu_performance.log

# Extract timing
grep -E "(Inference time|Total time)" npu_performance.log
```

---

## ✅ Quick Validation Checklist

Before closing, verify:
- [ ] QIDK connected: `adb devices` shows device
- [ ] Setup complete: `adb shell "ls /data/local/tmp/yolo_npu/"` shows files
- [ ] DLC pushed: `adb shell "ls -lh /data/local/tmp/yolo_npu/*.dlc"`
- [ ] NPU inference works: `./remote_npu_pipeline.py` completes successfully
- [ ] Results valid: 9-11 detections in output JSON
- [ ] Performance good: Inference < 100ms on DSP
- [ ] Visualization correct: Bounding boxes align with objects

---

## 🎯 Summary Commands

```bash
# 1. Connect QIDK
adb devices

# 2. Setup device (one-time)
./setup_qidk_device.sh

# 3. Run inference
./remote_npu_pipeline.py --image test.jpg --dlc best_yolo_int8.dlc --runtime dsp

# 4. Check results
cat remote_npu_result.json
xdg-open remote_npu_result.jpg
```

---

## 📚 Additional Resources

- **SNPE Documentation**: https://developer.qualcomm.com/docs/snpe/
- **QNN Documentation**: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html
- **QIDK User Guide**: Check your device documentation
- **ADB Reference**: https://developer.android.com/studio/command-line/adb

---

**Ready to test! Connect your QIDK and follow steps 1-3.** 🚀
