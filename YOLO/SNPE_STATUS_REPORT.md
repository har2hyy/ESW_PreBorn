# 📊 SNPE Status Report - PC & QIDK

## ✅ **What You Already Have**

### **On PC:**
```
Location: /home/harshyy/snpe-sdk/2.40.0.251030/
```

✅ **SNPE SDK 2.40.0.251030** - Fully installed
✅ **Android ARM64 executables:**
  - `bin/aarch64-android/snpe-net-run` (1.1MB)
  - `bin/aarch64-android/qnn-net-run` (4.0MB) ⭐
  - Plus 9 more SNPE/QNN tools

✅ **Android ARM64 libraries (26 QNN libraries):**
  - `lib/aarch64-android/libQnnHtp.so` - Hexagon Tensor Processor (NPU)
  - `lib/aarch64-android/libQnnDsp.so` - DSP runtime
  - `lib/aarch64-android/libQnnGpu.so` - GPU runtime
  - `lib/aarch64-android/libQnnCpu.so` - CPU runtime
  - `lib/aarch64-android/libSNPE.so` (18MB) - Core SNPE library
  - Plus 21 more support libraries

### **On QIDK Device:**
```
Location: /data/local/tmp/snpe_benchmark/
```

✅ **SNPE runtime executable:**
  - `snpe-net-run` (1.0MB) - Working (tested with --help)

✅ **Core library:**
  - `libSNPE.so` (18MB) - Present

❌ **Missing QNN runtime libraries:**
  - libQnnHtp.so - **NOT on device** (needed for NPU)
  - libQnnDsp.so - **NOT on device** (needed for DSP)
  - libQnnGpu.so - **NOT on device** (needed for GPU)

✅ **Existing DLC models:**
  - `model_fp32.dlc` (10MB)
  - `model_fp16.dlc` (5.1MB)
  - **Issue:** DLC read failure - may be corrupted or wrong version

✅ **Test inputs:**
  - `inputs/test_input.raw` (12MB)
  - `input_list.txt` - Configured

---

## 🔍 **Current Issue**

When trying to run SNPE with DSP:
```bash
error_code=310; error_message=Dlc read failure. 
Failed to initialize the DLC reader
```

**Root Causes:**
1. ❌ **Missing QNN libraries** - No libQnnDsp.so, libQnnHtp.so on device
2. ⚠️ **DLC model issue** - Existing model_fp32.dlc has read errors
3. ❌ **No INT8 quantized model** - Need to convert best_300_int8.tflite → DLC

---

## ✅ **What Needs to Be Done**

### **Step 1: Push Missing QNN Libraries** ⭐

These libraries are **essential** for NPU/DSP/GPU to work:

```bash
# Push QNN HTP (Hexagon NPU) - MOST IMPORTANT
adb push /home/harshyy/snpe-sdk/2.40.0.251030/lib/aarch64-android/libQnnHtp.so \
  /data/local/tmp/snpe_benchmark/

# Push QNN DSP runtime
adb push /home/harshyy/snpe-sdk/2.40.0.251030/lib/aarch64-android/libQnnDsp.so \
  /data/local/tmp/snpe_benchmark/

# Push QNN GPU runtime
adb push /home/harshyy/snpe-sdk/2.40.0.251030/lib/aarch64-android/libQnnGpu.so \
  /data/local/tmp/snpe_benchmark/

# Push supporting libraries
adb push /home/harshyy/snpe-sdk/2.40.0.251030/lib/aarch64-android/libQnnHtaNetRunExtensions.so \
  /data/local/tmp/snpe_benchmark/
adb push /home/harshyy/snpe-sdk/2.40.0.251030/lib/aarch64-android/libQnnDspNetRunExtensions.so \
  /data/local/tmp/snpe_benchmark/
adb push /home/harshyy/snpe-sdk/2.40.0.251030/lib/aarch64-android/libQnnGpuNetRunExtensions.so \
  /data/local/tmp/snpe_benchmark/
```

**Why needed:** Without these, SNPE can ONLY run on CPU (no NPU/DSP/GPU access)

---

### **Step 2: Convert INT8 TFLite to DLC**

Convert your optimized INT8 model to SNPE format:

```bash
# On PC (requires snpe310 environment for conversion)
conda activate snpe310

snpe-tflite-to-dlc \
  --input_network models/300/best_300_int8.tflite \
  --output_path models/300/best_300_int8.dlc \
  --input_dim images 1,1024,1024,3 \
  --quantization_overrides quantization_overrides.json \
  --enable_quantization
```

**Result:** INT8 quantized DLC model optimized for DSP/NPU

---

### **Step 3: Push INT8 DLC Model to Device**

```bash
adb push models/300/best_300_int8.dlc /data/local/tmp/snpe_benchmark/
```

---

### **Step 4: Create Test Input**

Generate raw input file for SNPE (1024x1024x3 uint8):

```python
import numpy as np
from PIL import Image

# Load test image
img = Image.open('test_images/test_detection_v2.jpg')
img = img.resize((1024, 1024))
img_array = np.array(img, dtype=np.uint8)

# Save as raw binary
img_array.tofile('snpe_test_input.raw')
```

Push to device:
```bash
adb push snpe_test_input.raw /data/local/tmp/snpe_benchmark/inputs/yolo_input.raw
```

Update input_list.txt:
```bash
adb shell "echo 'images:=inputs/yolo_input.raw' > /data/local/tmp/snpe_benchmark/yolo_input_list.txt"
```

---

### **Step 5: Run Benchmark on CPU, GPU, DSP/NPU**

#### **CPU (Baseline):**
```bash
adb shell "cd /data/local/tmp/snpe_benchmark && \
  LD_LIBRARY_PATH=. ./snpe-net-run \
  --container best_300_int8.dlc \
  --input_list yolo_input_list.txt"
```

#### **GPU (Adreno 750):**
```bash
adb shell "cd /data/local/tmp/snpe_benchmark && \
  LD_LIBRARY_PATH=. ./snpe-net-run \
  --container best_300_int8.dlc \
  --input_list yolo_input_list.txt \
  --use_gpu"
```

#### **DSP/NPU (Hexagon) ⚡:**
```bash
adb shell "cd /data/local/tmp/snpe_benchmark && \
  LD_LIBRARY_PATH=. ./snpe-net-run \
  --container best_300_int8.dlc \
  --input_list yolo_input_list.txt \
  --use_dsp"
```

---

## 🎯 **Expected Performance (Once Libraries Are Pushed)**

| Runtime | Expected Time | FPS | Status |
|---------|---------------|-----|--------|
| CPU | 60-80 ms | 12-16 | ✅ Should work now |
| GPU (Adreno) | 30-40 ms | 25-33 | 🔧 After pushing libQnnGpu.so |
| DSP/NPU (Hexagon) | **20-30 ms** | **33-50** | 🔧 After pushing libQnnHtp.so ⚡ |

---

## 🚀 **Quick Action Plan**

**I can create a script to do ALL of this automatically!**

The script will:
1. ✅ Push all required QNN libraries to device
2. ✅ Convert INT8 TFLite → DLC (if conversion works)
3. ✅ Create and push test inputs
4. ✅ Run benchmarks on CPU, GPU, DSP/NPU
5. ✅ Generate performance comparison graphs
6. ✅ Save results

**Want me to create this automated script?**

---

## 📋 **Files Summary**

### **Already on PC (Ready to Push):**
- ✅ `/home/harshyy/snpe-sdk/2.40.0.251030/lib/aarch64-android/*.so` (26 libraries)
- ✅ `/home/harshyy/snpe-sdk/2.40.0.251030/bin/aarch64-android/qnn-net-run`
- ✅ `models/300/best_300_int8.tflite` (2.9MB INT8 model)

### **Already on Device:**
- ✅ `/data/local/tmp/snpe_benchmark/snpe-net-run`
- ✅ `/data/local/tmp/snpe_benchmark/libSNPE.so`
- ⚠️ `/data/local/tmp/snpe_benchmark/model_fp32.dlc` (corrupted/incompatible)

### **Needs to Be Pushed:**
- 🔧 QNN runtime libraries (libQnnHtp.so, libQnnDsp.so, libQnnGpu.so)
- 🔧 INT8 DLC model (after conversion)
- 🔧 Test input file

---

## ✅ **Bottom Line**

You have:
- ✅ SNPE SDK on PC (complete)
- ✅ SNPE runtime on device (partial)
- ❌ Missing QNN libraries (critical for NPU/GPU)

**Solution:** Push 6-10 missing libraries (~50MB total) → Get NPU working!

**Want me to create the automated setup script now?**
