# 🔍 QIDK NPU/GPU Issues - Diagnosis & Fixes

## 🚨 **Problem Identified**

Your QIDK device shows:
```
ERROR: Could not find the specified NNAPI accelerator: qti-htp. 
Must be one of: {nnapi-reference}.
```

**Root Cause:** The Qualcomm NNAPI drivers are **NOT installed/enabled** on your device.

---

## 📊 **Current Results**

From your benchmark run:

| Runtime | Time (ms) | FPS | Status |
|---------|-----------|-----|--------|
| CPU (4 threads) | 74.44 ms | 13.43 FPS | ✅ Working |
| NNAPI (Auto) | 78.61 ms | 12.72 FPS | ⚠️ Fallback to CPU |
| GPU (qti-gpu) | - | - | ❌ Not available |
| NPU (qti-htp) | - | - | ❌ Not available |

**Issue:** NNAPI is actually **slower** than direct CPU (78.61 vs 74.44 ms) because it's using the reference implementation (software fallback).

---

## 🔍 **Why This Happens**

### Missing Components:
1. **Qualcomm NNAPI Driver** - Not installed/enabled
2. **QNN (Qualcomm Neural Network) SDK** - Not on device
3. **Hexagon NN Libraries** - Missing from Android system
4. **GPU Compute Libraries** - Adreno drivers not exposed via NNAPI

### Device Configuration Issues:
- QIDK might be running stock Android (no Qualcomm optimizations)
- NNAPI HAL (Hardware Abstraction Layer) not configured
- SELinux policies blocking accelerator access
- Device needs vendor-specific firmware/libraries

---

## ✅ **Solutions (Ranked by Ease)**

### **Solution 1: Use SNPE SDK Instead of NNAPI** ⭐ (RECOMMENDED)

NNAPI is a generic Android API - SNPE is **Qualcomm's native runtime** for NPU/DSP.

#### Why SNPE is Better:
- ✅ Direct access to Hexagon DSP/NPU (bypasses NNAPI)
- ✅ Optimized for Qualcomm hardware
- ✅ Supports quantized models natively
- ✅ You already have SNPE SDK installed (`/home/harshyy/snpe-sdk/2.40.0.251030`)

#### Steps:
1. Convert TFLite → DLC (SNPE format)
2. Push DLC to device
3. Run with `snpe-net-run` on device
4. Get direct NPU measurements

**I can create a script to do this - want me to?**

---

### **Solution 2: Install Qualcomm NNAPI Drivers on Device**

Your device needs Qualcomm's vendor implementation of NNAPI.

#### Check Current NNAPI Support:
```bash
# Check NNAPI implementation
adb shell getprop ro.nnapi.extensions.deny_on_product

# List NNAPI libraries
adb shell ls -la /vendor/lib*/libneural*
adb shell ls -la /vendor/lib*/libhexagon*
adb shell ls -la /system/lib*/libneural*

# Check QNN libraries
adb shell ls -la /vendor/lib*/libQnn*
```

#### If Missing, You Need:
```bash
# These should exist on device for NNAPI to work:
/vendor/lib64/libQnnHtp.so          # Hexagon NPU
/vendor/lib64/libQnnGpu.so          # Adreno GPU  
/vendor/lib64/libneuralnetworks.so  # NNAPI implementation
/vendor/lib64/libhexagon_nn_skel.so # DSP runtime
```

#### Installation (Requires Root):
1. Root your QIDK device
2. Install Qualcomm vendor image with NNAPI support
3. Flash firmware with NPU drivers enabled
4. Reboot and verify

**This is complex and may void warranty!**

---

### **Solution 3: Try Generic NNAPI with --use_nnapi** (Already Tried)

You already tried this - it falls back to CPU reference implementation.

Result: **78.61 ms** (slower than direct CPU)

---

### **Solution 4: Enable QNN Runtime on Device**

QNN (Qualcomm Neural Network) is newer than NNAPI.

#### Check if QNN is available:
```bash
# Look for QNN libraries
adb shell find /vendor -name "*Qnn*" 2>/dev/null
adb shell find /system -name "*Qnn*" 2>/dev/null

# Check for HTP (Hexagon Tensor Processor) support
adb shell cat /sys/devices/soc0/soc_id
```

#### If QNN exists:
You can use QNN runtime directly (bypassing NNAPI):
- Push QNN libraries to device
- Use `qnn-net-run` command
- Run model on HTP/DSP

---

### **Solution 5: Use TFLite with XNNPack Delegate** (CPU Optimization)

Since NPU/GPU aren't available, optimize CPU performance:

```bash
adb shell "/data/local/tmp/qidk_benchmark/benchmark_model \
  --graph=/data/local/tmp/qidk_benchmark/model.tflite \
  --use_xnnpack=true \
  --num_threads=8"
```

Expected: **40-60 ms** (2x faster than current 74.44 ms)

---

## 🎯 **Recommended Next Steps**

### **Option A: Use SNPE (Best for NPU Access)** ⭐

Since NNAPI isn't working, use SNPE directly:

1. **Convert TFLite to DLC:**
   ```bash
   # I can create a conversion script
   python convert_tflite_to_dlc.py
   ```

2. **Push DLC to device:**
   ```bash
   adb push model.dlc /data/local/tmp/
   adb push $SNPE_ROOT/lib/aarch64-android/* /data/local/tmp/lib/
   ```

3. **Run on NPU:**
   ```bash
   adb shell "cd /data/local/tmp && \
     LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH \
     ./snpe-net-run \
     --container model.dlc \
     --runtime dsp"  # or 'gpu' or 'cpu'
   ```

**This will give you REAL NPU measurements!**

---

### **Option B: Optimize CPU Performance**

If NPU isn't critical, maximize CPU speed:

```bash
# Try with more threads and XNNPack
adb shell "/data/local/tmp/qidk_benchmark/benchmark_model \
  --graph=/data/local/tmp/qidk_benchmark/model.tflite \
  --num_threads=8 \
  --use_xnnpack=true"
```

Expected: **50-60 ms** (real-time at 16-20 FPS)

---

### **Option C: Verify Device Capabilities**

Let me check what your device actually supports:

```bash
# Run device info script
python check_qidk_capabilities.py
```

This will tell us:
- ✓ CPU cores and architecture
- ✓ Available NNAPI accelerators (we know: only nnapi-reference)
- ✓ GPU info (Adreno version)
- ✓ DSP/NPU libraries present
- ✓ QNN/SNPE runtime support

---

## 🔧 **Quick Diagnostic Commands**

Run these to understand your device:

```bash
# 1. Check NNAPI accelerators (we already know: nnapi-reference only)
adb shell "/data/local/tmp/qidk_benchmark/benchmark_model \
  --graph=/data/local/tmp/qidk_benchmark/model.tflite \
  --nnapi_accelerator_name=?"

# 2. List neural network libraries
adb shell "find /vendor /system -name '*neural*' -o -name '*hexagon*' -o -name '*Qnn*' 2>/dev/null"

# 3. Check CPU info
adb shell cat /proc/cpuinfo | grep -E "processor|CPU|Hardware"

# 4. Check Android version (newer = better NNAPI support)
adb shell getprop ro.build.version.release

# 5. Check Qualcomm chip
adb shell getprop ro.board.platform
```

---

## 🎬 **What I Recommend NOW**

### Immediate Action:

**1. Let me create a SNPE-based benchmark script:**
   - Converts TFLite → DLC
   - Pushes SNPE runtime to device
   - Runs on CPU, GPU, DSP/NPU
   - Gets actual measurements

**2. Run device capability check:**
   - See what's actually available
   - Determine if NPU is even accessible

**3. Try optimized CPU benchmark:**
   - Use XNNPack + 8 threads
   - Should get ~50-60ms

---

## 📊 **Expected Performance (Once Fixed)**

If we get SNPE working:

| Runtime | Expected Time | Status |
|---------|---------------|--------|
| CPU (XNNPack) | 50-60 ms | ✅ Should work now |
| GPU (Adreno) | 30-40 ms | 🔧 Need SNPE |
| NPU (Hexagon) | 20-30 ms | 🔧 Need SNPE |

---

## ❓ **What Do You Want to Do?**

**Option 1:** "Create SNPE benchmark script" → I'll make a script to run via SNPE
**Option 2:** "Check device capabilities first" → See what hardware is available
**Option 3:** "Optimize CPU performance" → Get best CPU-only speed
**Option 4:** "Install NNAPI drivers" → Complex, needs root access

**Which approach should we take?** 🤔
