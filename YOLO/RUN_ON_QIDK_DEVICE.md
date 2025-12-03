# 🚀 How to Actually Run and Measure on QIDK Device

## Overview
This guide shows you how to run the INT8 TFLite model **directly on the QIDK device** and get **actual measurements** from CPU, GPU, and NPU.

---

## 🎯 **Two Methods Available**

### **Method 1: TFLite Benchmark Tool** (Recommended ✅)
- Official TensorFlow benchmarking tool
- Runs directly on Android device
- Supports CPU, GPU (via NNAPI), and NPU (via NNAPI)
- **Most accurate measurements**

### **Method 2: Python Script on Device**
- Custom Python script via ADB
- More flexible but requires Python on device
- Can be harder to set up

---

## 📦 **Method 1: TFLite Benchmark Tool** (EASIEST)

### Step 1: Download the Benchmark Tool

I'll download the pre-built ARM64 binary for you:

```bash
# Run this on your PC
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO
wget https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model -O benchmark_model
chmod +x benchmark_model
```

Or build from source:
```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
bazel build -c opt --config=android_arm64 tensorflow/lite/tools/benchmark:benchmark_model
```

### Step 2: Push Everything to Device

```bash
# Push the benchmark tool
adb push benchmark_model /data/local/tmp/

# Push the INT8 model (already done, but just in case)
adb push models/300/best_300_int8.tflite /data/local/tmp/model.tflite

# Make executable
adb shell chmod +x /data/local/tmp/benchmark_model
```

### Step 3: Run Benchmarks

#### **CPU Only** (Baseline)
```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model.tflite \
  --num_threads=4 \
  --warmup_runs=10 \
  --num_runs=50"
```

#### **GPU via NNAPI** (Adreno GPU)
```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model.tflite \
  --use_nnapi=true \
  --nnapi_accelerator_name=qti-gpu \
  --warmup_runs=10 \
  --num_runs=50"
```

#### **NPU via NNAPI** (Hexagon DSP/NPU) ⭐
```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model.tflite \
  --use_nnapi=true \
  --nnapi_accelerator_name=qti-default \
  --warmup_runs=10 \
  --num_runs=50"
```

Or use HTP (Hexagon Tensor Processor) directly:
```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model.tflite \
  --use_nnapi=true \
  --nnapi_accelerator_name=qti-htp \
  --warmup_runs=10 \
  --num_runs=50"
```

### Step 4: Read the Results

The output will look like:
```
STARTING!
Min num runs: [50]
Num threads: [4]
Min warmup runs: [10]
Graph: [/data/local/tmp/model.tflite]
Enable op profiling: [0]

Loaded model /data/local/tmp/model.tflite
...
Inference timings in us: Init: 4521, First inference: 38234, Warmup (avg): 37856, Inference (avg): 37234
```

**Look for**: `Inference (avg): 37234` (in microseconds)
- Convert to milliseconds: 37234 µs = **37.23 ms** ✅

---

## 🐍 **Method 2: Python Script on Device**

I'll create an automated script that does everything for you!

### What This Script Does:
1. ✅ Pushes model to device
2. ✅ Pushes Python benchmark script to device
3. ✅ Runs benchmarks on CPU, GPU, NPU
4. ✅ Pulls results back to PC
5. ✅ Generates comparison graph

### Just Run:
```bash
python run_actual_qidk_benchmark.py
```

---

## 📊 **Understanding NNAPI Accelerators**

Query available accelerators on your device:
```bash
adb shell "/data/local/tmp/benchmark_model \
  --nnapi_accelerator_name=? \
  --graph=/data/local/tmp/model.tflite"
```

Common Qualcomm accelerators:
- `qti-default` - Auto-select best (usually NPU)
- `qti-htp` - Hexagon Tensor Processor (NPU)
- `qti-dsp` - Hexagon DSP
- `qti-gpu` - Adreno GPU
- `qti-aip` - AI accelerator

---

## 🎯 **Expected Results**

Based on our estimates, you should see:

| Runtime | Expected Time | FPS |
|---------|---------------|-----|
| CPU | 240-260 ms | ~4 FPS |
| GPU | 140-160 ms | ~6-7 FPS |
| NPU | **30-45 ms** ⚡ | **22-33 FPS** |

If NPU gives you **under 50ms**, you're golden for real-time! 🎉

---

## 🔍 **Troubleshooting**

### Issue: "benchmark_model: not found"
**Fix**: Make sure you downloaded the ARM64 version and made it executable
```bash
file benchmark_model  # Should say "ARM aarch64"
adb shell chmod +x /data/local/tmp/benchmark_model
```

### Issue: "NNAPI delegate not available"
**Fix**: Your device might not support NNAPI. Try CPU-only first:
```bash
adb shell "/data/local/tmp/benchmark_model --graph=/data/local/tmp/model.tflite"
```

### Issue: "Failed to create NNAPI delegate"
**Fix**: Try different accelerator names:
```bash
# List available
adb shell "/data/local/tmp/benchmark_model --nnapi_accelerator_name=?"

# Or try generic NNAPI (auto-select)
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model.tflite \
  --use_nnapi=true"
```

### Issue: Model not found
**Fix**: Make sure model is on device:
```bash
adb shell ls -lh /data/local/tmp/model.tflite
```

---

## 🚀 **Quick Start (Easiest Way)**

I've created an automated script for you. Just run:

```bash
python run_actual_qidk_benchmark.py
```

This will:
1. Download benchmark_model tool if needed
2. Push everything to device
3. Run CPU/GPU/NPU benchmarks
4. Pull results and create graphs
5. Show you actual measured times!

---

## 📈 **After Running**

You'll get:
- ✅ **Actual CPU time** from device
- ✅ **Actual GPU time** from device  
- ✅ **Actual NPU time** from device
- ✅ Comparison graph with real measurements
- ✅ JSON file with all timing data

No more estimates - **100% real measurements!** 🎯

---

## 💡 **Pro Tips**

1. **Warmup is important**: First few runs are slower (model loading, optimization)
2. **Run 50+ iterations**: Get stable average (variance decreases)
3. **NPU needs warmup**: Hexagon compiles graph on first run
4. **Check temperature**: Device throttling can affect results
5. **Close other apps**: For consistent measurements

---

## 🎬 **Next Steps**

1. Run the automated script (easiest):
   ```bash
   python run_actual_qidk_benchmark.py
   ```

2. Or manually run benchmark_model (more control):
   ```bash
   # Download, push, run as shown above
   ```

3. View results:
   - Terminal output shows average inference time
   - Graph shows CPU vs GPU vs NPU comparison
   - JSON has all raw data

**Let's get those real NPU measurements!** 🚀
