# 🎯 QUICK START - Run Model on QIDK Device

## ⚡ Fastest Way (Automated)

```bash
# Make sure yolo11 environment is activated
conda activate yolo11

# Run the automated benchmark script
python run_actual_qidk_benchmark.py
```

**This script will:**
1. ✅ Download TFLite benchmark tool
2. ✅ Push model to your QIDK device  
3. ✅ Run on CPU, GPU, and NPU
4. ✅ Get ACTUAL measurements (no estimates!)
5. ✅ Generate graphs and save results

**Results saved to:** `qidk_ondevice_actual_results/`

---

## 🔧 Manual Method (More Control)

### Step 1: Download Benchmark Tool
```bash
wget https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model -O benchmark_model
chmod +x benchmark_model
```

### Step 2: Push to Device
```bash
adb push benchmark_model /data/local/tmp/
adb push models/300/best_300_int8.tflite /data/local/tmp/model.tflite
adb shell chmod +x /data/local/tmp/benchmark_model
```

### Step 3: Run Benchmarks

**CPU:**
```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model.tflite \
  --num_threads=4 \
  --warmup_runs=10 \
  --num_runs=50"
```

**GPU (Adreno):**
```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model.tflite \
  --use_nnapi=true \
  --nnapi_accelerator_name=qti-gpu \
  --warmup_runs=10 \
  --num_runs=50"
```

**NPU (Hexagon):**
```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model.tflite \
  --use_nnapi=true \
  --nnapi_accelerator_name=qti-htp \
  --warmup_runs=10 \
  --num_runs=50"
```

### Step 4: Read Results
Look for line: `Inference (avg): 37234` (in microseconds)
- Divide by 1000 to get milliseconds: 37234 µs = **37.23 ms**

---

## 📊 What to Expect

Based on our estimates:
- **CPU**: 240-260 ms (~4 FPS)
- **GPU**: 140-160 ms (~6-7 FPS)  
- **NPU**: **30-45 ms (22-33 FPS)** ⚡

If NPU gives **under 50 ms** → You're good for real-time! 🎉

---

## 🚨 Troubleshooting

### "benchmark_model: not found"
```bash
file benchmark_model  # Should say "ARM aarch64"
adb shell chmod +x /data/local/tmp/benchmark_model
```

### "NNAPI delegate failed"
Try listing available accelerators:
```bash
adb shell "/data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/model.tflite \
  --nnapi_accelerator_name=?"
```

### Device not found
```bash
adb devices  # Should show your QIDK
adb kill-server && adb start-server  # Restart ADB if needed
```

---

## 📂 Output Files

After running `run_actual_qidk_benchmark.py`:

```
qidk_ondevice_actual_results/
├── ondevice_benchmark_actual.png      # Main graph
├── ondevice_benchmark_actual.pdf      # PDF version
├── ondevice_benchmark_actual.json     # Raw data
└── ONDEVICE_BENCHMARK_SUMMARY.txt     # Text summary
```

---

## 🎬 Just Run This!

```bash
conda activate yolo11
python run_actual_qidk_benchmark.py
```

**That's it!** Real measurements in ~5 minutes. 🚀
