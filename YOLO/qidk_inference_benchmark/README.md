# INT8 Model Inference Benchmark Results

## Summary

✅ **Benchmark completed successfully!**

### What Was Measured
- **Pure inference time only** (excluding pre-processing and post-processing)
- **100 iterations** per runtime configuration
- **Model**: INT8 TFLite (best_300_int8.tflite, 2.9 MB)
- **Input**: 1024x1024x3 uint8

### Results Location

📁 **All results saved to**: `qidk_inference_benchmark/`

**Generated Files:**
1. 📊 **Graph (PNG)**: `qidk_inference_benchmark/inference_benchmark_graph.png` ⭐
2. 📄 **Graph (PDF)**: `qidk_inference_benchmark/inference_benchmark_graph.pdf`
3. 📈 **JSON Data**: `qidk_inference_benchmark/inference_benchmark_results.json`
4. 📝 **Text Summary**: `qidk_inference_benchmark/BENCHMARK_SUMMARY.txt`

## Current Results (PC CPU)

⚠️ **Note**: These results are from running on your **PC CPU**, not on QIDK device.
GPU and NPU delegates were not available on PC.

| Runtime | Avg Time | FPS | Notes |
|---------|----------|-----|-------|
| CPU | 215.23 ms | 4.65 | PC Intel/AMD CPU |
| NPU (Fallback) | 217.74 ms | 4.59 | Actually CPU (no delegate) |

**Statistics:**
- **CPU**:
  - Average: 215.23 ms
  - Min: 209.78 ms
  - P95: 222.84 ms
  - Std Dev: 3.26 ms

## To Get TRUE QIDK NPU Performance

The current benchmark ran on PC. To get actual QIDK CPU/GPU/NPU speeds, you need to:

### Option 1: Use TFLite Benchmark Tool on QIDK (Recommended)

1. **Download TFLite benchmark_model for Android ARM64**:
   ```bash
   # Get from TensorFlow releases or build from source
   # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark
   ```

2. **Push to QIDK**:
   ```bash
   adb push benchmark_model /data/local/tmp/yolo_inference_benchmark/
   adb shell "chmod +x /data/local/tmp/yolo_inference_benchmark/benchmark_model"
   ```

3. **Run benchmarks**:
   ```bash
   # CPU
   adb shell "/data/local/tmp/yolo_inference_benchmark/benchmark_model \
     --graph=/data/local/tmp/yolo_inference_benchmark/model.tflite \
     --num_threads=4"
   
   # GPU
   adb shell "/data/local/tmp/yolo_inference_benchmark/benchmark_model \
     --graph=/data/local/tmp/yolo_inference_benchmark/model.tflite \
     --use_gpu=true"
   
   # NPU (NNAPI)
   adb shell "/data/local/tmp/yolo_inference_benchmark/benchmark_model \
     --graph=/data/local/tmp/yolo_inference_benchmark/model.tflite \
     --use_nnapi=true \
     --nnapi_accelerator_name=qti-default"
   ```

### Option 2: Use DLC with SNPE (Alternative)

Use the existing DLC models with SNPE runtime on QIDK:
```bash
python run_qidk_benchmark_simple.py
```

This will test the DLC FP32/FP16 models on CPU/GPU/NPU.

## Expected Performance on QIDK

Based on typical Qualcomm Snapdragon performance:

| Runtime | Expected Speed | Speedup vs CPU |
|---------|----------------|----------------|
| **CPU** | ~200-300 ms | 1x baseline |
| **GPU** | ~100-150 ms | ~2x faster |
| **NPU (HTP)** | **~20-40 ms** ⚡ | **~5-10x faster** |

**INT8 quantization typically gives 2-4x speedup on NPU compared to FP32!**

## Visualization

The generated graph shows:
1. **Average Inference Time** - Bar chart comparing runtimes
2. **FPS Comparison** - Frames per second by runtime
3. **Time Distribution** - Box plot showing variance
4. **Statistics Table** - Detailed performance metrics

**View the graph**: Open `qidk_inference_benchmark/inference_benchmark_graph.png`

## Next Steps

### 1. View the Results Graph
```bash
# Open the PNG file
xdg-open qidk_inference_benchmark/inference_benchmark_graph.png

# Or view PDF
xdg-open qidk_inference_benchmark/inference_benchmark_graph.pdf
```

### 2. Run on Actual QIDK Device
To get real NPU performance, either:
- Get TFLite benchmark_model tool and run on device (see Option 1 above)
- Use DLC models with SNPE: `python run_qidk_benchmark_simple.py`

### 3. Compare Models
Compare INT8 vs FP16 vs FP32 performance on NPU:
- INT8 should be fastest (2-4x vs FP32)
- FP16 good balance (1.5-2x vs FP32)
- FP32 baseline (most accurate)

## Files Ready for QIDK

The script already pushed to your QIDK device:
- Model: `/data/local/tmp/yolo_inference_benchmark/model.tflite`
- Test input: `/data/local/tmp/yolo_inference_benchmark/input.raw`

You can now run on-device benchmarks using the commands above!

---

**Summary**: 
- ✅ PC CPU benchmark: ~215 ms (4.65 FPS)
- 📊 Graph saved to: `qidk_inference_benchmark/inference_benchmark_graph.png`
- 🚀 For true NPU speed, run on QIDK device (expected: ~20-40 ms, 25-50 FPS)
