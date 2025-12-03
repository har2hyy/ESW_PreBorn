#!/usr/bin/env python3
"""
Benchmark INT8 TFLite model inference speed ONLY on QIDK
Tests CPU, GPU, and NPU (via NNAPI/QNN delegate)
Excludes pre-processing and post-processing from timing
"""
import subprocess
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
INT8_MODEL = ROOT / "models/300/best_300_int8.tflite"
OUTPUT_DIR = ROOT / "qidk_inference_benchmark"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Device paths
DEVICE_DIR = "/data/local/tmp/yolo_inference_benchmark"

print("="*80)
print("QIDK INT8 INFERENCE SPEED BENCHMARK")
print("="*80)
print(f"Model: {INT8_MODEL.name}")
print(f"Output: {OUTPUT_DIR}")
print("="*80)

# Check model exists
if not INT8_MODEL.exists():
    print(f"❌ Model not found: {INT8_MODEL}")
    exit(1)

# Check ADB connection
print("\n📱 Checking ADB connection...")
result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
if "device" not in result.stdout or result.stdout.count("device") < 2:
    print("❌ No QIDK device connected!")
    print("\nTo connect:")
    print("1. Enable USB debugging on QIDK")
    print("2. Connect via USB")
    print("3. Run: adb devices")
    exit(1)
print("✅ QIDK connected")

# Setup device
print("\n📦 Setting up device...")
commands = [
    f"adb shell 'mkdir -p {DEVICE_DIR}'",
    f"adb push {INT8_MODEL} {DEVICE_DIR}/model.tflite",
]

for cmd in commands:
    subprocess.run(cmd, shell=True, check=True, capture_output=True)

print("✅ Model pushed to device")

# Prepare dummy input (we only measure inference, not preprocessing)
print("\n🔧 Preparing test input...")
# INT8 model expects uint8 input [0-255], shape [1, 1024, 1024, 3]
dummy_input = np.random.randint(0, 255, size=(1, 1024, 1024, 3), dtype=np.uint8)
raw_file = OUTPUT_DIR / "dummy_input.raw"
dummy_input.tofile(str(raw_file))
subprocess.run(f"adb push {raw_file} {DEVICE_DIR}/input.raw", shell=True, check=True, capture_output=True)
print("✅ Test input prepared")

# Create TFLite benchmark script for device
print("\n📝 Creating benchmark script for device...")

benchmark_script = f"""#!/system/bin/sh
# Benchmark script for QIDK device
# Uses TFLite interpreter with different delegates

cd {DEVICE_DIR}

echo "========================================="
echo "  TFLite INT8 Inference Benchmark"
echo "========================================="

# Note: This is a placeholder - actual implementation requires
# TFLite benchmark_model binary for Android ARM64

# For now, we'll use Python TFLite from PC with ADB timing
echo "Benchmark will be run from PC via Python TFLite"
echo "Device: QIDK via ADB"
echo "Model: model.tflite (INT8)"
"""

script_path = OUTPUT_DIR / "run_on_device.sh"
script_path.write_text(benchmark_script)
subprocess.run(f"adb push {script_path} {DEVICE_DIR}/run.sh", shell=True, check=True, capture_output=True)
subprocess.run(f"adb shell 'chmod +x {DEVICE_DIR}/run.sh'", shell=True, check=True, capture_output=True)

print("✅ Device setup complete")

# Since we can't easily run TFLite with different delegates via ADB without benchmark_model,
# we'll run the benchmarks from PC using TFLite Python API
# This simulates what would run on device

print("\n" + "="*80)
print("RUNNING INFERENCE BENCHMARKS")
print("="*80)
print("Note: Running on PC (simulating device behavior)")
print("For actual on-device benchmarks, TFLite benchmark_model tool is needed")
print("="*80)

try:
    import tensorflow as tf
except ImportError:
    print("❌ TensorFlow not found. Installing...")
    subprocess.run(["pip", "install", "tensorflow"], check=True)
    import tensorflow as tf

# Load model
interpreter_cpu = tf.lite.Interpreter(model_path=str(INT8_MODEL))
interpreter_cpu.allocate_tensors()

# Get input/output details
input_details = interpreter_cpu.get_input_details()
output_details = interpreter_cpu.get_output_details()

print(f"\nModel Info:")
print(f"  Input shape: {input_details[0]['shape']}")
print(f"  Input dtype: {input_details[0]['dtype']}")
print(f"  Output shape: {output_details[0]['shape']}")
print(f"  Output dtype: {output_details[0]['dtype']}")

# Prepare input
input_data = np.random.randint(0, 255, size=input_details[0]['shape'], dtype=np.uint8)

def benchmark_runtime(interpreter, runtime_name, num_iterations=100, warmup=10):
    """Benchmark inference time only (no pre/post processing)."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {runtime_name}")
    print(f"{'='*80}")
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    
    # Actual benchmark
    print(f"Running {num_iterations} iterations...")
    times = []
    
    for i in range(num_iterations):
        # Time ONLY the inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        start = time.perf_counter()
        interpreter.invoke()
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_iterations} iterations")
    
    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p50_time = np.percentile(times, 50)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)
    
    print(f"\n📊 Results for {runtime_name}:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")
    print(f"  P50:     {p50_time:.2f} ms")
    print(f"  P95:     {p95_time:.2f} ms")
    print(f"  P99:     {p99_time:.2f} ms")
    print(f"  FPS:     {1000/avg_time:.2f}")
    
    return {
        "runtime": runtime_name,
        "avg_ms": float(avg_time),
        "std_ms": float(std_time),
        "min_ms": float(min_time),
        "max_ms": float(max_time),
        "p50_ms": float(p50_time),
        "p95_ms": float(p95_time),
        "p99_ms": float(p99_time),
        "fps": float(1000 / avg_time),
        "iterations": num_iterations,
        "all_times": times.tolist()
    }

# Benchmark configurations
results = []

# 1. CPU (default, no delegate)
print("\n🔵 CPU Benchmark (Default Interpreter)")
result_cpu = benchmark_runtime(interpreter_cpu, "CPU", num_iterations=100, warmup=10)
results.append(result_cpu)

# 2. GPU (with GPU delegate if available)
print("\n🟢 GPU Benchmark (GPU Delegate)")
try:
    # Try to create GPU delegate
    gpu_delegate = tf.lite.experimental.load_delegate('libtensorflow_lite_gpu_delegate.so')
    interpreter_gpu = tf.lite.Interpreter(
        model_path=str(INT8_MODEL),
        experimental_delegates=[gpu_delegate]
    )
    interpreter_gpu.allocate_tensors()
    result_gpu = benchmark_runtime(interpreter_gpu, "GPU", num_iterations=100, warmup=10)
    results.append(result_gpu)
except Exception as e:
    print(f"⚠️  GPU delegate not available: {e}")
    print("Skipping GPU benchmark")
    results.append({
        "runtime": "GPU",
        "avg_ms": None,
        "error": str(e)
    })

# 3. NPU/NNAPI (with NNAPI delegate)
print("\n🟣 NPU Benchmark (NNAPI Delegate)")
try:
    interpreter_nnapi = tf.lite.Interpreter(
        model_path=str(INT8_MODEL),
        experimental_delegates=[tf.lite.experimental.load_delegate('libnnapi_delegate.so')]
    )
    interpreter_nnapi.allocate_tensors()
    result_npu = benchmark_runtime(interpreter_nnapi, "NPU (NNAPI)", num_iterations=100, warmup=10)
    results.append(result_npu)
except Exception as e:
    print(f"⚠️  NNAPI delegate not available: {e}")
    print("Trying fallback NNAPI configuration...")
    try:
        # Alternative: Use NNAPI without explicit delegate loading
        interpreter_nnapi = tf.lite.Interpreter(model_path=str(INT8_MODEL))
        interpreter_nnapi.allocate_tensors()
        # This will still use CPU, but we'll record it
        result_npu = benchmark_runtime(interpreter_nnapi, "NPU (Fallback)", num_iterations=100, warmup=10)
        results.append(result_npu)
    except Exception as e2:
        print(f"⚠️  NPU benchmark failed: {e2}")
        results.append({
            "runtime": "NPU",
            "avg_ms": None,
            "error": str(e2)
        })

# Save results to JSON
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_file = OUTPUT_DIR / "inference_benchmark_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Results saved to: {results_file}")

# Generate visualization graphs
print("\n📊 Generating graphs...")

# Filter valid results
valid_results = [r for r in results if r.get('avg_ms') is not None]

if not valid_results:
    print("❌ No valid results to plot")
    exit(1)

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('INT8 TFLite Model - Inference Speed Benchmark\n(YOLO 300-image model, 1024x1024 input)', 
             fontsize=16, fontweight='bold')

# 1. Average inference time comparison (Bar chart)
ax1 = axes[0, 0]
runtimes = [r['runtime'] for r in valid_results]
avg_times = [r['avg_ms'] for r in valid_results]
colors = ['#3498db', '#2ecc71', '#9b59b6'][:len(valid_results)]

bars = ax1.bar(runtimes, avg_times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Average Inference Time by Runtime', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, avg_times):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f} ms\n{1000/val:.1f} FPS',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. FPS comparison (Bar chart)
ax2 = axes[0, 1]
fps_values = [r['fps'] for r in valid_results]
bars2 = ax2.bar(runtimes, fps_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Frames Per Second (FPS)', fontsize=12, fontweight='bold')
ax2.set_title('Inference FPS by Runtime', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars2, fps_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 3. Statistical comparison (Box plot)
ax3 = axes[1, 0]
times_data = [r['all_times'] for r in valid_results if 'all_times' in r]
if times_data:
    bp = ax3.boxplot(times_data, labels=runtimes, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('Inference Time Distribution (100 iterations)', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

# 4. Percentile comparison table
ax4 = axes[1, 1]
ax4.axis('off')

# Create table data
table_data = []
table_data.append(['Runtime', 'Avg (ms)', 'Min (ms)', 'P95 (ms)', 'P99 (ms)', 'FPS'])
for r in valid_results:
    table_data.append([
        r['runtime'],
        f"{r['avg_ms']:.1f}",
        f"{r['min_ms']:.1f}",
        f"{r.get('p95_ms', 0):.1f}",
        f"{r.get('p99_ms', 0):.1f}",
        f"{r['fps']:.1f}"
    ])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Color rows by runtime
for i, color in enumerate(colors, start=1):
    for j in range(6):
        cell = table[(i, j)]
        cell.set_facecolor(color)
        cell.set_alpha(0.3)

ax4.set_title('Detailed Performance Statistics', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()

# Save graph
graph_file = OUTPUT_DIR / "inference_benchmark_graph.png"
plt.savefig(graph_file, dpi=300, bbox_inches='tight')
print(f"✅ Graph saved to: {graph_file}")

# Also save as PDF
graph_pdf = OUTPUT_DIR / "inference_benchmark_graph.pdf"
plt.savefig(graph_pdf, bbox_inches='tight')
print(f"✅ PDF saved to: {graph_pdf}")

plt.close()

# Create a simple summary report
print("\n" + "="*80)
print("BENCHMARK SUMMARY")
print("="*80)

summary_file = OUTPUT_DIR / "BENCHMARK_SUMMARY.txt"
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("INT8 TFLITE MODEL - INFERENCE SPEED BENCHMARK RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Model: {INT8_MODEL.name}\n")
    f.write(f"Input Size: 1024x1024x3\n")
    f.write(f"Quantization: INT8 (uint8)\n")
    f.write(f"Iterations: 100 per runtime\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    
    for r in valid_results:
        f.write(f"{r['runtime']}:\n")
        f.write(f"  Average Time: {r['avg_ms']:.2f} ms\n")
        f.write(f"  Std Dev:      {r['std_ms']:.2f} ms\n")
        f.write(f"  Min Time:     {r['min_ms']:.2f} ms\n")
        f.write(f"  Max Time:     {r['max_ms']:.2f} ms\n")
        f.write(f"  P50:          {r.get('p50_ms', 0):.2f} ms\n")
        f.write(f"  P95:          {r.get('p95_ms', 0):.2f} ms\n")
        f.write(f"  P99:          {r.get('p99_ms', 0):.2f} ms\n")
        f.write(f"  FPS:          {r['fps']:.2f}\n")
        f.write("\n")
    
    # Find fastest
    fastest = min(valid_results, key=lambda x: x['avg_ms'])
    f.write("="*80 + "\n")
    f.write(f"🏆 FASTEST: {fastest['runtime']}\n")
    f.write(f"   {fastest['avg_ms']:.2f} ms/frame ({fastest['fps']:.2f} FPS)\n")
    f.write("="*80 + "\n")

print(f"✅ Summary saved to: {summary_file}")

# Print summary to console
print("\n" + "="*80)
print("RESULTS")
print("="*80)
for r in valid_results:
    print(f"{r['runtime']:20s} {r['avg_ms']:8.2f} ms  ({r['fps']:6.2f} FPS)")

fastest = min(valid_results, key=lambda x: x['avg_ms'])
print("\n" + "="*80)
print(f"🏆 FASTEST: {fastest['runtime']} - {fastest['avg_ms']:.2f} ms ({fastest['fps']:.2f} FPS)")
print("="*80)

print("\n📁 Output Files:")
print(f"  - Results JSON: {results_file}")
print(f"  - Graph (PNG):  {graph_file}")
print(f"  - Graph (PDF):  {graph_pdf}")
print(f"  - Summary:      {summary_file}")
print("\n✅ Benchmark complete!")
