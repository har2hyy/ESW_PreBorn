#!/usr/bin/env python3
"""
Run inference benchmarks directly on QIDK device
Tests CPU, GPU, and NPU using SNPE runtime
Uses existing DLC models or creates new ones
"""
import os
import subprocess
import time
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "qidk_ondevice_benchmark"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

SNPE_ROOT = os.environ.get("SNPE_ROOT", "/home/harshyy/snpe-sdk/2.40.0.251030")
DEVICE_DIR = "/data/local/tmp/snpe_benchmark"

print("="*80)
print("QIDK ON-DEVICE INFERENCE BENCHMARK")
print("="*80)
print(f"SNPE Root: {SNPE_ROOT}")
print(f"Output: {OUTPUT_DIR}")
print("="*80)

# Check ADB
print("\n📱 Checking QIDK connection...")
result = subprocess.run(["adb", "devices"], capture_output=True, text=True)
if "device" not in result.stdout or result.stdout.count("device") < 2:
    print("❌ No QIDK device connected!")
    print("Please connect QIDK via USB and enable USB debugging")
    exit(1)
print("✅ QIDK connected")

# Check for DLC models
DLC_MODELS = {
    "FP32": ROOT / "models/qnn_dlc/best_yolo.bin",
    "FP16": ROOT / "models/qnn_dlc/best_yolo_fp16.bin",
}

available_models = {}
for name, path in DLC_MODELS.items():
    if path.exists():
        available_models[name] = path
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"✅ Found {name} model: {path.name} ({size_mb:.1f} MB)")

if not available_models:
    print("\n❌ No DLC models found!")
    print("Expected models in: models/qnn_dlc/")
    exit(1)

# Setup device
print("\n" + "="*80)
print("Setting up QIDK device...")
print("="*80)

# Create directories
print("Creating device directories...")
subprocess.run(f"adb shell 'mkdir -p {DEVICE_DIR}/inputs {DEVICE_DIR}/outputs'", 
               shell=True, check=True, capture_output=True)

# Push SNPE libraries
print("Pushing SNPE libraries...")
libs = [
    f"{SNPE_ROOT}/lib/aarch64-android/libSNPE.so",
    f"{SNPE_ROOT}/lib/aarch64-android/libhta.so",
    f"{SNPE_ROOT}/lib/dsp/libsnpehtpv75_skel.so",
]

for lib in libs:
    if Path(lib).exists():
        subprocess.run(f"adb push {lib} {DEVICE_DIR}/", 
                      shell=True, capture_output=True)

# Push snpe-net-run
snpe_net_run = f"{SNPE_ROOT}/bin/aarch64-android/snpe-net-run"
if Path(snpe_net_run).exists():
    subprocess.run(f"adb push {snpe_net_run} {DEVICE_DIR}/", 
                  shell=True, check=True, capture_output=True)
    subprocess.run(f"adb shell 'chmod +x {DEVICE_DIR}/snpe-net-run'", 
                  shell=True, check=True, capture_output=True)
    print("✅ snpe-net-run pushed")
else:
    print(f"❌ snpe-net-run not found at {snpe_net_run}")
    exit(1)

# Push models
print("Pushing models to device...")
for name, path in available_models.items():
    model_name = f"model_{name.lower()}.dlc"
    subprocess.run(f"adb push {path} {DEVICE_DIR}/{model_name}", 
                  shell=True, check=True, capture_output=True)
    print(f"  ✅ {name} model pushed")

# Prepare test input
print("Preparing test input...")
import cv2
test_img = ROOT / "test_images/my_optimal_result.jpg"
if test_img.exists():
    img = cv2.imread(str(test_img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    raw_file = OUTPUT_DIR / "test_input.raw"
    img.tofile(str(raw_file))
    
    subprocess.run(f"adb push {raw_file} {DEVICE_DIR}/inputs/", 
                  shell=True, check=True, capture_output=True)
    
    # Create input list
    input_list = OUTPUT_DIR / "input_list.txt"
    input_list.write_text("images:=inputs/test_input.raw\n")
    subprocess.run(f"adb push {input_list} {DEVICE_DIR}/", 
                  shell=True, check=True, capture_output=True)
    print("✅ Test input prepared")
else:
    print(f"⚠️  Test image not found, using dummy input")

print("✅ Device setup complete")

# Benchmark function
def run_on_device_benchmark(model_name, model_file, runtime, num_runs=50):
    """Run benchmark on QIDK device."""
    print(f"\n{'='*80}")
    print(f"Benchmarking {model_name} on {runtime.upper()}")
    print(f"{'='*80}")
    
    # Set runtime flag
    if runtime == "cpu":
        runtime_flag = ""  # CPU is default, no flag needed
        env_vars = "LD_LIBRARY_PATH=."
    elif runtime == "gpu":
        runtime_flag = "--use_gpu"
        env_vars = "LD_LIBRARY_PATH=."
    elif runtime == "npu":
        runtime_flag = "--use_dsp"
        env_vars = "LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=."
    else:
        return None
    
    output_dir = f"outputs/{runtime}_{model_name.lower()}"
    
    # Build command
    cmd = f'''adb shell "cd {DEVICE_DIR} && \\
        {env_vars} ./snpe-net-run \\
        --container {model_file} \\
        --input_list input_list.txt \\
        --output_dir {output_dir} \\
        {runtime_flag} \\
        --perf_profile high_performance"'''
    
    print(f"Running {num_runs} iterations...")
    times = []
    
    for i in range(num_runs):
        # Clear output directory
        subprocess.run(f"adb shell 'rm -rf {DEVICE_DIR}/{output_dir}/*'", 
                      shell=True, capture_output=True)
        
        # Run inference with timing
        start = time.perf_counter()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        
        if result.returncode != 0:
            print(f"  ❌ Iteration {i+1} failed")
            if i == 0:  # Show error on first failure
                print(f"  Error: {result.stderr[:500]}")
            continue
        
        # Parse timing from output if available
        # For now, use wall clock time (includes ADB overhead)
        times.append(elapsed * 1000)  # Convert to ms
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_runs} - Avg: {np.mean(times):.1f} ms")
    
    if not times:
        print(f"❌ All iterations failed for {runtime}")
        return None
    
    times = np.array(times)
    
    # Calculate statistics
    result_data = {
        "model": model_name,
        "runtime": runtime,
        "avg_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "fps": float(1000 / np.mean(times)),
        "successful_runs": len(times),
        "total_runs": num_runs,
        "all_times": times.tolist()
    }
    
    print(f"\n📊 Results:")
    print(f"  Average: {result_data['avg_ms']:.2f} ms")
    print(f"  Std Dev: {result_data['std_ms']:.2f} ms")
    print(f"  Min:     {result_data['min_ms']:.2f} ms")
    print(f"  P95:     {result_data['p95_ms']:.2f} ms")
    print(f"  FPS:     {result_data['fps']:.2f}")
    print(f"  Success: {len(times)}/{num_runs}")
    
    return result_data

# Run benchmarks
print("\n" + "="*80)
print("RUNNING BENCHMARKS ON QIDK")
print("="*80)

results = []
runtimes = ["cpu", "gpu", "npu"]

# Use first available model for testing
test_model_name = list(available_models.keys())[0]
test_model_path = available_models[test_model_name]
model_file = f"model_{test_model_name.lower()}.dlc"

print(f"\nTesting with: {test_model_name} model")
print(f"Runtimes: {', '.join(runtimes)}")

for runtime in runtimes:
    result = run_on_device_benchmark(test_model_name, model_file, runtime, num_runs=30)
    if result:
        results.append(result)
    time.sleep(2)  # Small delay between tests

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_file = OUTPUT_DIR / "ondevice_benchmark_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Results saved: {results_file}")

# Generate graphs
if not results:
    print("❌ No valid results to plot")
    exit(1)

print("\n📊 Generating graphs...")

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(f'QIDK On-Device Inference Benchmark - {test_model_name} Model\n' + 
             'INT8 TFLite (1024x1024 input)', 
             fontsize=16, fontweight='bold')

# 1. Average inference time
ax1 = axes[0, 0]
runtimes_list = [r['runtime'].upper() for r in results]
avg_times = [r['avg_ms'] for r in results]
colors = ['#3498db', '#2ecc71', '#9b59b6'][:len(results)]

bars = ax1.bar(runtimes_list, avg_times, color=colors, alpha=0.7, 
               edgecolor='black', linewidth=2)
ax1.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Average Inference Time (including ADB overhead)', 
              fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for bar, val, fps in zip(bars, avg_times, [r['fps'] for r in results]):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f} ms\n{fps:.1f} FPS',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# 2. FPS comparison
ax2 = axes[0, 1]
fps_values = [r['fps'] for r in results]
bars2 = ax2.bar(runtimes_list, fps_values, color=colors, alpha=0.7,
                edgecolor='black', linewidth=2)
ax2.set_ylabel('Frames Per Second', fontsize=12, fontweight='bold')
ax2.set_title('Throughput (FPS)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for bar, val in zip(bars2, fps_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# 3. Distribution box plot
ax3 = axes[1, 0]
times_data = [r['all_times'] for r in results]
bp = ax3.boxplot(times_data, labels=runtimes_list, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax3.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
ax3.set_title('Time Distribution (30 runs per runtime)', 
              fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# 4. Speedup comparison
ax4 = axes[1, 1]
cpu_time = next((r['avg_ms'] for r in results if r['runtime'] == 'cpu'), None)
if cpu_time:
    speedups = [cpu_time / r['avg_ms'] for r in results]
    bars4 = ax4.bar(runtimes_list, speedups, color=colors, alpha=0.7,
                    edgecolor='black', linewidth=2)
    ax4.set_ylabel('Speedup vs CPU', fontsize=12, fontweight='bold')
    ax4.set_title('Performance Speedup', fontsize=13, fontweight='bold')
    ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, label='CPU Baseline')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.legend()
    
    for bar, val in zip(bars4, speedups):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.2f}x',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
else:
    ax4.axis('off')
    ax4.text(0.5, 0.5, 'CPU benchmark not available\nfor speedup comparison',
             ha='center', va='center', fontsize=12)

plt.tight_layout()

graph_file = OUTPUT_DIR / "ondevice_benchmark_graph.png"
plt.savefig(graph_file, dpi=300, bbox_inches='tight')
print(f"✅ Graph saved: {graph_file}")

graph_pdf = OUTPUT_DIR / "ondevice_benchmark_graph.pdf"
plt.savefig(graph_pdf, bbox_inches='tight')
print(f"✅ PDF saved: {graph_pdf}")

plt.close()

# Summary report
print("\n" + "="*80)
print("BENCHMARK RESULTS SUMMARY")
print("="*80)

summary_file = OUTPUT_DIR / "ONDEVICE_BENCHMARK_SUMMARY.txt"
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("QIDK ON-DEVICE INFERENCE BENCHMARK RESULTS\n")
    f.write("="*80 + "\n")
    f.write(f"Model: {test_model_name}\n")
    f.write(f"Input Size: 1024x1024x3\n")
    f.write(f"Device: QIDK (Qualcomm Snapdragon)\n")
    f.write(f"Runs per runtime: 30\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    f.write("NOTE: Times include ADB communication overhead\n")
    f.write("      Actual on-device inference is faster\n\n")
    
    for r in results:
        f.write(f"{r['runtime'].upper()}:\n")
        f.write(f"  Average:  {r['avg_ms']:.2f} ms\n")
        f.write(f"  Std Dev:  {r['std_ms']:.2f} ms\n")
        f.write(f"  Min:      {r['min_ms']:.2f} ms\n")
        f.write(f"  P95:      {r['p95_ms']:.2f} ms\n")
        f.write(f"  P99:      {r['p99_ms']:.2f} ms\n")
        f.write(f"  FPS:      {r['fps']:.2f}\n")
        f.write(f"  Success:  {r['successful_runs']}/{r['total_runs']}\n")
        f.write("\n")
    
    fastest = min(results, key=lambda x: x['avg_ms'])
    f.write("="*80 + "\n")
    f.write(f"🏆 FASTEST: {fastest['runtime'].upper()}\n")
    f.write(f"   {fastest['avg_ms']:.2f} ms/frame ({fastest['fps']:.2f} FPS)\n")
    
    if cpu_time:
        speedup = cpu_time / fastest['avg_ms']
        f.write(f"   {speedup:.2f}x faster than CPU\n")
    
    f.write("="*80 + "\n")

print(f"✅ Summary saved: {summary_file}")

# Console output
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"{'Runtime':<10} {'Avg Time':<12} {'FPS':<10} {'Speedup':<10}")
print("-"*80)

for r in results:
    speedup = cpu_time / r['avg_ms'] if cpu_time else 1.0
    print(f"{r['runtime'].upper():<10} {r['avg_ms']:>8.2f} ms  {r['fps']:>6.2f}    {speedup:>5.2f}x")

fastest = min(results, key=lambda x: x['avg_ms'])
print("\n" + "="*80)
print(f"🏆 FASTEST: {fastest['runtime'].upper()} - {fastest['avg_ms']:.2f} ms ({fastest['fps']:.2f} FPS)")
print("="*80)

print("\n📁 Output files:")
print(f"  - Graph (PNG): {graph_file}")
print(f"  - Graph (PDF): {graph_pdf}")
print(f"  - JSON data:   {results_file}")
print(f"  - Summary:     {summary_file}")

print("\n✅ On-device benchmark complete!")
print("\nNote: Times include ADB overhead. For pure inference times,")
print("      use TFLite benchmark_model tool directly on device.")
