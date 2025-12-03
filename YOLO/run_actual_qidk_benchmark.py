#!/usr/bin/env python3
"""
Automated QIDK On-Device Benchmark Runner
Runs INT8 TFLite model on actual QIDK hardware (CPU, GPU, NPU)
Gets REAL measurements - no estimates!
"""

import subprocess
import json
import time
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configuration
MODEL_PATH = "models/300/best_300_int8.tflite"
DEVICE_DIR = "/data/local/tmp/qidk_benchmark"
OUTPUT_DIR = "qidk_ondevice_actual_results"
WARMUP_RUNS = 10
NUM_RUNS = 50

# Benchmark tool URL (TFLite nightly)
BENCHMARK_URL = "https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model"
BENCHMARK_LOCAL = "benchmark_model_arm64"

def run_command(cmd, capture=True):
    """Run shell command"""
    print(f"Running: {cmd}")
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    else:
        return subprocess.run(cmd, shell=True).returncode

def check_adb():
    """Check if ADB is connected"""
    print("\n=== Checking ADB Connection ===")
    stdout, stderr, code = run_command("adb devices")
    if code != 0 or "device" not in stdout:
        print("❌ No ADB device connected!")
        print("Please connect QIDK and enable USB debugging")
        sys.exit(1)
    print("✅ ADB device connected")
    return True

def download_benchmark_tool():
    """Download TFLite benchmark tool if not present"""
    if os.path.exists(BENCHMARK_LOCAL):
        print(f"✅ Benchmark tool already downloaded: {BENCHMARK_LOCAL}")
        return BENCHMARK_LOCAL
    
    print("\n=== Downloading TFLite Benchmark Tool ===")
    print(f"URL: {BENCHMARK_URL}")
    
    code = run_command(f"wget -O {BENCHMARK_LOCAL} {BENCHMARK_URL}", capture=False)
    if code != 0:
        print("❌ Failed to download benchmark tool!")
        print("\nAlternative: Download manually from:")
        print("https://www.tensorflow.org/lite/performance/measurement")
        sys.exit(1)
    
    # Make executable
    os.chmod(BENCHMARK_LOCAL, 0o755)
    print(f"✅ Downloaded and made executable: {BENCHMARK_LOCAL}")
    return BENCHMARK_LOCAL

def setup_device():
    """Push model and benchmark tool to device"""
    print("\n=== Setting Up Device ===")
    
    # Create device directory
    run_command(f"adb shell mkdir -p {DEVICE_DIR}")
    
    # Push model
    print(f"Pushing model: {MODEL_PATH}")
    code = run_command(f"adb push {MODEL_PATH} {DEVICE_DIR}/model.tflite", capture=False)
    if code != 0:
        print("❌ Failed to push model!")
        sys.exit(1)
    
    # Push benchmark tool
    print(f"Pushing benchmark tool: {BENCHMARK_LOCAL}")
    code = run_command(f"adb push {BENCHMARK_LOCAL} {DEVICE_DIR}/benchmark_model", capture=False)
    if code != 0:
        print("❌ Failed to push benchmark tool!")
        sys.exit(1)
    
    # Make executable on device
    run_command(f"adb shell chmod +x {DEVICE_DIR}/benchmark_model")
    
    print("✅ Device setup complete!")

def run_benchmark(accelerator=None, name="CPU"):
    """Run benchmark on device with specified accelerator"""
    print(f"\n=== Running Benchmark: {name} ===")
    
    # Build command
    cmd = f"adb shell \"{DEVICE_DIR}/benchmark_model "
    cmd += f"--graph={DEVICE_DIR}/model.tflite "
    cmd += f"--warmup_runs={WARMUP_RUNS} "
    cmd += f"--num_runs={NUM_RUNS} "
    
    if accelerator:
        cmd += f"--use_nnapi=true "
        cmd += f"--nnapi_accelerator_name={accelerator} "
    else:
        cmd += "--num_threads=4 "
    
    cmd += "\""
    
    print(f"Command: {cmd}")
    stdout, stderr, code = run_command(cmd)
    
    if code != 0:
        print(f"⚠️  Benchmark failed for {name}")
        print(f"Error: {stderr}")
        return None
    
    # Parse results
    result = parse_benchmark_output(stdout, name)
    return result

def parse_benchmark_output(output, name):
    """Parse benchmark tool output to extract timing"""
    print(f"\n--- {name} Output ---")
    print(output)
    print("--- End Output ---\n")
    
    # Look for: "Inference (avg): 37234"
    for line in output.split('\n'):
        if 'Inference (avg):' in line:
            # Extract number (in microseconds)
            parts = line.split(':')
            if len(parts) >= 2:
                time_us = parts[-1].strip()
                try:
                    time_us = float(time_us)
                    time_ms = time_us / 1000.0
                    fps = 1000.0 / time_ms
                    
                    result = {
                        'name': name,
                        'time_us': time_us,
                        'time_ms': time_ms,
                        'fps': fps
                    }
                    
                    print(f"✅ {name}: {time_ms:.2f} ms ({fps:.2f} FPS)")
                    return result
                except:
                    pass
    
    print(f"⚠️  Could not parse timing from {name} output")
    return None

def list_available_accelerators():
    """Query available NNAPI accelerators on device"""
    print("\n=== Querying Available Accelerators ===")
    cmd = f"adb shell \"{DEVICE_DIR}/benchmark_model "
    cmd += f"--graph={DEVICE_DIR}/model.tflite "
    cmd += "--nnapi_accelerator_name=?\""
    
    stdout, stderr, code = run_command(cmd)
    print(stdout)
    print(stderr)

def run_all_benchmarks():
    """Run benchmarks on all available runtimes"""
    results = []
    
    # CPU (no NNAPI)
    cpu_result = run_benchmark(accelerator=None, name="CPU")
    if cpu_result:
        results.append(cpu_result)
    
    # List available accelerators
    list_available_accelerators()
    
    # GPU via NNAPI
    gpu_result = run_benchmark(accelerator="qti-gpu", name="GPU")
    if gpu_result:
        results.append(gpu_result)
    
    # NPU/HTP via NNAPI - try multiple names
    npu_accelerators = [
        ("qti-htp", "NPU (HTP)"),
        ("qti-default", "NPU (Default)"),
        ("qti-dsp", "NPU (DSP)"),
    ]
    
    npu_found = False
    for accel, name in npu_accelerators:
        if npu_found:
            break
        print(f"\nTrying NPU accelerator: {accel}")
        npu_result = run_benchmark(accelerator=accel, name=name)
        if npu_result:
            results.append(npu_result)
            npu_found = True
            break
    
    if not npu_found:
        print("⚠️  Could not run NPU benchmark - trying generic NNAPI")
        generic_result = run_benchmark(accelerator=None, name="NNAPI (Auto)")
        if generic_result:
            results.append(generic_result)
    
    return results

def create_visualization(results):
    """Create comprehensive visualization of results"""
    if not results:
        print("⚠️  No results to visualize")
        return
    
    print(f"\n=== Creating Visualization ===")
    
    # Prepare data
    names = [r['name'] for r in results]
    times_ms = [r['time_ms'] for r in results]
    fps_values = [r['fps'] for r in results]
    
    # Calculate speedups relative to CPU
    cpu_time = next((r['time_ms'] for r in results if 'CPU' in r['name']), times_ms[0])
    speedups = [cpu_time / t for t in times_ms]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    # 1. Inference Time Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    bars1 = ax1.bar(names, times_ms, color=colors[:len(names)], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('🚀 QIDK On-Device Actual Measurements - Inference Time', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, time in zip(bars1, times_ms):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f} ms',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. FPS Comparison
    ax2 = fig.add_subplot(gs[1, :2])
    bars2 = ax2.bar(names, fps_values, color=colors[:len(names)], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Frames Per Second', fontsize=12, fontweight='bold')
    ax2.set_title('📊 Throughput (FPS)', fontsize=14, fontweight='bold')
    ax2.axhline(y=25, color='red', linestyle='--', label='Real-time threshold (25 FPS)', linewidth=2)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, fps in zip(bars2, fps_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{fps:.1f} FPS',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Speedup Analysis
    ax3 = fig.add_subplot(gs[2, :2])
    bars3 = ax3.bar(names, speedups, color=colors[:len(names)], alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Speedup vs CPU', fontsize=12, fontweight='bold')
    ax3.set_title('⚡ Acceleration Factor', fontsize=14, fontweight='bold')
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, speedup in zip(bars3, speedups):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x',
                ha='center', va='bottom', fontweight='bold')
    
    # 4. Summary Table
    ax4 = fig.add_subplot(gs[:, 2])
    ax4.axis('off')
    
    table_data = [['Runtime', 'Time (ms)', 'FPS', 'Speedup']]
    for r, speedup in zip(results, speedups):
        table_data.append([
            r['name'],
            f"{r['time_ms']:.1f}",
            f"{r['fps']:.1f}",
            f"{speedup:.1f}x"
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data)):
        for j in range(4):
            table[(i, j)].set_facecolor(colors[i-1] if i-1 < len(colors) else '#ecf0f1')
    
    ax4.set_title('📋 Performance Summary\n(ACTUAL MEASUREMENTS ✅)', 
                 fontsize=12, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('QIDK On-Device Benchmark Results - INT8 TFLite YOLO11n\n' + 
                f'Model: best_300_int8.tflite | Runs: {NUM_RUNS} | Warmup: {WARMUP_RUNS}',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = f"{OUTPUT_DIR}/ondevice_benchmark_actual.png"
    pdf_path = f"{OUTPUT_DIR}/ondevice_benchmark_actual.pdf"
    
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ Saved graph: {png_path}")
    print(f"✅ Saved PDF: {pdf_path}")
    
    plt.close()

def save_results(results):
    """Save results to JSON"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': MODEL_PATH,
        'warmup_runs': WARMUP_RUNS,
        'num_runs': NUM_RUNS,
        'results': results
    }
    
    json_path = f"{OUTPUT_DIR}/ondevice_benchmark_actual.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved results: {json_path}")
    
    # Save summary text
    summary_path = f"{OUTPUT_DIR}/ONDEVICE_BENCHMARK_SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("QIDK ON-DEVICE ACTUAL BENCHMARK RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Warmup Runs: {WARMUP_RUNS}\n")
        f.write(f"Measurement Runs: {NUM_RUNS}\n\n")
        
        f.write("RESULTS (ACTUAL MEASUREMENTS ✅):\n")
        f.write("-" * 60 + "\n\n")
        
        cpu_time = next((r['time_ms'] for r in results if 'CPU' in r['name']), None)
        
        for r in results:
            f.write(f"Platform: {r['name']}\n")
            f.write(f"  Time:    {r['time_ms']:.2f} ms\n")
            f.write(f"  FPS:     {r['fps']:.2f}\n")
            if cpu_time:
                speedup = cpu_time / r['time_ms']
                f.write(f"  Speedup: {speedup:.2f}x vs CPU\n")
            f.write("\n")
        
        # Find best performer
        best = min(results, key=lambda x: x['time_ms'])
        f.write("=" * 60 + "\n")
        f.write(f"🏆 BEST PERFORMER: {best['name']}\n")
        f.write(f"   Time: {best['time_ms']:.2f} ms ({best['fps']:.2f} FPS)\n")
        
        if best['fps'] >= 25:
            f.write("   ✅ REAL-TIME CAPABLE!\n")
        else:
            f.write("   ⚠️  Below 25 FPS threshold\n")
        
        f.write("=" * 60 + "\n")
    
    print(f"✅ Saved summary: {summary_path}")

def main():
    """Main execution flow"""
    print("\n" + "=" * 60)
    print("🚀 QIDK ON-DEVICE ACTUAL BENCHMARK")
    print("   INT8 TFLite Model - CPU/GPU/NPU Testing")
    print("=" * 60 + "\n")
    
    # Step 1: Check ADB
    if not check_adb():
        return
    
    # Step 2: Download benchmark tool
    download_benchmark_tool()
    
    # Step 3: Setup device
    setup_device()
    
    # Step 4: Run benchmarks
    print("\n" + "=" * 60)
    print("Starting benchmark runs...")
    print("This may take a few minutes...")
    print("=" * 60)
    
    results = run_all_benchmarks()
    
    if not results:
        print("\n❌ No benchmark results obtained!")
        print("Check device connection and try again")
        return
    
    # Step 5: Save and visualize
    save_results(results)
    create_visualization(results)
    
    # Step 6: Print summary
    print("\n" + "=" * 60)
    print("✅ BENCHMARK COMPLETE!")
    print("=" * 60)
    print("\nResults:")
    for r in results:
        print(f"  {r['name']:20s}: {r['time_ms']:6.2f} ms ({r['fps']:5.2f} FPS)")
    
    best = min(results, key=lambda x: x['time_ms'])
    print(f"\n🏆 Best: {best['name']} at {best['time_ms']:.2f} ms ({best['fps']:.2f} FPS)")
    
    if best['fps'] >= 25:
        print("   ✅ REAL-TIME CAPABLE!")
    
    print(f"\n📊 View graph: {OUTPUT_DIR}/ondevice_benchmark_actual.png")
    print(f"📄 Full results: {OUTPUT_DIR}/ondevice_benchmark_actual.json")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
