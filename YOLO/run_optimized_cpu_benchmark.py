#!/usr/bin/env python3
"""
Optimized CPU Benchmark for QIDK
Uses XNNPack + 8 threads for maximum CPU performance
"""

import subprocess
import json
import re
import matplotlib.pyplot as plt
import os

BENCHMARK_TOOL = "/data/local/tmp/qidk_benchmark/benchmark_model"
MODEL_PATH = "/data/local/tmp/qidk_benchmark/model.tflite"
OUTPUT_DIR = "qidk_optimized_cpu_results"
WARMUP_RUNS = 10
NUM_RUNS = 50

def run_adb(cmd):
    """Run ADB shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(f'adb shell "{cmd}"', shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def parse_inference_time(output):
    """Extract inference time from benchmark output"""
    match = re.search(r'Inference \(avg\):\s+(\d+\.?\d*)', output)
    if match:
        time_us = float(match.group(1))
        time_ms = time_us / 1000.0
        fps = 1000.0 / time_ms
        return time_ms, fps
    return None, None

def run_benchmark(config_name, extra_args=""):
    """Run benchmark with specific configuration"""
    print(f"\n{'='*70}")
    print(f"  {config_name}")
    print('='*70)
    
    cmd = f"{BENCHMARK_TOOL} "
    cmd += f"--graph={MODEL_PATH} "
    cmd += f"--warmup_runs={WARMUP_RUNS} "
    cmd += f"--num_runs={NUM_RUNS} "
    cmd += extra_args
    
    stdout, stderr, code = run_adb(cmd)
    
    if code != 0:
        print(f"❌ Benchmark failed!")
        print(f"Error: {stderr}")
        return None
    
    print(stdout)
    
    time_ms, fps = parse_inference_time(stdout)
    if time_ms:
        print(f"\n✅ Result: {time_ms:.2f} ms ({fps:.2f} FPS)")
        return {'name': config_name, 'time_ms': time_ms, 'fps': fps, 'output': stdout}
    else:
        print(f"⚠️  Could not parse timing")
        return None

def main():
    print("\n" + "="*70)
    print("  🚀 QIDK OPTIMIZED CPU BENCHMARK")
    print("  Testing different CPU configurations")
    print("="*70)
    
    results = []
    
    # Test 1: Baseline (4 threads, no XNNPack)
    result = run_benchmark(
        "Baseline: 4 threads",
        "--num_threads=4"
    )
    if result:
        results.append(result)
    
    # Test 2: 8 threads (use all cores)
    result = run_benchmark(
        "Optimized: 8 threads",
        "--num_threads=8"
    )
    if result:
        results.append(result)
    
    # Test 3: XNNPack delegate (4 threads)
    result = run_benchmark(
        "XNNPack: 4 threads",
        "--use_xnnpack=true --num_threads=4"
    )
    if result:
        results.append(result)
    
    # Test 4: XNNPack + 8 threads (BEST)
    result = run_benchmark(
        "XNNPack: 8 threads ⭐",
        "--use_xnnpack=true --num_threads=8"
    )
    if result:
        results.append(result)
    
    if not results:
        print("\n❌ No successful benchmarks!")
        return
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results_data = {
        'results': [{'name': r['name'], 'time_ms': r['time_ms'], 'fps': r['fps']} 
                   for r in results]
    }
    
    json_path = f"{OUTPUT_DIR}/optimized_cpu_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n✅ Saved results: {json_path}")
    
    # Create visualization
    create_visualization(results)
    
    # Print summary
    print("\n" + "="*70)
    print("  📊 RESULTS SUMMARY")
    print("="*70)
    
    baseline = results[0]
    best = min(results, key=lambda x: x['time_ms'])
    
    print(f"\n  Baseline: {baseline['time_ms']:.2f} ms ({baseline['fps']:.2f} FPS)")
    print(f"  Best:     {best['time_ms']:.2f} ms ({best['fps']:.2f} FPS)")
    
    speedup = baseline['time_ms'] / best['time_ms']
    print(f"\n  🚀 Speedup: {speedup:.2f}x faster!")
    
    if best['fps'] >= 25:
        print(f"  ✅ REAL-TIME CAPABLE! ({best['fps']:.1f} FPS > 25 FPS threshold)")
    elif best['fps'] >= 20:
        print(f"  ✓ Near real-time ({best['fps']:.1f} FPS, close to 25 FPS)")
    else:
        print(f"  ⚠️  Below real-time ({best['fps']:.1f} FPS < 25 FPS)")
    
    print(f"\n  📊 Graph: {OUTPUT_DIR}/optimized_cpu_comparison.png")
    print("="*70 + "\n")

def create_visualization(results):
    """Create comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QIDK CPU Optimization Results\nYOLO11n INT8 (1024x1024)', 
                 fontsize=16, fontweight='bold')
    
    names = [r['name'] for r in results]
    times = [r['time_ms'] for r in results]
    fps_vals = [r['fps'] for r in results]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # 1. Inference Time
    ax1 = axes[0, 0]
    bars = ax1.barh(names, times, color=colors[:len(results)])
    ax1.set_xlabel('Inference Time (ms)', fontweight='bold')
    ax1.set_title('Inference Time Comparison', fontweight='bold')
    ax1.invert_yaxis()
    
    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2, 
                f' {time:.1f} ms', va='center', fontweight='bold')
    
    # 2. FPS
    ax2 = axes[0, 1]
    bars = ax2.barh(names, fps_vals, color=colors[:len(results)])
    ax2.set_xlabel('Frames Per Second', fontweight='bold')
    ax2.set_title('Throughput (FPS)', fontweight='bold')
    ax2.axvline(x=25, color='red', linestyle='--', label='Real-time (25 FPS)', linewidth=2)
    ax2.legend()
    ax2.invert_yaxis()
    
    for bar, fps in zip(bars, fps_vals):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f' {fps:.1f} FPS', va='center', fontweight='bold')
    
    # 3. Speedup
    ax3 = axes[1, 0]
    baseline_time = times[0]
    speedups = [baseline_time / t for t in times]
    bars = ax3.barh(names, speedups, color=colors[:len(results)])
    ax3.set_xlabel('Speedup vs Baseline', fontweight='bold')
    ax3.set_title('Acceleration Factor', fontweight='bold')
    ax3.invert_yaxis()
    
    for bar, speedup in zip(bars, speedups):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2,
                f' {speedup:.2f}x', va='center', fontweight='bold')
    
    # 4. Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [['Configuration', 'Time (ms)', 'FPS', 'Speedup']]
    for r, speedup in zip(results, speedups):
        table_data.append([
            r['name'].replace(' ⭐', ''),
            f"{r['time_ms']:.1f}",
            f"{r['fps']:.1f}",
            f"{speedup:.2f}x"
        ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data)):
        for j in range(4):
            table[(i, j)].set_facecolor(colors[i-1] if i-1 < len(colors) else '#ecf0f1')
    
    ax4.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    png_path = f"{OUTPUT_DIR}/optimized_cpu_comparison.png"
    pdf_path = f"{OUTPUT_DIR}/optimized_cpu_comparison.pdf"
    
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"✅ Saved graph: {png_path}")
    
    plt.close()

if __name__ == "__main__":
    main()
