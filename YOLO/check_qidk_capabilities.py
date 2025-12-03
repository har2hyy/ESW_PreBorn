#!/usr/bin/env python3
"""
QIDK Device Capability Checker
Diagnoses what hardware acceleration is available
"""

import subprocess
import json
import re

def run_adb(cmd):
    """Run ADB command and return output"""
    full_cmd = f"adb shell \"{cmd}\""
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip()

def check_adb_connected():
    """Verify ADB connection"""
    stdout, _ = run_adb("echo 'connected'")
    return "connected" in stdout

def get_device_info():
    """Get basic device information"""
    info = {}
    
    # Android version
    info['android_version'], _ = run_adb("getprop ro.build.version.release")
    info['android_sdk'], _ = run_adb("getprop ro.build.version.sdk")
    
    # Device model
    info['manufacturer'], _ = run_adb("getprop ro.product.manufacturer")
    info['model'], _ = run_adb("getprop ro.product.model")
    info['device'], _ = run_adb("getprop ro.product.device")
    
    # Qualcomm platform
    info['platform'], _ = run_adb("getprop ro.board.platform")
    info['hardware'], _ = run_adb("getprop ro.hardware")
    
    # SoC info
    info['soc_id'], _ = run_adb("cat /sys/devices/soc0/soc_id 2>/dev/null")
    info['chip_name'], _ = run_adb("cat /sys/devices/soc0/chip_name 2>/dev/null")
    
    return info

def get_cpu_info():
    """Get CPU information"""
    cpu_info = {}
    
    # CPU architecture
    stdout, _ = run_adb("cat /proc/cpuinfo")
    
    # Count processors
    processors = len(re.findall(r'processor\s+:\s+\d+', stdout))
    cpu_info['num_processors'] = processors
    
    # Get CPU implementer and architecture
    impl_match = re.search(r'CPU implementer\s+:\s+(\w+)', stdout)
    arch_match = re.search(r'CPU architecture\s+:\s+(\w+)', stdout)
    part_match = re.search(r'CPU part\s+:\s+(\w+)', stdout)
    
    cpu_info['implementer'] = impl_match.group(1) if impl_match else "unknown"
    cpu_info['architecture'] = arch_match.group(1) if arch_match else "unknown"
    cpu_info['part'] = part_match.group(1) if part_match else "unknown"
    
    # Hardware name
    hw_match = re.search(r'Hardware\s+:\s+(.+)', stdout)
    cpu_info['hardware_name'] = hw_match.group(1).strip() if hw_match else "unknown"
    
    return cpu_info

def check_neural_libraries():
    """Check for neural network libraries"""
    libraries = {}
    
    search_patterns = [
        ("NNAPI Core", "libneuralnetworks.so"),
        ("QNN HTP", "libQnnHtp.so"),
        ("QNN GPU", "libQnnGpu.so"),
        ("QNN CPU", "libQnnCpu.so"),
        ("Hexagon NN", "libhexagon_nn_skel.so"),
        ("SNPE Core", "libSNPE.so"),
        ("OpenCL", "libOpenCL.so"),
        ("Vulkan", "libvulkan.so"),
    ]
    
    for name, lib in search_patterns:
        stdout, _ = run_adb(f"find /vendor /system /apex -name '{lib}' 2>/dev/null")
        libraries[name] = stdout.split('\n') if stdout else []
    
    return libraries

def check_nnapi_accelerators():
    """Check available NNAPI accelerators"""
    # This requires the benchmark tool to be on device
    benchmark_path = "/data/local/tmp/qidk_benchmark/benchmark_model"
    model_path = "/data/local/tmp/qidk_benchmark/model.tflite"
    
    stdout, stderr = run_adb(f"{benchmark_path} --graph={model_path} --nnapi_accelerator_name=? 2>&1 || true")
    
    # Look for "Must be one of: {accelerators}"
    match = re.search(r'Must be one of:\s+\{([^}]+)\}', stdout + stderr)
    if match:
        accelerators = [a.strip() for a in match.group(1).split(',')]
        return accelerators
    return []

def check_gpu_info():
    """Get GPU information"""
    gpu_info = {}
    
    # OpenGL renderer (Adreno GPU info)
    stdout, _ = run_adb("dumpsys SurfaceFlinger | grep GLES")
    gpu_info['gles_info'] = stdout
    
    # Vulkan support
    stdout, _ = run_adb("pm list features | grep vulkan")
    gpu_info['vulkan_support'] = "yes" if stdout else "no"
    
    # OpenCL support
    stdout, _ = run_adb("ls /system/vendor/lib*/libOpenCL.so 2>/dev/null")
    gpu_info['opencl_support'] = "yes" if stdout else "no"
    
    return gpu_info

def check_dsp_support():
    """Check for DSP/NPU support"""
    dsp_info = {}
    
    # Hexagon DSP device nodes
    stdout, _ = run_adb("ls /dev/adsprpc* 2>/dev/null")
    dsp_info['adsprpc_devices'] = stdout.split('\n') if stdout else []
    
    # FastRPC support
    stdout, _ = run_adb("ls /dev/fastrpc* 2>/dev/null")
    dsp_info['fastrpc_devices'] = stdout.split('\n') if stdout else []
    
    # Check for compute DSP
    stdout, _ = run_adb("getprop vendor.fastrpc.process.attrs")
    dsp_info['fastrpc_attrs'] = stdout
    
    return dsp_info

def check_selinux():
    """Check SELinux status (might block NPU access)"""
    stdout, _ = run_adb("getenforce")
    return stdout

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def main():
    print("\n" + "=" * 70)
    print("  🔍 QIDK DEVICE CAPABILITY DIAGNOSTIC")
    print("=" * 70)
    
    # Check ADB
    if not check_adb_connected():
        print("\n❌ ADB device not connected!")
        return
    print("\n✅ ADB connected")
    
    # Device Info
    print_section("📱 Device Information")
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key:20s}: {value}")
    
    # CPU Info
    print_section("🖥️  CPU Information")
    cpu_info = get_cpu_info()
    for key, value in cpu_info.items():
        print(f"  {key:20s}: {value}")
    
    # CPU Architecture Analysis
    impl = cpu_info.get('implementer', '')
    part = cpu_info.get('part', '')
    if impl == '0x41':  # ARM
        if part == '0xd80':
            print("\n  💡 Detected: ARM Cortex-A78 (High Performance)")
        elif part == '0xd40':
            print("\n  💡 Detected: ARM Cortex-A76")
        elif part == '0xd0c':
            print("\n  💡 Detected: ARM Cortex-A76AE")
    
    # GPU Info
    print_section("🎮 GPU Information")
    gpu_info = check_gpu_info()
    if gpu_info['gles_info']:
        print(f"  OpenGL ES: {gpu_info['gles_info']}")
    print(f"  Vulkan Support: {gpu_info['vulkan_support']}")
    print(f"  OpenCL Support: {gpu_info['opencl_support']}")
    
    # DSP/NPU Support
    print_section("🧠 DSP/NPU Support")
    dsp_info = check_dsp_support()
    if dsp_info['adsprpc_devices']:
        print("  ✅ ADSPRPC devices found:")
        for dev in dsp_info['adsprpc_devices']:
            if dev:
                print(f"     {dev}")
    else:
        print("  ❌ No ADSPRPC devices found")
    
    if dsp_info['fastrpc_devices']:
        print("  ✅ FastRPC devices found:")
        for dev in dsp_info['fastrpc_devices']:
            if dev:
                print(f"     {dev}")
    else:
        print("  ❌ No FastRPC devices found")
    
    # Neural Network Libraries
    print_section("📚 Neural Network Libraries")
    libraries = check_neural_libraries()
    for name, paths in libraries.items():
        if paths and paths[0]:
            print(f"  ✅ {name:20s}:")
            for path in paths:
                if path:
                    print(f"     {path}")
        else:
            print(f"  ❌ {name:20s}: Not found")
    
    # NNAPI Accelerators
    print_section("⚡ NNAPI Accelerators")
    try:
        accelerators = check_nnapi_accelerators()
        if accelerators:
            print(f"  Available accelerators: {', '.join(accelerators)}")
            
            if 'nnapi-reference' in accelerators and len(accelerators) == 1:
                print("\n  ⚠️  WARNING: Only 'nnapi-reference' available!")
                print("     This is a CPU fallback - NO GPU/NPU acceleration!")
            elif any('qti' in acc for acc in accelerators):
                print("\n  ✅ Qualcomm accelerators detected!")
        else:
            print("  ❌ Could not detect NNAPI accelerators")
            print("     (benchmark tool may not be on device)")
    except Exception as e:
        print(f"  ⚠️  Could not query NNAPI: {e}")
    
    # SELinux
    print_section("🔒 Security")
    selinux_status = check_selinux()
    print(f"  SELinux: {selinux_status}")
    if selinux_status == "Enforcing":
        print("  ⚠️  SELinux is enforcing - may block some accelerators")
    
    # Summary
    print_section("📊 SUMMARY & RECOMMENDATIONS")
    
    # Analyze what's available
    has_qnn = any(libraries.get('QNN HTP', []))
    has_snpe = any(libraries.get('SNPE Core', []))
    has_hexagon = any(libraries.get('Hexagon NN', []))
    has_opencl = gpu_info['opencl_support'] == 'yes'
    has_vulkan = gpu_info['vulkan_support'] == 'yes'
    has_dsp_device = bool(dsp_info['adsprpc_devices'] or dsp_info['fastrpc_devices'])
    
    print("\n  Hardware Capabilities:")
    print(f"    CPU cores: {cpu_info.get('num_processors', 'unknown')}")
    print(f"    GPU (OpenCL): {'✅ Yes' if has_opencl else '❌ No'}")
    print(f"    GPU (Vulkan): {'✅ Yes' if has_vulkan else '❌ No'}")
    print(f"    DSP devices: {'✅ Yes' if has_dsp_device else '❌ No'}")
    
    print("\n  Software/Runtime Support:")
    print(f"    QNN libraries: {'✅ Yes' if has_qnn else '❌ No'}")
    print(f"    SNPE libraries: {'✅ Yes' if has_snpe else '❌ No'}")
    print(f"    Hexagon NN: {'✅ Yes' if has_hexagon else '❌ No'}")
    
    print("\n  Recommended Approach:")
    
    if has_qnn or has_snpe:
        print("\n    ⭐ USE SNPE/QNN RUNTIME (Best option)")
        print("       - Convert model to DLC format")
        print("       - Use snpe-net-run or qnn-net-run")
        print("       - Direct access to NPU/DSP/GPU")
        print("       - Bypass broken NNAPI")
        
        print("\n    📝 Next steps:")
        print("       1. Convert TFLite → DLC")
        print("       2. Push SNPE/QNN runtime to device")
        print("       3. Run with --runtime dsp (NPU) or gpu")
    else:
        print("\n    💻 OPTIMIZE CPU PERFORMANCE")
        print("       - NPU/GPU libraries not found")
        print("       - Use XNNPack for CPU acceleration")
        print("       - Expected: 50-60ms with optimization")
        
        print("\n    📝 Next steps:")
        print("       1. Run with --use_xnnpack=true")
        print("       2. Increase threads (--num_threads=8)")
        print("       3. Consider installing SNPE/QNN libraries")
    
    if not has_dsp_device:
        print("\n    ⚠️  WARNING: No DSP/NPU device nodes found!")
        print("       Your device may not have Hexagon DSP enabled")
        print("       or the kernel doesn't expose it.")
    
    print("\n" + "=" * 70)
    print()

if __name__ == "__main__":
    main()
