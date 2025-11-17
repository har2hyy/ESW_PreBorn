#!/bin/bash
###############################################################################
# Setup QIDK Device for NPU Inference
###############################################################################
#
# This script prepares your QIDK device for running YOLO models on the NPU.
# It pushes SNPE runtime libraries and creates the necessary directory structure.
#
# Usage:
#   chmod +x setup_qidk_device.sh
#   ./setup_qidk_device.sh [device_serial]
#
# Requirements:
#   - ADB installed and in PATH
#   - QIDK connected via USB
#   - SNPE SDK downloaded and extracted
#
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}QIDK NPU Setup Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if ADB is installed
if ! command -v adb &> /dev/null; then
    echo -e "${RED}✗ ADB not found!${NC}"
    echo "  Install: sudo apt install adb"
    exit 1
fi

echo -e "${GREEN}✓${NC} ADB found"

# Check for connected devices
DEVICES=$(adb devices | grep -w "device" | awk '{print $1}')
if [ -z "$DEVICES" ]; then
    echo -e "${RED}✗ No ADB devices found!${NC}"
    echo "  Connect QIDK via USB and enable USB debugging"
    exit 1
fi

# Select device
if [ -n "$1" ]; then
    DEVICE=$1
else
    DEVICE=$(echo "$DEVICES" | head -n 1)
fi

echo -e "${GREEN}✓${NC} Using device: ${DEVICE}"
echo ""

# Check SNPE_ROOT
if [ -z "$SNPE_ROOT" ]; then
    echo -e "${YELLOW}⚠${NC}  SNPE_ROOT not set!"
    echo "  Please enter SNPE SDK path (e.g., /opt/qcom/aistack/snpe/2.x.x.xxxx):"
    read -r SNPE_ROOT
    
    if [ ! -d "$SNPE_ROOT" ]; then
        echo -e "${RED}✗ Invalid SNPE path: $SNPE_ROOT${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓${NC} SNPE SDK: ${SNPE_ROOT}"
echo ""

# Create directory structure on device
echo "[1/4] Creating directory structure..."
adb -s "$DEVICE" shell "mkdir -p /data/local/tmp/yolo_npu"
adb -s "$DEVICE" shell "mkdir -p /data/local/tmp/yolo_npu/output"
echo -e "${GREEN}✓${NC} Directories created"
echo ""

# Push SNPE runtime libraries
echo "[2/4] Pushing SNPE runtime libraries..."

SNPE_LIBS=(
    "libSNPE.so"
    "libhta.so"
    "libhta_hexagon_runtime.so"
)

LIB_DIR="$SNPE_ROOT/lib/aarch64-android"

if [ ! -d "$LIB_DIR" ]; then
    echo -e "${YELLOW}⚠${NC}  aarch64-android libs not found, trying arm64-v8a..."
    LIB_DIR="$SNPE_ROOT/lib/arm64-v8a"
fi

if [ ! -d "$LIB_DIR" ]; then
    echo -e "${RED}✗ Could not find SNPE libraries in:${NC}"
    echo "  $SNPE_ROOT/lib/aarch64-android"
    echo "  $SNPE_ROOT/lib/arm64-v8a"
    exit 1
fi

for lib in "${SNPE_LIBS[@]}"; do
    if [ -f "$LIB_DIR/$lib" ]; then
        echo "  Pushing $lib..."
        adb -s "$DEVICE" push "$LIB_DIR/$lib" /data/local/tmp/yolo_npu/ 2>&1 | grep -v "^$"
    else
        echo -e "${YELLOW}  ⚠ $lib not found (may not be required)${NC}"
    fi
done

echo -e "${GREEN}✓${NC} Libraries pushed"
echo ""

# Push snpe-net-run binary
echo "[3/4] Pushing snpe-net-run executable..."

BIN_DIR="$SNPE_ROOT/bin/aarch64-android"
if [ ! -d "$BIN_DIR" ]; then
    BIN_DIR="$SNPE_ROOT/bin/arm64-v8a"
fi

if [ -f "$BIN_DIR/snpe-net-run" ]; then
    adb -s "$DEVICE" push "$BIN_DIR/snpe-net-run" /data/local/tmp/yolo_npu/ 2>&1 | grep -v "^$"
    adb -s "$DEVICE" shell "chmod +x /data/local/tmp/yolo_npu/snpe-net-run"
    echo -e "${GREEN}✓${NC} snpe-net-run pushed and made executable"
else
    echo -e "${RED}✗ snpe-net-run not found in $BIN_DIR${NC}"
    exit 1
fi
echo ""

# Verify setup
echo "[4/4] Verifying setup..."

# Check if files exist on device
echo "  Checking device files..."
adb -s "$DEVICE" shell "ls -lh /data/local/tmp/yolo_npu/" 

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ QIDK Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Convert ONNX to DLC:"
echo "     cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO"
echo "     snpe-onnx-to-dlc --input_network best_simplified.onnx --output_path best_yolo_fp32.dlc --input_dim images 1,3,1024,1024"
echo ""
echo "  2. Quantize to INT8:"
echo "     snpe-dlc-quantize --input_dlc best_yolo_fp32.dlc --output_dlc best_yolo_int8.dlc --input_list calibration_list.txt --enable_htp"
echo ""
echo "  3. Test remote pipeline:"
echo "     python remote_npu_pipeline.py --image /path/to/image.jpg --dlc best_yolo_int8.dlc --device $DEVICE"
echo ""
