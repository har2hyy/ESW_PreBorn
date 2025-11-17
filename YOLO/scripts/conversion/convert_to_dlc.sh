#!/bin/bash
###############################################################################
# Quick Start: ONNX to DLC Conversion
###############################################################################
#
# This script automates the complete DLC conversion workflow:
#   1. ONNX → FP32 DLC
#   2. FP32 DLC → INT8 DLC (with calibration)
#   3. Validation on PC
#
# Usage:
#   ./convert_to_dlc.sh
#
# Requirements:
#   - SNPE SDK installed (SNPE_ROOT set)
#   - best_simplified.onnx exists
#   - calibration_list.txt exists
#
###############################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ONNX to DLC Conversion${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check SNPE SDK
if [ -z "$SNPE_ROOT" ]; then
    echo -e "${RED}✗ SNPE_ROOT not set!${NC}"
    echo "  Please install SNPE SDK and run:"
    echo "  export SNPE_ROOT=/path/to/snpe"
    echo "  export PATH=\$SNPE_ROOT/bin/x86_64-linux-clang:\$PATH"
    exit 1
fi

if ! command -v snpe-onnx-to-dlc &> /dev/null; then
    echo -e "${RED}✗ snpe-onnx-to-dlc not found!${NC}"
    echo "  Check your SNPE SDK installation"
    exit 1
fi

echo -e "${GREEN}✓${NC} SNPE SDK: $SNPE_ROOT"
echo ""

# Paths
ONNX_MODEL="runs/detect/train/weights/best_simplified.onnx"
FP32_DLC="runs/detect/train/weights/best_yolo_fp32.dlc"
INT8_DLC="runs/detect/train/weights/best_yolo_int8.dlc"
CALIBRATION="calibration_list.txt"

# Check input files
if [ ! -f "$ONNX_MODEL" ]; then
    echo -e "${RED}✗ ONNX model not found: $ONNX_MODEL${NC}"
    exit 1
fi

if [ ! -f "$CALIBRATION" ]; then
    echo -e "${YELLOW}⚠ Calibration list not found: $CALIBRATION${NC}"
    echo "  INT8 quantization will be skipped"
    SKIP_INT8=true
fi

echo -e "${GREEN}✓${NC} Input files ready"
echo ""

# Step 1: ONNX → FP32 DLC
echo "[Step 1/4] Converting ONNX to FP32 DLC..."
echo "  Input:  $ONNX_MODEL"
echo "  Output: $FP32_DLC"
echo ""

snpe-onnx-to-dlc \
  --input_network "$ONNX_MODEL" \
  --output_path "$FP32_DLC" \
  --input_dim images 1,3,1024,1024

if [ $? -eq 0 ]; then
    FILE_SIZE=$(du -h "$FP32_DLC" | cut -f1)
    echo -e "${GREEN}✓${NC} FP32 DLC created: $FILE_SIZE"
else
    echo -e "${RED}✗ Conversion failed!${NC}"
    exit 1
fi
echo ""

# Step 2: Inspect FP32 DLC
echo "[Step 2/4] Inspecting FP32 DLC..."
snpe-dlc-info --input_dlc "$FP32_DLC" | head -n 30
echo ""
echo -e "${GREEN}✓${NC} Inspection complete"
echo ""

# Step 3: Quantize to INT8
if [ "$SKIP_INT8" != true ]; then
    echo "[Step 3/4] Quantizing to INT8 DLC..."
    echo "  Input:  $FP32_DLC"
    echo "  Output: $INT8_DLC"
    echo "  Calibration: $CALIBRATION ($(wc -l < $CALIBRATION) images)"
    echo ""
    echo "  This may take 5-15 minutes..."
    echo ""
    
    snpe-dlc-quantize \
      --input_dlc "$FP32_DLC" \
      --output_dlc "$INT8_DLC" \
      --input_list "$CALIBRATION" \
      --enable_htp \
      --use_enhanced_quantizer
    
    if [ $? -eq 0 ]; then
        FILE_SIZE=$(du -h "$INT8_DLC" | cut -f1)
        echo -e "${GREEN}✓${NC} INT8 DLC created: $FILE_SIZE"
    else
        echo -e "${RED}✗ Quantization failed!${NC}"
        exit 1
    fi
    echo ""
    
    # Inspect INT8 DLC
    echo "  Inspecting INT8 DLC..."
    snpe-dlc-info --input_dlc "$INT8_DLC" | head -n 30
    echo ""
else
    echo "[Step 3/4] Skipping INT8 quantization (no calibration data)"
    echo ""
fi

# Step 4: Test on PC
echo "[Step 4/4] Testing DLC on PC..."
echo ""

if [ -f test_dlc_on_pc.py ]; then
    if [ -f "$INT8_DLC" ]; then
        echo "  Testing INT8 DLC on CPU..."
        python test_dlc_on_pc.py "$INT8_DLC" --runtime cpu || true
    else
        echo "  Testing FP32 DLC on CPU..."
        python test_dlc_on_pc.py "$FP32_DLC" --runtime cpu || true
    fi
else
    echo -e "${YELLOW}⚠ test_dlc_on_pc.py not found, skipping PC test${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Conversion Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [ -f "$INT8_DLC" ]; then
    echo "DLC Files Created:"
    echo "  FP32: $FP32_DLC ($(du -h "$FP32_DLC" | cut -f1))"
    echo "  INT8: $INT8_DLC ($(du -h "$INT8_DLC" | cut -f1))"
    echo ""
    echo "Next steps:"
    echo "  1. Setup QIDK device:"
    echo "     ./setup_qidk_device.sh"
    echo ""
    echo "  2. Run remote pipeline:"
    echo "     ./remote_npu_pipeline.py --image /path/to/image.jpg --dlc $INT8_DLC"
else
    echo "DLC File Created:"
    echo "  FP32: $FP32_DLC ($(du -h "$FP32_DLC" | cut -f1))"
    echo ""
    echo "To create INT8 DLC:"
    echo "  1. Prepare calibration_list.txt with image paths"
    echo "  2. Run this script again"
fi
echo ""
