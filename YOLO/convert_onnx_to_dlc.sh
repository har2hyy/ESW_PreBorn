#!/bin/bash
# Script to convert YOLO11 ONNX to SNPE DLC format
# Handles Python 3.10 requirement for SNPE SDK

set -e

echo "========================================="
echo "  YOLO11 ONNX → SNPE DLC Converter"
echo "========================================="

# SNPE SDK path
export SNPE_ROOT=/home/harshyy/snpe-sdk/2.40.0.251030
export PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH
export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH

# Paths
ONNX_INPUT="./models/pytorch/best.onnx"
DLC_OUTPUT="./models/onnx/best.dlc"

echo ""
echo "Input ONNX:  $ONNX_INPUT"
echo "Output DLC:  $DLC_OUTPUT"
echo "SNPE_ROOT:   $SNPE_ROOT"
echo ""

# Check if ONNX exists
if [ ! -f "$ONNX_INPUT" ]; then
    echo "❌ Error: ONNX file not found at $ONNX_INPUT"
    exit 1
fi

# Create output directory
mkdir -p ./models/onnx

echo "Running snpe-onnx-to-dlc..."
echo ""

# Check if we need Python 3.10
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "Current Python: $PYTHON_VERSION"

# Try to find or create Python 3.10 environment
if command -v conda &> /dev/null; then
    echo "Conda found. Checking for Python 3.10 environment..."
    
    # Check if snpe310 env exists
    if conda env list | grep -q "^snpe310 "; then
        echo "Using existing snpe310 conda environment"
        PYTHON_CMD="conda run -n snpe310 --no-capture-output"
    else
        echo "Creating Python 3.10 conda environment (snpe310)..."
        conda create -n snpe310 python=3.10 -y
        PYTHON_CMD="conda run -n snpe310 --no-capture-output"
    fi
    
    # Run converter through conda env
    $PYTHON_CMD snpe-onnx-to-dlc \
        --input_network "$ONNX_INPUT" \
        --output_path "$DLC_OUTPUT"
else
    echo "Conda not found. Attempting direct conversion..."
    echo "Note: This may fail if system Python is not 3.10"
    
    snpe-onnx-to-dlc \
        --input_network "$ONNX_INPUT" \
        --output_path "$DLC_OUTPUT"
fi

# Check result
if [ -f "$DLC_OUTPUT" ]; then
    DLC_SIZE=$(du -h "$DLC_OUTPUT" | cut -f1)
    echo ""
    echo "========================================="
    echo "✅ DLC conversion successful!"
    echo "   Output: $DLC_OUTPUT"
    echo "   Size: $DLC_SIZE"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "❌ DLC conversion failed"
    echo "   No output file created"
    echo "========================================="
    exit 1
fi
