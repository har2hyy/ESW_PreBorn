#!/bin/bash
# Quick conversion script: PyTorch → ONNX → TFLite → QNN/DLC

set -e  # Exit on error

echo "🚀 YOLO Model Conversion Pipeline"
echo "=================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PYTORCH_MODEL="models/pytorch/best.pt"
ONNX_DIR="models/onnx"
TFLITE_DIR="models/tflite"
QNN_DIR="models/qnn_dlc"

# Check if PyTorch model exists
if [ ! -f "$PYTORCH_MODEL" ]; then
    echo -e "${RED}❌ PyTorch model not found: $PYTORCH_MODEL${NC}"
    echo "Please train the model first: cd scripts/training && python train.py"
    exit 1
fi

echo -e "${GREEN}✅ Found PyTorch model: $PYTORCH_MODEL${NC}"

# Menu
echo ""
echo "Select conversion target:"
echo "1) ONNX (cross-platform)"
echo "2) TFLite (mobile - float32/float16)"
echo "3) TFLite INT8 (mobile NPU)"
echo "4) QNN/DLC FP32 (QIDK NPU)"
echo "5) QNN/DLC INT8 (QIDK NPU - needs YOLOv8)"
echo "6) ALL (convert to all formats)"
echo "0) Exit"
echo ""
read -p "Enter choice [0-6]: " choice

case $choice in
    1)
        echo -e "${YELLOW}📦 Converting PyTorch → ONNX...${NC}"
        python3 -c "
from ultralytics import YOLO
model = YOLO('$PYTORCH_MODEL')
model.export(format='onnx', simplify=True, opset=12)
print('✅ ONNX export complete')
"
        mv runs/detect/train/weights/*.onnx $ONNX_DIR/ 2>/dev/null || true
        echo -e "${GREEN}✅ ONNX model saved to: $ONNX_DIR/${NC}"
        ;;
    
    2)
        echo -e "${YELLOW}📦 Converting ONNX → TFLite (FP32/FP16)...${NC}"
        if [ ! -f "$ONNX_DIR/best_simplified.onnx" ]; then
            echo -e "${RED}❌ ONNX model not found. Converting PyTorch → ONNX first...${NC}"
            python3 -c "
from ultralytics import YOLO
model = YOLO('$PYTORCH_MODEL')
model.export(format='onnx', simplify=True, opset=12)
"
            mv runs/detect/train/weights/*.onnx $ONNX_DIR/ 2>/dev/null || true
        fi
        
        pip install -q onnx2tf tensorflow >/dev/null 2>&1
        
        # Float32
        echo "Creating float32 model..."
        onnx2tf -i $ONNX_DIR/best_simplified.onnx -o $TFLITE_DIR/best_yolo_tflite
        
        # Float16
        echo "Creating float16 model..."
        onnx2tf -i $ONNX_DIR/best_simplified.onnx -o $TFLITE_DIR/best_yolo_tflite_fp16 -oiqt -qt float16
        
        echo -e "${GREEN}✅ TFLite models saved to: $TFLITE_DIR/${NC}"
        ;;
    
    3)
        echo -e "${YELLOW}📦 Converting to TFLite INT8 (with calibration)...${NC}"
        if [ ! -d "calibration/calibration_raw" ]; then
            echo -e "${YELLOW}⚠️  Calibration data not found. Preparing...${NC}"
            cd calibration
            python prepare_calibration_data.py
            cd ..
        fi
        
        python3 << 'EOF'
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

print("Loading TFLite float32 model...")
converter = tf.lite.TFLiteConverter.from_saved_model('models/tflite/best_yolo_tflite')

# Enable INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Calibration dataset
def representative_dataset():
    calib_dir = Path('calibration/calibration_raw')
    raw_files = sorted(calib_dir.glob('*.raw'))[:100]  # Use 100 samples
    
    for raw_file in raw_files:
        # Load raw file (already preprocessed)
        data = np.fromfile(raw_file, dtype=np.float32)
        data = data.reshape(1, 3, 1024, 1024)
        # Convert NCHW → NHWC for TFLite
        data = np.transpose(data, (0, 2, 3, 1))
        yield [data.astype(np.float32)]

converter.representative_dataset = representative_dataset

# Force INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

print("Converting to INT8... (this may take a few minutes)")
tflite_model = converter.convert()

# Save
output_path = 'models/tflite/best_yolo_tflite/best_int8.tflite'
Path(output_path).write_bytes(tflite_model)
print(f"✅ INT8 model saved: {output_path}")
print(f"   Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
EOF
        echo -e "${GREEN}✅ INT8 quantized TFLite model created${NC}"
        ;;
    
    4)
        echo -e "${YELLOW}📦 Converting ONNX → QNN/DLC (FP32)...${NC}"
        if [ ! -f "$ONNX_DIR/best_simplified.onnx" ]; then
            echo -e "${RED}❌ ONNX model not found. Run option 1 first.${NC}"
            exit 1
        fi
        
        # Check if QNN is installed
        if [ -z "$SNPE_ROOT" ]; then
            echo -e "${YELLOW}⚠️  QNN/SNPE SDK not configured. Setting up...${NC}"
            if [ -d "$HOME/snpe-sdk/2.40.0.251030" ]; then
                export SNPE_ROOT=$HOME/snpe-sdk/2.40.0.251030
                source $SNPE_ROOT/bin/envsetup.sh
            else
                echo -e "${RED}❌ QNN SDK not found at ~/snpe-sdk/2.40.0.251030${NC}"
                echo "See: docs/INSTALL_SNPE_SDK.md"
                exit 1
            fi
        fi
        
        cd scripts/conversion
        ./run_qnn_converter.sh
        cd ../..
        echo -e "${GREEN}✅ QNN FP32 model created${NC}"
        ;;
    
    5)
        echo -e "${RED}⚠️  YOLO11 INT8 quantization not supported by QNN 2.40.0${NC}"
        echo ""
        echo "Recommended solutions:"
        echo "1. Train with YOLOv8 instead (90% success rate)"
        echo "2. Use TFLite INT8 with Hexagon Delegate"
        echo "3. Update to QNN 2.50+ (if available)"
        echo ""
        echo "See: docs/ALL_DLC_CONVERSION_PATHS.md for all options"
        ;;
    
    6)
        echo -e "${YELLOW}📦 Converting to ALL formats...${NC}"
        
        # ONNX
        echo -e "\n${YELLOW}[1/4] PyTorch → ONNX${NC}"
        $0 <<< "1"
        
        # TFLite FP32/FP16
        echo -e "\n${YELLOW}[2/4] ONNX → TFLite (FP32/FP16)${NC}"
        $0 <<< "2"
        
        # TFLite INT8
        echo -e "\n${YELLOW}[3/4] TFLite INT8${NC}"
        $0 <<< "3"
        
        # QNN FP32
        echo -e "\n${YELLOW}[4/4] ONNX → QNN/DLC (FP32)${NC}"
        $0 <<< "4"
        
        echo -e "\n${GREEN}✅ All conversions complete!${NC}"
        ;;
    
    0)
        echo "Exiting..."
        exit 0
        ;;
    
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 Conversion complete!${NC}"
echo ""
echo "Next steps:"
echo "- Test models: cd scripts/testing"
echo "- Deploy to device: cd scripts/deployment"
echo "- Read docs: ls docs/"
