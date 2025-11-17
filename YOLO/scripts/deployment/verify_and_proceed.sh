#!/bin/bash

# Post-SNPE Installation Summary
# Run this after installing SNPE SDK to verify everything is ready

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear

echo -e "${BLUE}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║     NPU DEPLOYMENT - POST-INSTALLATION VERIFICATION          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo ""
echo "Checking installation status..."
echo ""

# Check SNPE_ROOT
if [ -z "$SNPE_ROOT" ]; then
    echo -e "${RED}❌ SNPE_ROOT not set${NC}"
    echo ""
    echo "Please run:"
    echo "  source ~/.bashrc"
    echo ""
    echo "Then run this script again."
    exit 1
else
    echo -e "${GREEN}✅ SNPE_ROOT=$SNPE_ROOT${NC}"
fi

# Check snpe-onnx-to-dlc
if command -v snpe-onnx-to-dlc &> /dev/null; then
    echo -e "${GREEN}✅ snpe-onnx-to-dlc available${NC}"
else
    echo -e "${RED}❌ snpe-onnx-to-dlc not found in PATH${NC}"
    exit 1
fi

# Check snpe-dlc-quantize
if command -v snpe-dlc-quantize &> /dev/null; then
    echo -e "${GREEN}✅ snpe-dlc-quantize available${NC}"
else
    echo -e "${RED}❌ snpe-dlc-quantize not found in PATH${NC}"
    exit 1
fi

# Check snpe-dlc-info
if command -v snpe-dlc-info &> /dev/null; then
    echo -e "${GREEN}✅ snpe-dlc-info available${NC}"
else
    echo -e "${RED}❌ snpe-dlc-info not found in PATH${NC}"
    exit 1
fi

# Check required files
echo ""
echo "Checking required files..."
echo ""

if [ -f "best_simplified.onnx" ]; then
    SIZE=$(ls -lh best_simplified.onnx | awk '{print $5}')
    echo -e "${GREEN}✅ best_simplified.onnx ($SIZE)${NC}"
else
    echo -e "${RED}❌ best_simplified.onnx not found${NC}"
    exit 1
fi

if [ -f "calibration_list.txt" ]; then
    COUNT=$(wc -l < calibration_list.txt)
    echo -e "${GREEN}✅ calibration_list.txt ($COUNT images)${NC}"
else
    echo -e "${RED}❌ calibration_list.txt not found${NC}"
    exit 1
fi

if [ -f "convert_to_dlc.sh" ]; then
    echo -e "${GREEN}✅ convert_to_dlc.sh${NC}"
else
    echo -e "${RED}❌ convert_to_dlc.sh not found${NC}"
fi

if [ -f "test_dlc_on_pc.py" ]; then
    echo -e "${GREEN}✅ test_dlc_on_pc.py${NC}"
else
    echo -e "${RED}❌ test_dlc_on_pc.py not found${NC}"
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ All checks passed! Ready to convert ONNX to DLC          ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo "1️⃣  Convert ONNX to DLC (5-15 minutes):"
echo "   ${BLUE}./convert_to_dlc.sh${NC}"
echo ""
echo "2️⃣  Test on PC with CPU (1-2 minutes):"
echo "   ${BLUE}./test_dlc_on_pc.py --dlc best_yolo_int8.dlc --image val/images/IMG_20240916_112859.jpg${NC}"
echo ""
echo "3️⃣  When ready, connect QIDK and test on NPU:"
echo "   See: ${BLUE}QIDK_NPU_TESTING_GUIDE.md${NC}"
echo ""

read -p "Do you want to start conversion now? (y/n): " start_conversion

if [ "$start_conversion" == "y" ]; then
    echo ""
    echo -e "${YELLOW}Starting conversion...${NC}"
    echo ""
    ./convert_to_dlc.sh
else
    echo ""
    echo "You can start conversion anytime by running:"
    echo "  ./convert_to_dlc.sh"
    echo ""
fi
