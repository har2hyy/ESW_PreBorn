#!/bin/bash

# SNPE SDK Installation Helper Script
# This script helps you install and configure SNPE SDK

set -e

echo "============================================"
echo "  SNPE SDK Installation Helper"
echo "============================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if SNPE is already installed
if [ -n "$SNPE_ROOT" ] && [ -d "$SNPE_ROOT" ]; then
    echo -e "${GREEN}✅ SNPE SDK already installed at: $SNPE_ROOT${NC}"
    echo ""
    snpe-onnx-to-dlc --help > /dev/null 2>&1 && echo -e "${GREEN}✅ SNPE tools are working${NC}" || echo -e "${RED}❌ SNPE tools not accessible${NC}"
    echo ""
    read -p "Do you want to reinstall/reconfigure? (y/n): " choice
    if [ "$choice" != "y" ]; then
        echo "Exiting..."
        exit 0
    fi
fi

echo ""
echo -e "${YELLOW}SNPE SDK is NOT installed. Let's set it up!${NC}"
echo ""
echo "Please follow these steps:"
echo ""
echo "1️⃣  Download SNPE SDK from Qualcomm:"
echo "   URL: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk"
echo ""
echo "2️⃣  You need to:"
echo "   - Create account / Login"
echo "   - Accept license agreement"
echo "   - Download SNPE 2.x (recommended) or QNN 2.x"
echo "   - File will be: snpe-2.x.x.xxxx.zip (size: ~500MB - 1.5GB)"
echo ""
echo "3️⃣  Save the downloaded file to ~/Downloads/"
echo ""

read -p "Have you downloaded the SNPE SDK .zip file? (y/n): " downloaded

if [ "$downloaded" != "y" ]; then
    echo ""
    echo -e "${YELLOW}Please download the SNPE SDK first, then run this script again.${NC}"
    echo ""
    echo "Quick steps:"
    echo "  1. Visit: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk"
    echo "  2. Login/Register"
    echo "  3. Download SNPE 2.x.x.zip"
    echo "  4. Run this script again"
    exit 0
fi

echo ""
echo "Looking for SNPE SDK in ~/Downloads/..."
echo ""

# Find SNPE zip file
SNPE_ZIP=$(ls ~/Downloads/snpe-*.zip 2>/dev/null | head -1)

if [ -z "$SNPE_ZIP" ]; then
    echo -e "${RED}❌ No SNPE SDK .zip file found in ~/Downloads/${NC}"
    echo ""
    echo "Please ensure the downloaded file is in ~/Downloads/ and named like: snpe-2.x.x.xxxx.zip"
    exit 1
fi

echo -e "${GREEN}✅ Found: $SNPE_ZIP${NC}"
echo ""

# Ask for installation location
echo "Choose installation location:"
echo "  1) /opt/qcom/aistack/snpe (System-wide, requires sudo)"
echo "  2) ~/snpe-sdk (User directory, no sudo needed)"
echo ""
read -p "Enter choice (1 or 2): " install_choice

if [ "$install_choice" == "1" ]; then
    INSTALL_DIR="/opt/qcom/aistack/snpe"
    echo ""
    echo "Installing to: $INSTALL_DIR"
    echo -e "${YELLOW}Note: This will require sudo password${NC}"
    echo ""
    
    # Create directory
    sudo mkdir -p /opt/qcom/aistack
    
    # Extract
    echo "Extracting $SNPE_ZIP..."
    cd ~/Downloads
    unzip -q "$SNPE_ZIP"
    
    # Find extracted directory
    EXTRACTED_DIR=$(ls -d snpe-* 2>/dev/null | grep -v ".zip" | head -1)
    
    if [ -z "$EXTRACTED_DIR" ]; then
        echo -e "${RED}❌ Failed to extract SNPE SDK${NC}"
        exit 1
    fi
    
    # Move to /opt
    echo "Moving to $INSTALL_DIR..."
    sudo mv "$EXTRACTED_DIR" "$INSTALL_DIR"
    
    SNPE_ROOT="$INSTALL_DIR"
    
elif [ "$install_choice" == "2" ]; then
    INSTALL_DIR="$HOME/snpe-sdk"
    echo ""
    echo "Installing to: $INSTALL_DIR"
    echo ""
    
    # Extract
    echo "Extracting $SNPE_ZIP..."
    cd ~/Downloads
    unzip -q "$SNPE_ZIP"
    
    # Find extracted directory
    EXTRACTED_DIR=$(ls -d snpe-* 2>/dev/null | grep -v ".zip" | head -1)
    
    if [ -z "$EXTRACTED_DIR" ]; then
        echo -e "${RED}❌ Failed to extract SNPE SDK${NC}"
        exit 1
    fi
    
    # Move to home
    echo "Moving to $INSTALL_DIR..."
    mv "$EXTRACTED_DIR" "$INSTALL_DIR"
    
    SNPE_ROOT="$INSTALL_DIR"
else
    echo -e "${RED}Invalid choice${NC}"
    exit 1
fi

echo -e "${GREEN}✅ SNPE SDK installed at: $SNPE_ROOT${NC}"
echo ""

# Setup environment variables
echo "Setting up environment variables..."
echo ""

# Add to bashrc
echo "# SNPE SDK Environment Variables" >> ~/.bashrc
echo "export SNPE_ROOT=$SNPE_ROOT" >> ~/.bashrc
echo 'export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH' >> ~/.bashrc

# Export for current session
export SNPE_ROOT="$SNPE_ROOT"
export PATH="$SNPE_ROOT/bin/x86_64-linux-clang:$PATH"
export LD_LIBRARY_PATH="$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"

echo -e "${GREEN}✅ Environment variables configured${NC}"
echo ""

# Verify installation
echo "Verifying installation..."
echo ""

if [ -f "$SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc" ]; then
    echo -e "${GREEN}✅ snpe-onnx-to-dlc found${NC}"
else
    echo -e "${RED}❌ snpe-onnx-to-dlc not found${NC}"
fi

if [ -f "$SNPE_ROOT/bin/x86_64-linux-clang/snpe-dlc-quantize" ]; then
    echo -e "${GREEN}✅ snpe-dlc-quantize found${NC}"
else
    echo -e "${RED}❌ snpe-dlc-quantize not found${NC}"
fi

if [ -f "$SNPE_ROOT/bin/x86_64-linux-clang/snpe-dlc-info" ]; then
    echo -e "${GREEN}✅ snpe-dlc-info found${NC}"
else
    echo -e "${RED}❌ snpe-dlc-info not found${NC}"
fi

echo ""

# Test command
echo "Testing SNPE tools..."
if $SNPE_ROOT/bin/x86_64-linux-clang/snpe-onnx-to-dlc --help > /dev/null 2>&1; then
    echo -e "${GREEN}✅ SNPE tools are working!${NC}"
else
    echo -e "${YELLOW}⚠️  SNPE tools found but may need Python dependencies${NC}"
    echo ""
    echo "Installing Python dependencies..."
    pip install numpy onnx protobuf
fi

echo ""
echo "============================================"
echo -e "${GREEN}✅ SNPE SDK Installation Complete!${NC}"
echo "============================================"
echo ""
echo "SNPE_ROOT: $SNPE_ROOT"
echo ""
echo "⚠️  IMPORTANT: Reload your bashrc or open a new terminal:"
echo "   source ~/.bashrc"
echo ""
echo "Next steps:"
echo "  1. Open a new terminal (or run: source ~/.bashrc)"
echo "  2. Verify: echo \$SNPE_ROOT"
echo "  3. Convert model: cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO && ./convert_to_dlc.sh"
echo ""
