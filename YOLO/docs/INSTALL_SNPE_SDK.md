# SNPE SDK Installation Guide

## ⚠️ SNPE SDK Required

The SNPE SDK must be installed before converting ONNX models to DLC format.

## 📥 Download SNPE SDK

### Step 1: Register on Qualcomm Developer Network
1. Visit: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
2. Click "Download" button
3. Create account or login
4. Accept license agreement

### Step 2: Download SDK
- Download: **SNPE 2.x** (Snapdragon Neural Processing Engine)
  - OR **QNN 2.x** (Qualcomm AI Engine Direct)
- File size: ~500 MB - 1.5 GB
- Format: `.zip` file

## 🔧 Installation Steps

### Option 1: System-wide Installation (Recommended)

```bash
# 1. Extract the downloaded file
cd ~/Downloads
unzip snpe-*.zip

# 2. Move to /opt directory
sudo mkdir -p /opt/qcom/aistack
sudo mv snpe-* /opt/qcom/aistack/snpe

# 3. Set up environment variables
export SNPE_ROOT=/opt/qcom/aistack/snpe
export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH

# 4. Add to ~/.bashrc for persistence
echo 'export SNPE_ROOT=/opt/qcom/aistack/snpe' >> ~/.bashrc
echo 'export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH' >> ~/.bashrc

# 5. Reload bashrc
source ~/.bashrc
```

### Option 2: Local User Installation

```bash
# 1. Extract in home directory
cd ~/Downloads
unzip snpe-*.zip
mv snpe-* ~/snpe-sdk

# 2. Set up environment variables
export SNPE_ROOT=~/snpe-sdk
export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH

# 3. Add to ~/.bashrc
echo 'export SNPE_ROOT=~/snpe-sdk' >> ~/.bashrc
echo 'export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH' >> ~/.bashrc

# 4. Reload
source ~/.bashrc
```

## ✅ Verify Installation

```bash
# Check SNPE_ROOT
echo $SNPE_ROOT
# Should output: /opt/qcom/aistack/snpe or ~/snpe-sdk

# Check if tools are available
snpe-onnx-to-dlc --help
snpe-dlc-quantize --help
snpe-dlc-info --help

# Should show help messages for each command
```

## 🐍 Python Dependencies (if needed)

```bash
# Install Python dependencies for SNPE tools
pip install numpy onnx protobuf
```

## 🎯 After Installation - Next Steps

Once SNPE SDK is installed and verified:

```bash
# Go to YOLO directory
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO

# Run automated conversion
./convert_to_dlc.sh

# Or run manual conversion steps (see COMPLETE_NPU_DEPLOYMENT.md)
```

## 🔍 Troubleshooting

### "snpe-onnx-to-dlc: command not found"
- Check if SNPE_ROOT is set: `echo $SNPE_ROOT`
- Check if PATH includes SNPE bin: `echo $PATH | grep snpe`
- Reload bashrc: `source ~/.bashrc`

### "libSNPE.so: cannot open shared object file"
- Check LD_LIBRARY_PATH: `echo $LD_LIBRARY_PATH | grep snpe`
- Add to LD_LIBRARY_PATH: `export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH`

### "Permission denied"
- For /opt installation: Use `sudo` when moving files
- For home installation: No sudo needed

### Version mismatch
- SNPE 2.x and QNN 2.x both work
- Prefer latest stable version (2.10+ for SNPE, 2.18+ for QNN)

## 📚 Alternative: Use Docker (Advanced)

If installation issues persist, you can use SNPE in Docker:

```bash
# Pull SNPE Docker image (if available from Qualcomm)
docker pull qualcomm/snpe:latest

# Run conversion in container
docker run -v $(pwd):/workspace qualcomm/snpe snpe-onnx-to-dlc ...
```

## 📞 Support

- Qualcomm Developer Support: https://developer.qualcomm.com/support
- SNPE Documentation: https://developer.qualcomm.com/docs/snpe/overview.html
