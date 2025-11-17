# 🎯 Complete NPU Deployment Workflow

## Current Status: ✅ Ready for SNPE SDK Installation

---

## 📊 Progress Overview

| Step | Task | Status | Time Required |
|------|------|--------|---------------|
| 1 | ONNX Model Validation | ✅ Complete | - |
| 2 | Workspace Cleanup | ✅ Complete | - |
| 3 | **SNPE SDK Installation** | ⏳ **PENDING** | 10-20 min |
| 4 | ONNX → DLC Conversion | ⏳ Pending | 5-15 min |
| 5 | PC Testing | ⏳ Pending | 1-2 min |
| 6 | QIDK NPU Testing | ⏳ Pending | 5-10 min |

---

## 🎬 What You Need to Do Now

### IMMEDIATE ACTIONS (Do These First)

#### ✅ Step 1: Download SNPE SDK

1. **Open browser** and visit:
   ```
   https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
   ```

2. **Login/Register** with Qualcomm Developer Network
   - Create account if you don't have one
   - It's free!

3. **Download SNPE 2.x**
   - Click "Download" button
   - Accept license agreement
   - Download: `snpe-2.x.x.xxxx.zip` (500MB - 1.5GB)
   - Save to: `~/Downloads/`

4. **Wait for download** to complete
   - Grab a coffee! ☕

---

#### ✅ Step 2: Install SNPE SDK

Once download completes, run:

```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO
./install_snpe_helper.sh
```

**What this does:**
- ✅ Finds the downloaded .zip file
- ✅ Extracts and installs SNPE SDK
- ✅ Configures environment variables automatically
- ✅ Verifies installation
- ✅ Installs Python dependencies

**Installation locations:**
- Option 1: `/opt/qcom/aistack/snpe` (system-wide, requires sudo)
- Option 2: `~/snpe-sdk` (user directory, no sudo)

Choose whichever you prefer!

---

#### ✅ Step 3: Verify Installation

After installation, **reload your terminal**:

```bash
source ~/.bashrc
```

Then verify:

```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO
./verify_and_proceed.sh
```

This will:
- ✅ Check if SNPE_ROOT is set
- ✅ Verify SNPE tools are accessible
- ✅ Confirm all required files exist
- ✅ Give you option to start conversion immediately

---

### AUTOMATED ACTIONS (Scripts Will Do These)

#### ✅ Step 4: Convert ONNX to DLC

**Automated script (recommended):**

```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO
./convert_to_dlc.sh
```

**What this does:**

1. **ONNX → FP32 DLC** (1-2 minutes)
   ```
   best_simplified.onnx → best_yolo_fp32.dlc (~10 MB)
   ```

2. **Inspect FP32 DLC** (verify layers)
   ```
   snpe-dlc-info shows model architecture
   ```

3. **FP32 → INT8 DLC** (5-15 minutes)
   ```
   best_yolo_fp32.dlc → best_yolo_int8.dlc (~2-3 MB)
   Using 150 calibration images
   ```

4. **Verify on PC** (validates model works)
   ```
   Test with CPU runtime
   Expected: 9-11 detections
   ```

**Outputs:**
- 📦 `best_yolo_fp32.dlc` - Full precision (~10 MB)
- 📦 `best_yolo_int8.dlc` - Quantized for NPU (~2-3 MB) ⭐

---

#### ✅ Step 5: Test on PC (No NPU Needed)

**Purpose:** Validate DLC files work correctly before deploying to QIDK

```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO

# Test INT8 DLC
./test_dlc_on_pc.py \
    --dlc best_yolo_int8.dlc \
    --image val/images/IMG_20240916_112859.jpg

# Expected output:
# ✅ 9-11 detections
# ✅ Inference time: 300-500ms (CPU)
# ✅ Saves: test_result.jpg with bounding boxes
```

**What this validates:**
- ✅ DLC file is valid
- ✅ Model produces correct detections
- ✅ Preprocessing/postprocessing works
- ✅ Ready for NPU deployment

---

## 🚀 QIDK NPU Testing (When You Connect Device)

### Prerequisites
- ✅ SNPE SDK installed
- ✅ DLC files created and tested on PC
- ✅ QIDK device powered on
- ✅ USB cable connected
- ✅ ADB installed (`sudo apt install android-tools-adb`)

---

### Quick NPU Testing (3 Commands)

```bash
cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO

# 1. Verify QIDK connection
adb devices
# Should show: XXXXXXXX    device

# 2. Setup QIDK (one-time)
./setup_qidk_device.sh

# 3. Run inference on NPU
./remote_npu_pipeline.py \
    --image val/images/IMG_20240916_112859.jpg \
    --dlc best_yolo_int8.dlc \
    --runtime dsp

# Expected:
# ✅ 9-11 detections
# ✅ Inference time: 50-100ms ⚡
# ✅ Output: remote_npu_result.jpg + remote_npu_result.json
```

---

### Detailed NPU Testing Guide

**See:** `QIDK_NPU_TESTING_GUIDE.md`

This comprehensive guide covers:
1. ✅ Connecting QIDK device (USB debugging setup)
2. ✅ ADB connection verification
3. ✅ Device setup (automated + manual options)
4. ✅ Running inference (automated + manual step-by-step)
5. ✅ Validating results
6. ✅ Performance monitoring
7. ✅ Troubleshooting all common issues
8. ✅ Batch processing for production

---

## 📁 All Files Created for You

### Documentation (5 files)
- 📄 `COMPLETE_WORKFLOW.md` ← **You are here**
- 📄 `INSTALL_SNPE_SDK.md` - Detailed installation guide
- 📄 `QIDK_NPU_TESTING_GUIDE.md` - Complete NPU testing guide
- 📄 `COMPLETE_NPU_DEPLOYMENT.md` - Full deployment documentation
- 📄 `QUICK_REFERENCE.txt` - Quick command reference

### Scripts (7 files)
- 🔧 `install_snpe_helper.sh` - Interactive SNPE installer ⭐
- 🔧 `verify_and_proceed.sh` - Post-installation verification ⭐
- 🔧 `convert_to_dlc.sh` - Automated ONNX → DLC conversion ⭐
- 🔧 `test_dlc_on_pc.py` - Test DLC on PC (CPU/GPU) ⭐
- 🔧 `setup_qidk_device.sh` - One-time QIDK setup
- 🔧 `remote_npu_pipeline.py` - Run inference on QIDK from PC ⭐
- 🔧 `integrated_pipeline_onnx.py` - Production ONNX pipeline

### Model Files
- 📦 `best_simplified.onnx` (10.2 MB) - Source model ✅
- 📦 `calibration_list.txt` (150 images) - For INT8 quantization ✅
- 📦 `best_yolo_fp32.dlc` (will be created) ⏳
- 📦 `best_yolo_int8.dlc` (will be created) ⏳

---

## ⏱️ Time Estimate

| Task | Time |
|------|------|
| Download SNPE SDK | 5-15 min (depends on internet) |
| Install SNPE SDK | 2-5 min |
| Convert ONNX → DLC | 5-15 min |
| Test on PC | 1-2 min |
| Setup QIDK | 2-3 min |
| Test on NPU | 1-2 min |
| **Total** | **~20-45 minutes** |

Most of this is automated! Just follow the steps above.

---

## 🎯 Expected Performance

| Platform | Runtime | Inference Time | Accuracy |
|----------|---------|----------------|----------|
| PC | CPU | 300-500ms | 100% (FP32) |
| PC | GPU | 150-200ms | 100% (FP32) |
| QIDK | NPU/DSP | **50-100ms** ⚡ | 95-99% (INT8) |
| QIDK | GPU | 150-200ms | 95-99% (INT8) |
| QIDK | CPU | 300-500ms | 95-99% (INT8) |

**Recommendation:** Use DSP/NPU runtime on QIDK for best performance!

---

## 🔄 Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     CURRENT STATUS: HERE                        │
│                            ⬇                                    │
│  1. Download SNPE SDK from Qualcomm                             │
│     https://developer.qualcomm.com/software/...                 │
│     File: snpe-2.x.x.xxxx.zip → ~/Downloads/                    │
│                            ⬇                                    │
│  2. Install SNPE SDK                                            │
│     ./install_snpe_helper.sh                                    │
│     → Extracts, installs, configures environment                │
│                            ⬇                                    │
│  3. Verify Installation                                         │
│     source ~/.bashrc                                            │
│     ./verify_and_proceed.sh                                     │
│                            ⬇                                    │
├─────────────────────────────────────────────────────────────────┤
│                     PC WORKFLOW (No QIDK)                       │
│                            ⬇                                    │
│  4. Convert ONNX to DLC                                         │
│     ./convert_to_dlc.sh                                         │
│     best_simplified.onnx ──→ best_yolo_fp32.dlc (~10 MB)        │
│                          └──→ best_yolo_int8.dlc (~2-3 MB) ⭐   │
│                            ⬇                                    │
│  5. Test on PC                                                  │
│     ./test_dlc_on_pc.py --dlc best_yolo_int8.dlc --image ...    │
│     Expected: 9-11 detections, 300-500ms on CPU                 │
│                            ⬇                                    │
├─────────────────────────────────────────────────────────────────┤
│                  QIDK WORKFLOW (Connect Device)                 │
│                            ⬇                                    │
│  6. Connect QIDK via USB                                        │
│     adb devices → Should show device                            │
│                            ⬇                                    │
│  7. Setup QIDK (one-time)                                       │
│     ./setup_qidk_device.sh                                      │
│     → Pushes SNPE libraries + binaries to device                │
│                            ⬇                                    │
│  8. Run Inference on NPU                                        │
│     ./remote_npu_pipeline.py --image ... --dlc ... --runtime dsp│
│                                                                 │
│     [PC] Preprocess image → resize, normalize, save input.raw   │
│       ⬇                                                         │
│     [ADB] Push best_yolo_int8.dlc + input.raw to QIDK           │
│       ⬇                                                         │
│     [QIDK NPU] Run inference (50-100ms) ⚡                      │
│       ⬇                                                         │
│     [ADB] Pull output.raw from QIDK                             │
│       ⬇                                                         │
│     [PC] Postprocess → NMS, visualize, save results             │
│                            ⬇                                    │
│  9. Validate Results                                            │
│     remote_npu_result.jpg - Visualization                       │
│     remote_npu_result.json - Detection data                     │
│     Expected: 9-11 detections, 50-100ms inference ✅            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🆘 Need Help?

### Quick Troubleshooting

**SNPE installation issues:**
- See: `INSTALL_SNPE_SDK.md`
- Common: SNPE_ROOT not set → Run `source ~/.bashrc`

**Conversion errors:**
- See: `COMPLETE_NPU_DEPLOYMENT.md` → Troubleshooting section
- Common: Python dependencies → Run `pip install numpy onnx protobuf`

**QIDK connection issues:**
- See: `QIDK_NPU_TESTING_GUIDE.md` → Step 1 & Troubleshooting
- Common: Device not detected → Check USB debugging enabled

**Performance issues:**
- See: `QIDK_NPU_TESTING_GUIDE.md` → Troubleshooting
- Common: Slow inference → Ensure using DSP runtime, not CPU

### Documentation Quick Access

- **Installation problems:** `INSTALL_SNPE_SDK.md`
- **Conversion problems:** `COMPLETE_NPU_DEPLOYMENT.md`
- **NPU testing problems:** `QIDK_NPU_TESTING_GUIDE.md`
- **Quick commands:** `QUICK_REFERENCE.txt`

---

## ✅ Success Criteria

You'll know everything is working when:

✅ **After SNPE Installation:**
- `echo $SNPE_ROOT` shows installation path
- `snpe-onnx-to-dlc --help` shows help message

✅ **After Conversion:**
- `best_yolo_fp32.dlc` exists (~10 MB)
- `best_yolo_int8.dlc` exists (~2-3 MB)
- `snpe-dlc-info -i best_yolo_int8.dlc` shows model info

✅ **After PC Testing:**
- `test_dlc_on_pc.py` completes without errors
- `test_result.jpg` shows 9-11 bounding boxes
- Detections include: workers, trucks, bikes, etc.

✅ **After NPU Testing:**
- `adb devices` shows QIDK device
- `remote_npu_pipeline.py` completes in <100ms
- `remote_npu_result.json` has 9-11 detections
- Results match PC testing (~95-99% accuracy)

---

## 🎯 START HERE

1. **Download SNPE SDK** from:
   ```
   https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
   ```

2. **Then run:**
   ```bash
   cd /home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO
   ./install_snpe_helper.sh
   ```

3. **Follow the prompts!**

Everything else is automated! 🚀

---

**Last Updated:** 2025-11-16
**Your Location:** `/home/harshyy/Desktop/ESW/ESW_PreBorn/YOLO/`
**All files ready:** ✅
