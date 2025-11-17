# All Paths to Convert YOLO11 to DLC Format

## Current Situation
- **Model**: YOLO11 (best.pt)
- **Problem**: YOLO11 architecture incompatible with QNN 2.40.0
- **Error**: "Unable to broadcast shapes" in C2f module Add operations
- **Goal**: Get quantized DLC for QIDK NPU

---

## Path 1: Downgrade to YOLOv8 Architecture ⭐ RECOMMENDED

### Why This Works
- YOLOv8 has better QNN support (tested and working)
- Same training data, just different backbone
- Your trained weights can be converted/retrained
- QNN 2.40.0 fully supports YOLOv8

### Steps

#### Option 1A: Export as YOLOv8 from Ultralytics
```python
from ultralytics import YOLO

# Load your trained YOLO11 model
model = YOLO('runs/detect/train/weights/best.pt')

# Check if direct conversion is possible
model.export(format='onnx', simplify=True, opset=12)

# If that creates compatible ONNX, convert to DLC
```

#### Option 1B: Retrain with YOLOv8 Architecture
```bash
# Install YOLOv8
pip install ultralytics==8.0.196  # Stable YOLOv8 version

# Create training script
cat > train_yolov8.py << 'EOF'
from ultralytics import YOLO

# Load YOLOv8 model (not YOLO11)
model = YOLO('yolov8n.pt')  # Use v8, not v11

# Train with your existing dataset
results = model.train(
    data='construction_data.yaml',
    epochs=100,
    imgsz=1024,
    batch=8,
    name='yolov8_construction',
    device='cpu'
)

# Export to ONNX
model.export(format='onnx', simplify=True, opset=12)
EOF

python train_yolov8.py
```

#### Convert YOLOv8 ONNX to DLC
```bash
# Activate environment
conda activate dlc
source ~/snpe-sdk/2.40.0.251030/bin/envsetup.sh

# Convert to DLC with quantization
qnn-onnx-converter \
  --input_network runs/detect/yolov8_construction/weights/best.onnx \
  --output_path yolov8_best_int8.cpp \
  --input_list calibration_list_raw.txt \
  --param_quantizer tf \
  --act_quantizer tf \
  --weight_bw 8 \
  --act_bw 8 \
  --use_per_channel_quantization

# Should work! YOLOv8 is compatible with QNN
```

**Pros:**
- ✅ Highest success rate (YOLOv8 tested with QNN)
- ✅ Full quantization support
- ✅ Official Qualcomm support
- ✅ Can reuse training data

**Cons:**
- ⚠️ Requires retraining (1-2 hours on CPU, 10-20 min on GPU)
- ⚠️ Slightly different architecture than YOLO11

**Time Estimate**: 2-3 hours (retraining + conversion)

---

## Path 2: Update QNN SDK to Latest Version

### Why This Might Work
- Newer QNN versions may have YOLO11 support
- QNN 2.50+ might include better ONNX operator coverage
- Latest SDK has improved shape inference

### Steps

#### Check Latest QNN Version
```bash
# Check Qualcomm's website
# https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct

# Current: QNN 2.40.0.251030
# Latest:  QNN 2.50.0+ (check website)
```

#### Download and Install Latest QNN
```bash
# Download from Qualcomm Developer Network
# Requires account: https://qdn.qualcomm.com/

# Extract
unzip qnn-v2.XX.X.XXXXXX.zip -d ~/snpe-sdk/

# Setup environment
export SNPE_ROOT=~/snpe-sdk/2.XX.X.XXXXXX
source $SNPE_ROOT/bin/envsetup.sh

# Install dependencies
cd $SNPE_ROOT
python3 -m pip install -r requirements.txt
```

#### Test YOLO11 Conversion with New SDK
```bash
conda activate dlc

qnn-onnx-converter \
  --input_network runs/detect/train/weights/best_simplified.onnx \
  --output_path best_yolo11_new_sdk.cpp \
  --input_list calibration_list_raw.txt \
  --param_quantizer tf \
  --act_quantizer tf \
  --weight_bw 8 \
  --act_bw 8 \
  --use_per_channel_quantization
```

**Pros:**
- ✅ No model retraining needed
- ✅ Uses your existing YOLO11 model
- ✅ Latest features and bug fixes

**Cons:**
- ❌ Might still not support YOLO11
- ⚠️ Requires Qualcomm account
- ⚠️ May have compatibility issues with QIDK

**Time Estimate**: 1-2 hours (download + test)
**Success Rate**: ~40% (uncertain if YOLO11 supported)

---

## Path 3: Simplify ONNX Graph with Custom Tools

### Why This Might Work
- Current error is in shape inference
- ONNX simplification might resolve broadcast issues
- Can manually modify problematic operations

### Steps

#### Install ONNX Optimization Tools
```bash
conda activate dlc

pip install onnx==1.15.0 \
            onnxsim==0.4.33 \
            onnx-simplifier \
            onnxruntime==1.16.0 \
            onnxoptimizer
```

#### Simplify ONNX Graph
```python
# Create simplify_onnx_advanced.py
import onnx
from onnxsim import simplify
import onnxoptimizer

# Load model
model = onnx.load('runs/detect/train/weights/best_simplified.onnx')

# Apply aggressive simplification
model_simp, check = simplify(
    model,
    skip_shape_inference=False,
    skip_optimization=False,
    skip_fuse_bn=False,
    overwrite_input_shapes={'images': [1, 3, 1024, 1024]}
)

# Apply ONNX optimizer passes
passes = [
    'eliminate_identity',
    'eliminate_nop_transpose',
    'eliminate_nop_pad',
    'eliminate_unused_initializer',
    'extract_constant_to_initializer',
    'fuse_add_bias_into_conv',
    'fuse_bn_into_conv',
    'fuse_consecutive_reduce',
    'fuse_consecutive_squeezes',
    'fuse_consecutive_transposes',
    'fuse_matmul_add_bias_into_gemm',
    'fuse_pad_into_conv',
    'fuse_transpose_into_gemm',
]

model_optimized = onnxoptimizer.optimize(model_simp, passes)

# Save
onnx.save(model_optimized, 'best_ultra_simplified.onnx')
print("✅ Ultra-simplified ONNX saved")
```

```bash
python simplify_onnx_advanced.py
```

#### Try Conversion with Simplified Model
```bash
qnn-onnx-converter \
  --input_network best_ultra_simplified.onnx \
  --output_path best_simplified_int8.cpp \
  --input_list calibration_list_raw.txt \
  --param_quantizer tf \
  --act_quantizer tf \
  --weight_bw 8 \
  --act_bw 8 \
  --use_per_channel_quantization
```

**Pros:**
- ✅ No retraining needed
- ✅ Uses existing YOLO11 model
- ✅ Might resolve shape inference issues

**Cons:**
- ⚠️ May still fail (same fundamental issue)
- ⚠️ Optimization might break model accuracy
- ⚠️ Requires testing after simplification

**Time Estimate**: 30 minutes
**Success Rate**: ~30% (optimistic)

---

## Path 4: Manual ONNX Graph Surgery

### Why This Might Work
- Identify exact problematic nodes
- Replace incompatible operations with equivalents
- Manually fix broadcast shape issues

### Steps

#### Analyze Problematic Nodes
```python
# Create analyze_yolo11_nodes.py
import onnx
import numpy as np

model = onnx.load('runs/detect/train/weights/best_simplified.onnx')

print("=== Analyzing YOLO11 ONNX Graph ===\n")

# Find all Add operations (the problematic ones)
add_nodes = [node for node in model.graph.node if node.op_type == 'Add']
print(f"Found {len(add_nodes)} Add operations:")

for i, node in enumerate(add_nodes):
    print(f"\n{i+1}. Node: {node.name}")
    print(f"   Inputs: {node.input}")
    print(f"   Outputs: {node.output}")
    
    # Try to find input shapes
    for input_name in node.input:
        for value_info in model.graph.value_info:
            if value_info.name == input_name:
                shape = [d.dim_value for d in value_info.type.tensor_type.shape.dim]
                print(f"   Input '{input_name}' shape: {shape}")

# Find C2f modules (contain the problematic Add ops)
print("\n=== C2f Module Analysis ===")
c2f_nodes = [node for node in model.graph.node if 'model.2' in node.name or 'C2f' in node.name]
print(f"Found {len(c2f_nodes)} C2f-related nodes")
for node in c2f_nodes[:10]:  # First 10
    print(f"  {node.op_type}: {node.name}")

print("\n=== Recommendation ===")
print("The error occurs at: /model.2/m.0/Add")
print("This is inside YOLO11's C2f bottleneck module")
print("Possible fixes:")
print("  1. Replace C2f with simpler operation")
print("  2. Modify Add operation broadcast behavior")
print("  3. Use YOLOv8 which has compatible C2f implementation")
```

```bash
python analyze_yolo11_nodes.py
```

#### Modify ONNX Graph
```python
# Create fix_yolo11_graph.py
import onnx
from onnx import helper, numpy_helper

model = onnx.load('runs/detect/train/weights/best_simplified.onnx')

# Find and modify problematic Add node
for node in model.graph.node:
    if node.name == '/model.2/m.0/Add':
        print(f"Found problematic node: {node.name}")
        
        # Option 1: Replace with Mul + Add sequence
        # Option 2: Insert explicit Reshape before Add
        # Option 3: Remove residual connection (may affect accuracy)
        
        # This requires deep understanding of ONNX graph structure
        # and YOLO11 architecture

# Save modified model
onnx.save(model, 'best_fixed.onnx')
```

**Pros:**
- ✅ Targeted fix for specific issue
- ✅ No retraining needed

**Cons:**
- ❌ Complex and error-prone
- ❌ Requires deep ONNX expertise
- ❌ May break model accuracy
- ❌ Time-consuming debugging

**Time Estimate**: 4-8 hours
**Success Rate**: ~20% (very uncertain)

---

## Path 5: Convert via Intermediate Format

### Why This Might Work
- Convert ONNX → TensorFlow → TFLite → DLC
- Or ONNX → PyTorch → ONNX (v8 compatible) → DLC
- Intermediate conversion might resolve issues

### Steps

#### Path 5A: ONNX → TensorFlow → TFLite → DLC
```bash
# Already done - we have TFLite models
# But QNN can convert TFLite to DLC

conda activate dlc
source ~/snpe-sdk/2.40.0.251030/bin/envsetup.sh

# Convert TFLite to DLC
qnn-tflite-converter \
  --input_network best_yolo_tflite/best_simplified_float32.tflite \
  --output_path best_from_tflite.cpp \
  --input_list calibration_list_raw.txt \
  --param_quantizer tf \
  --act_quantizer tf \
  --weight_bw 8 \
  --act_bw 8
```

**Note**: This previously failed due to library conflicts, but worth retrying in clean env.

#### Path 5B: PyTorch → CoreML → ONNX → DLC
```python
# Create export_via_coreml.py
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

# Export to CoreML (Apple's format)
model.export(format='coreml', nms=True)

# Then convert CoreML → ONNX using coremltools
import coremltools as ct

coreml_model = ct.models.MLModel('best.mlmodel')
onnx_model = ct.converters.onnx.convert(coreml_model)
# ... then convert ONNX → DLC
```

**Pros:**
- ✅ Alternative conversion path
- ✅ Might bypass YOLO11 issues

**Cons:**
- ⚠️ Complex multi-step process
- ⚠️ May introduce new errors
- ⚠️ Library dependency hell

**Time Estimate**: 2-3 hours
**Success Rate**: ~25%

---

## Path 6: Use SNPE Instead of QNN

### Why This Might Work
- SNPE (older SDK) sometimes has different operator support
- Different conversion pipeline than QNN
- Your SDK includes both: `snpe-onnx-to-dlc`

### Steps

#### Try SNPE Converter
```bash
conda activate dlc
source ~/snpe-sdk/2.40.0.251030/bin/envsetup.sh

# Use SNPE converter instead of QNN
snpe-onnx-to-dlc \
  --input_network runs/detect/train/weights/best_simplified.onnx \
  --output_path best_snpe.dlc \
  --input_dim images 1,3,1024,1024

# If successful, quantize separately
snpe-dlc-quantize \
  --input_dlc best_snpe.dlc \
  --input_list calibration_list_raw.txt \
  --output_dlc best_snpe_int8.dlc \
  --use_enhanced_quantizer \
  --use_per_channel_quantization
```

**Note**: You already tried this and got same error, but worth checking with different flags.

**Pros:**
- ✅ Different conversion engine
- ✅ Same SDK, no download needed

**Cons:**
- ⚠️ Likely same result (already failed)
- ⚠️ SNPE is older, less maintained

**Time Estimate**: 30 minutes
**Success Rate**: ~15%

---

## Path 7: Compile Custom QNN Operators

### Why This Might Work
- Implement custom C2f operation for QNN
- Register custom operator in QNN
- Most advanced but most control

### Steps

#### Create Custom QNN Operator
```cpp
// custom_c2f_op.cpp
// Implement YOLO11 C2f module as custom QNN operation
// This requires:
// 1. Understanding QNN custom op API
// 2. C++ implementation of C2f forward pass
// 3. Registration with QNN runtime

// See: $SNPE_ROOT/share/QNN/OpPackageGenerator/
```

```bash
# Compile custom operator package
cd $SNPE_ROOT/share/QNN/OpPackageGenerator/
# Follow Qualcomm documentation to create custom op
```

**Pros:**
- ✅ Maximum control
- ✅ Can handle any operation

**Cons:**
- ❌ Extremely complex
- ❌ Requires C++ and QNN expertise
- ❌ Very time-consuming
- ❌ Documentation is limited

**Time Estimate**: 1-2 weeks
**Success Rate**: ~60% (if you have expertise)

---

## Path 8: Wait for Model Architecture Fix

### Contact Ultralytics or Qualcomm
```bash
# File issue with Ultralytics
# https://github.com/ultralytics/ultralytics/issues

# Contact Qualcomm support
# https://qdn.qualcomm.com/support
```

**Pros:**
- ✅ Official solution

**Cons:**
- ❌ Slow (weeks/months)
- ❌ No guarantee

---

## RECOMMENDED STRATEGY: Multi-Path Approach

### Priority 1: YOLOv8 Conversion (Highest Success)
```bash
# 1. Retrain with YOLOv8
pip install ultralytics==8.0.196
python train_yolov8.py  # See Path 1B script above

# 2. Export to ONNX
# (automatic in training script)

# 3. Convert to DLC
qnn-onnx-converter \
  --input_network runs/detect/yolov8_construction/weights/best.onnx \
  --output_path yolov8_int8.cpp \
  --input_list calibration_list_raw.txt \
  --param_quantizer tf \
  --act_quantizer tf \
  --weight_bw 8 \
  --act_bw 8 \
  --use_per_channel_quantization

# Expected: SUCCESS ✅
```

### Priority 2: While YOLOv8 Trains, Try ONNX Simplification
```bash
# In parallel, attempt Path 3
pip install onnxsim onnxoptimizer
python simplify_onnx_advanced.py
qnn-onnx-converter --input_network best_ultra_simplified.onnx ...
```

### Priority 3: Check for QNN Updates
```bash
# Check Qualcomm website for QNN 2.50+
# If available, download and test
```

---

## Quick Decision Matrix

| Path | Success Rate | Time | Effort | Recommended |
|------|-------------|------|---------|-------------|
| **1. YOLOv8 Retrain** | **90%** | 2-3h | Medium | ⭐⭐⭐⭐⭐ |
| 2. Update QNN SDK | 40% | 1-2h | Low | ⭐⭐⭐ |
| 3. ONNX Simplify | 30% | 30m | Low | ⭐⭐ |
| 4. Manual Graph Surgery | 20% | 4-8h | Very High | ⭐ |
| 5. Intermediate Format | 25% | 2-3h | High | ⭐⭐ |
| 6. SNPE Converter | 15% | 30m | Low | ⭐ |
| 7. Custom Operators | 60% | 1-2w | Extreme | ⭐ (experts only) |
| 8. Wait for Fix | ??? | weeks | None | ❌ |

---

## My Recommendation

### Start Now (5 minutes):
```bash
# Install YOLOv8 and start training
pip install ultralytics==8.0.196
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='construction_data.yaml', epochs=100, imgsz=1024, batch=8, device='cpu', name='yolov8_dlc')
"
```

### While Training (30 minutes):
Try ONNX simplification (Path 3) as backup

### After Training (10 minutes):
Convert YOLOv8 ONNX → DLC (should work!)

### Total Time to DLC: **~3 hours**

Would you like me to:
1. **Start the YOLOv8 training now?** (recommended)
2. **Try ONNX simplification first?** (quick test)
3. **Check for QNN SDK updates?** (may require account)
