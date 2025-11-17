# ONNX Models

This folder contains YOLOv11 models in ONNX format for cross-platform deployment.

## Files

- **best.onnx** - Original exported ONNX model
- **best_simplified.onnx** - Simplified ONNX model (recommended for deployment)

## Model Specifications
- **Input**: `images` - [1, 3, 1024, 1024], float32
- **Output**: [1, 9, 21504], float32 (4 bbox coords + 5 class scores per anchor)
- **Opset**: 12
- **IR Version**: 7

## Usage

### Test on PC with ONNX Runtime
```bash
cd ../../scripts/testing
python test_onnx_on_pc.py
```

### Load in Python
```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('best_simplified.onnx', 
                               providers=['CPUExecutionProvider'])

# Prepare input (1, 3, 1024, 1024) float32
input_data = np.random.randn(1, 3, 1024, 1024).astype(np.float32)

# Run inference
outputs = session.run(None, {'images': input_data})

print(f"Output shape: {outputs[0].shape}")  # (1, 9, 21504)
```

### Validate for NPU Deployment
```bash
cd ../../scripts/testing
python validate_onnx_for_npu.py
```

## Conversion from PyTorch

```bash
cd ../pytorch

# Using Ultralytics
python -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', simplify=True, opset=12)
"

# Or using export script
cd ../../scripts/training
python export_int8.py
```

## Next Steps

### Convert to TFLite
```bash
pip install onnx2tf
onnx2tf -i best_simplified.onnx -o ../tflite/best_tflite
```

### Convert to QNN/DLC
```bash
cd ../../scripts/conversion
./run_qnn_converter.sh
```

## Performance
- **PC (CPU)**: ~142ms per inference
- **PC (ONNX Runtime)**: 8 worker detections (validated)
- **Compatibility**: ✅ ONNX Runtime, ⚠️ QNN (YOLO11 incompatible)
