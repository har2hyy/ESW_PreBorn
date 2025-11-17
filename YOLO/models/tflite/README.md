# TFLite Models

This folder contains YOLOv11 models in TensorFlow Lite format for mobile/edge deployment.

## Files

Located in `best_yolo_tflite/`:
- **best_simplified_float32.tflite** - 11 MB, full precision
- **best_simplified_float16.tflite** - 5.2 MB, half precision
- **saved_model.pb** - TensorFlow SavedModel format
- **variables/** - Model weights

## Model Specifications
- **Input**: [1, 1024, 1024, 3], float32 (NHWC format)
- **Output**: [1, 9, 21504], float32
- **Quantization**: float32 or float16

## Usage

### Load in Python
```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='best_yolo_tflite/best_simplified_float32.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")   # [1, 1024, 1024, 3]
print(f"Input dtype: {input_details[0]['dtype']}")   # float32

# Prepare input (NHWC format: batch, height, width, channels)
input_data = np.random.randn(1, 1024, 1024, 3).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print(f"Output shape: {output_data.shape}")  # (1, 9, 21504)
```

### Android Deployment

#### Option 1: Standard TFLite (CPU)
```java
import org.tensorflow.lite.Interpreter;

Interpreter.Options opts = new Interpreter.Options();
opts.setNumThreads(4);
opts.setUseNNAPI(false);  // CPU only

Interpreter tflite = new Interpreter(loadModelFile("best_simplified_float32.tflite"), opts);
```

#### Option 2: With NNAPI (NPU/GPU Acceleration)
```java
Interpreter.Options opts = new Interpreter.Options();
opts.setUseNNAPI(true);  // Enable NPU/GPU acceleration

Interpreter tflite = new Interpreter(loadModelFile("best_simplified_float32.tflite"), opts);
```

#### Option 3: Hexagon Delegate (Qualcomm NPU - BEST)
```java
import org.tensorflow.lite.HexagonDelegate;

Interpreter.Options opts = new Interpreter.Options();
HexagonDelegate hexagonDelegate = new HexagonDelegate(context);
opts.addDelegate(hexagonDelegate);

Interpreter tflite = new Interpreter(loadModelFile("best_simplified_float32.tflite"), opts);
```

**See**: `../../docs/FIX_TFLITE_NPU.md` for fixing NPU fallback issues

## Conversion from ONNX

### Using onnx2tf
```bash
# Install dependencies
pip install onnx2tf onnxruntime tensorflow

# Convert ONNX to TFLite
onnx2tf -i ../onnx/best_simplified.onnx -o best_yolo_tflite

# Generate float16 version
onnx2tf -i ../onnx/best_simplified.onnx -o best_yolo_tflite \
        -oiqt -qt float16
```

### Create INT8 Quantized Version (for NPU)
```python
import tensorflow as tf
import cv2
import numpy as np

# Load float32 model
converter = tf.lite.TFLiteConverter.from_saved_model('best_yolo_tflite')

# Enable INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide calibration data
def representative_dataset():
    for i in range(100):
        img = cv2.imread(f'../../calibration/calibration_raw/image_{i}.jpg')
        img = cv2.resize(img, (1024, 1024))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        yield [np.expand_dims(img, axis=0)]

converter.representative_dataset = representative_dataset

# Force INT8 inputs/outputs
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert
tflite_model = converter.convert()

# Save
with open('best_yolo_tflite/best_int8.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ INT8 quantized model created")
```

## Model Comparison

| Model | Size | Precision | NPU Compatible | Speed (Qualcomm) |
|-------|------|-----------|----------------|------------------|
| float32.tflite | 11 MB | Full | ⚠️ (GPU only) | 300-500ms |
| float16.tflite | 5.2 MB | Half | ⚠️ (GPU only) | 200-400ms |
| int8.tflite | ~3 MB | 8-bit | ✅ Yes | 50-150ms |

## Performance Tips

1. **For CPU**: Use float32 model with `setNumThreads(4)`
2. **For GPU**: Use float16 model with GPU delegate
3. **For NPU**: Use INT8 quantized model with Hexagon Delegate
4. **Always enable NNAPI** on Qualcomm devices: `opts.setUseNNAPI(true)`

## Troubleshooting

**Problem**: Model runs on CPU instead of NPU
**Solution**: 
1. Use INT8 quantized model
2. Enable NNAPI or Hexagon Delegate
3. See `../../docs/FIX_TFLITE_NPU.md`

**Problem**: Poor accuracy after quantization
**Solution**: Provide more calibration images (100-1000 samples)

**Problem**: Slow inference on Android
**Solution**: Check if NNAPI is enabled and model is INT8
