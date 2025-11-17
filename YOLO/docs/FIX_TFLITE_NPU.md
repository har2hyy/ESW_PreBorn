# How to Fix TFLite INT8 CPU Fallback Issue

## Problem
Your TFLite INT8 model runs on CPU instead of NPU because:
1. NNAPI is disabled: `opts.setUseNNAPI(false);`
2. Missing Hexagon Delegate dependency
3. Using float32 model instead of INT8 quantized model

## Solution 1: Enable NNAPI (Simplest)

### Step 1: Update TFLiteRunner.java
```java
public TFLiteRunner(Context ctx) throws IOException {
    Interpreter.Options opts = new Interpreter.Options();
    opts.setNumThreads(4);
    
    // ENABLE NNAPI FOR NPU ACCELERATION
    opts.setUseNNAPI(true);  // ← Change from false to true
    
    tflite = new Interpreter(loadModelFile(ctx, MODEL_FILE), opts);
    // ... rest of code
}
```

### Step 2: Use INT8 Quantized Model
Replace `best_float32.tflite` with an INT8 quantized model:
```java
private static final String MODEL_FILE = "best_int8.tflite";  // ← Must be INT8
```

### Step 3: Verify NPU Usage
Add logging to confirm:
```java
android.util.Log.d("TFLiteRunner", "NNAPI enabled: " + opts.getUseNNAPI());
android.util.Log.d("TFLiteRunner", "Input dtype: " + tflite.getInputTensor(0).dataType());
// Should print: Input dtype: UINT8 (for quantized) or FLOAT32 (for float)
```

---

## Solution 2: Use Hexagon Delegate (Best Performance)

### Step 1: Add Hexagon Delegate Dependency
In `app/build.gradle`:
```gradle
dependencies {
    implementation "org.tensorflow:tensorflow-lite:2.13.0"
    implementation "org.tensorflow:tensorflow-lite-support:0.4.3"
    implementation "org.tensorflow:tensorflow-lite-hexagon:2.13.0"  // ← Add this
}
```

### Step 2: Update TFLiteRunner.java
```java
import org.tensorflow.lite.HexagonDelegate;

public class TFLiteRunner {
    private HexagonDelegate hexagonDelegate;
    
    public TFLiteRunner(Context ctx) throws IOException {
        Interpreter.Options opts = new Interpreter.Options();
        opts.setNumThreads(4);
        
        // Use Qualcomm Hexagon Delegate for NPU
        try {
            hexagonDelegate = new HexagonDelegate(ctx);
            opts.addDelegate(hexagonDelegate);
            android.util.Log.d("TFLiteRunner", "Hexagon Delegate initialized - using NPU");
        } catch (Exception e) {
            android.util.Log.w("TFLiteRunner", "Hexagon Delegate failed, falling back to NNAPI");
            opts.setUseNNAPI(true);
        }
        
        tflite = new Interpreter(loadModelFile(ctx, MODEL_FILE), opts);
        // ... rest of code
    }
    
    public void close() {
        if (tflite != null) tflite.close();
        if (hexagonDelegate != null) hexagonDelegate.close();  // ← Important!
    }
}
```

### Step 3: Copy Hexagon Libraries to Device
The Hexagon Delegate needs `.so` files on the device:
```bash
adb push $SNPE_ROOT/lib/aarch64-android/libhexagon_interface.so /data/local/tmp/
adb push $SNPE_ROOT/lib/hexagon-v68/unsigned/libqnnHtpV68Skel.so /vendor/lib/rfsa/adsp/
```

---

## Solution 3: Create INT8 Quantized TFLite Model

### Current Model Issue
Your current model:
```
best_float32.tflite  → Input/Output: float32 → Runs on CPU/GPU, NOT NPU
```

### Create INT8 Model
Run this Python script:
```python
import tensorflow as tf
import numpy as np
from pathlib import Path

# Load float32 model
converter = tf.lite.TFLiteConverter.from_saved_model("runs/detect/train/weights/best_saved_model")

# Enable INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for calibration
def representative_dataset():
    for i in range(100):
        img_path = f"calibration_images/image_{i}.jpg"
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1024, 1024))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        yield [img]

converter.representative_dataset = representative_dataset

# Force INT8 for inputs and outputs
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert
tflite_quant_model = converter.convert()

# Save
Path("best_int8.tflite").write_bytes(tflite_quant_model)
print("✅ INT8 quantized model created: best_int8.tflite")
```

---

## Verification Checklist

After applying the fix:

### 1. Check Model Type
```bash
# On PC
python -c "
import tensorflow as tf
interpreter = tf.lite.Interpreter('best_int8.tflite')
print('Input dtype:', interpreter.get_input_details()[0]['dtype'])
print('Output dtype:', interpreter.get_output_details()[0]['dtype'])
"
# Should print: dtype: <class 'numpy.uint8'> or <class 'numpy.int8'>
```

### 2. Check Android Logs
```bash
adb logcat -s TFLiteRunner:D
# Should see:
# "Hexagon Delegate initialized - using NPU"
# OR
# "NNAPI enabled: true"
```

### 3. Measure Inference Time
Before (CPU):
```
Inference time: 800-1200ms
```

After (NPU):
```
Inference time: 50-150ms  (8-10x faster!)
```

---

## Why DLC/QNN is Still Better

Even with proper TFLite setup:

| Feature | TFLite + Hexagon | QNN/DLC |
|---------|------------------|---------|
| Performance | Good (50-150ms) | Best (20-80ms) |
| Overhead | NNAPI/Delegate layer | Direct HTP access |
| Optimization | TensorFlow's generic | Qualcomm-optimized |
| Control | Limited | Full HTP configuration |
| Compatibility | Easier (works on multiple devices) | Qualcomm only |

**Recommendation**: 
- Use **TFLite + Hexagon** for quick deployment and testing
- Use **QNN/DLC** for production and best performance

---

## Quick Test Script

Create `test_tflite_backend.py`:
```python
import tensorflow as tf
import numpy as np
import time

model_path = "best_int8.tflite"  # or best_float32.tflite

interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Model: {model_path}")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Input dtype: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Output dtype: {output_details[0]['dtype']}")

# Test inference
dummy_input = np.random.randint(0, 255, input_details[0]['shape'], dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], dummy_input)

start = time.time()
interpreter.invoke()
inference_time = (time.time() - start) * 1000

print(f"Inference time: {inference_time:.2f}ms")
print(f"✅ Model loaded successfully")
```

Run on PC to verify INT8 model is correct:
```bash
python test_tflite_backend.py
```

Expected output:
```
Model: best_int8.tflite
Input shape: [1, 1024, 1024, 3]
Input dtype: <class 'numpy.uint8'>  ← Should be uint8, not float32!
Output shape: [1, 9, 21504]
Output dtype: <class 'numpy.uint8'>
Inference time: 245.32ms
✅ Model loaded successfully
```
