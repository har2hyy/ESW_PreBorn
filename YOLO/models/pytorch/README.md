# PyTorch Models

This folder contains YOLOv11 models in PyTorch format (.pt).

## Files

- **best.pt** - Your trained YOLOv11 model (1024x1024, 5 classes)
- **yolo11n.pt** - Pre-trained YOLOv11 nano model

## Usage

### Load and Test Model
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('best.pt')

# Run inference
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

### Export to Other Formats
```python
from ultralytics import YOLO

model = YOLO('best.pt')

# Export to ONNX
model.export(format='onnx', simplify=True, opset=12)

# Export to TFLite
model.export(format='tflite')

# Export to TensorRT
model.export(format='engine', device=0)
```

### Training
```bash
# See: scripts/training/train.py
cd ../../scripts/training
python train.py
```

## Model Info
- **Input Size**: 1024x1024
- **Classes**: 5 (worker, truck, bike, bulldozer, car)
- **Architecture**: YOLOv11n
- **Framework**: PyTorch / Ultralytics
