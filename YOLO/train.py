# train.py
from ultralytics import YOLO

# Load a pre-trained model (yolo11n.pt is a good starting point)
model = YOLO('/home/harshyy/Desktop/ESW/yolo11n.pt')    # CHANGE THIS to your model path (where downloaded yolo11n.pt)

# Main training execution block
if __name__ == '__main__':
    # Train the model using your dataset configuration
    results = model.train(
        data='ESW_PreBorn/YOLO/construction_data.yaml',     # Path to your dataset config file (CHANGE THIS)
        epochs=150,      # A good number of epochs to start with
        imgsz=1024,       # Image size for training
        batch=8          # Adjust based on your GPU's VRAM
    )