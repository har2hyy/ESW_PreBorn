#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(__file__))
from integrated_pipeline_tflite_int8 import YOLOv11TFLiteINT8
import cv2

model_path = '../YOLO/runs/detect/train/weights/best_saved_model/best_float32.tflite'
classes_path = '../YOLO/classes.txt'

yolo = YOLOv11TFLiteINT8(model_path, classes_path, conf_threshold=0.51)

img = cv2.imread('/home/harshyy/Desktop/20250103_104457.jpg')
print(f'Image shape: {img.shape}')

boxes, confs, class_ids = yolo.predict(img)

print(f'\nDetected {len(boxes)} objects')
for i, (box, conf, cid) in enumerate(zip(boxes, confs, class_ids)):
    print(f'  {i+1}. {yolo.class_names[cid]}: {conf:.3f} at {box}')
