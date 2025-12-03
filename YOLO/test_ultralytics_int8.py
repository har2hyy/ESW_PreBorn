#!/usr/bin/env python3
"""
Test the Ultralytics-exported INT8 model from training
"""

import numpy as np
import tensorflow as tf
import cv2

# Load the Ultralytics-exported INT8 model
model_path = 'runs/detect/train/weights/best_int8.tflite'
print(f"Loading model: {model_path}")

interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# Get details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\n=== Model Info ===")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Input dtype: {input_details[0]['dtype']}")
print(f"Input quantization: {input_details[0]['quantization']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Output dtype: {output_details[0]['dtype']}")
print(f"Output quantization: {output_details[0]['quantization']}")

# Test 1: Zero input
print(f"\n=== Test 1: Zero Input ===")
test_input = np.zeros(input_details[0]['shape'], dtype=input_details[0]['dtype'])
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"Output min/max: {output.min()}/{output.max()}")
print(f"Unique values count: {len(np.unique(output))}")

# Dequantize
if output_details[0]['dtype'] == np.uint8:
    scale, zero_point = output_details[0]['quantization']
    output_float = scale * (output.astype(np.float32) - zero_point)
    output_sigmoid = 1 / (1 + np.exp(-output_float))
    
    # Transpose and get scores
    output_t = output_sigmoid.transpose(0, 2, 1)[0]
    class_scores = output_t[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    
    print(f"Max confidence range: {max_scores.min():.3f} - {max_scores.max():.3f}")
    print(f"Detections > 0.25: {(max_scores > 0.25).sum()}")

# Test 2: Real image
print(f"\n=== Test 2: Real Image ===")
img = cv2.imread('test_images/my_optimal_result.jpg')
if img is not None:
    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 1024))
    
    if input_details[0]['dtype'] == np.uint8:
        img_input = img_resized.astype(np.uint8)
    else:
        img_input = (img_resized / 255.0).astype(np.float32)
    
    img_input = np.expand_dims(img_input, axis=0)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Output min/max: {output.min()}/{output.max()}")
    print(f"Unique values count: {len(np.unique(output))}")
    
    # Dequantize
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output_float = scale * (output.astype(np.float32) - zero_point)
        output_sigmoid = 1 / (1 + np.exp(-output_float))
        
        # Transpose and get scores
        output_t = output_sigmoid.transpose(0, 2, 1)[0]
        class_scores = output_t[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        
        print(f"Max confidence range: {max_scores.min():.3f} - {max_scores.max():.3f}")
        print(f"Detections > 0.25: {(max_scores > 0.25).sum()}")
        
        # Get top 10 detections
        top_indices = np.argsort(max_scores)[-10:][::-1]
        print(f"\nTop 10 confidences:")
        for idx in top_indices:
            print(f"  {max_scores[idx]:.4f}")

print(f"\n=== Comparison ===")
print("If zero input has detections > 0 AND same as real image:")
print("  → Model is BROKEN")
print("If zero input has 0 detections AND real image has many:")
print("  → Model is WORKING")
