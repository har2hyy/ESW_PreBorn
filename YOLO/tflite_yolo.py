import tensorflow as tf
import numpy as np
import cv2
import time
import os
from typing import List, Tuple, Optional

class OptimizedYOLOv11TFLite:
    """Optimized YOLOv11 TensorFlow Lite inference with proper bounding box implementation"""
    
    def __init__(self, model_path: str, classes_path: str, 
                 conf_threshold: float = 0.60, nms_threshold: float = 0.45):
        """
        Initialize the YOLOv11 TFLite model
        
        Args:
            model_path: Path to the TFLite model
            classes_path: Path to the classes.txt file
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold for overlapping box removal
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Load class names
        self.class_names = self._load_classes(classes_path)
        self.num_classes = len(self.class_names)
        print(f"Loaded {self.num_classes} classes: {self.class_names}")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get model input shape
        self.input_shape = self.input_details[0]['shape']
        self.model_height, self.model_width = self.input_shape[1], self.input_shape[2]
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_shape}")
        print(f"Model input size: {self.model_width}x{self.model_height}")
        print(f"Number of outputs: {len(self.output_details)}")
        for i, output in enumerate(self.output_details):
            print(f"  Output {i}: {output['shape']}")
    
    def _load_classes(self, classes_path: str) -> List[str]:
        """Load class names from file"""
        if not os.path.exists(classes_path):
            print(f"Warning: Classes file not found at {classes_path}")
            return [f"class_{i}" for i in range(80)]  # Default COCO classes count
        
        with open(classes_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Preprocess image for YOLOv11 inference
        
        Returns:
            input_tensor: Preprocessed image ready for inference
            original_image: Original image for drawing results
            x_scale: X scaling factor for coordinate conversion
            y_scale: Y scaling factor for coordinate conversion
        """
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Get original dimensions
        orig_height, orig_width = original_image.shape[:2]
        
        # Calculate scaling factors
        x_scale = orig_width / self.model_width
        y_scale = orig_height / self.model_height
        
        # Resize image to model input size
        image_resized = cv2.resize(original_image, (self.model_width, self.model_height))
        
        # Convert BGR to RGB and normalize to [0, 1]
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_tensor = np.expand_dims(image_normalized, axis=0)
        
        return input_tensor, original_image, x_scale, y_scale
    
    def postprocess_detections(self, outputs: List[np.ndarray], x_scale: float, y_scale: float) -> Tuple[List, List, List]:
        """
        Process YOLOv11 outputs to extract bounding boxes
        
        Args:
            outputs: Raw model outputs
            x_scale: X scaling factor
            y_scale: Y scaling factor
            
        Returns:
            boxes: List of bounding boxes [x1, y1, x2, y2]
            confidences: List of confidence scores
            class_ids: List of class IDs
        """
        boxes = []
        confidences = []
        class_ids = []
        
        # YOLOv11 output format: [batch, channels, anchors]
        # where channels = 4 (bbox) + num_classes
        for output in outputs:
            # Output shape should be [1, 9, 21504] for our model
            if len(output.shape) != 3:
                continue
                
            batch_size, channels, num_anchors = output.shape
            
            if channels != (4 + self.num_classes):
                print(f"Warning: Expected {4 + self.num_classes} channels, got {channels}")
                continue
            
            # Transpose to [batch, anchors, channels] format
            predictions = output.transpose(0, 2, 1)  # [1, 21504, 9]
            predictions = predictions[0]  # Remove batch dimension: [21504, 9]
            
            # Split into bbox and class predictions
            bbox_predictions = predictions[:, :4]  # [21504, 4] - x, y, w, h
            class_predictions = predictions[:, 4:]  # [21504, 5] - class logits
            
            # Apply sigmoid to class predictions to get probabilities
            class_probs = 1 / (1 + np.exp(-class_predictions))  # sigmoid activation
            
            # Get the best class and its confidence for each prediction
            max_class_probs = np.max(class_probs, axis=1)
            best_class_ids = np.argmax(class_probs, axis=1)
            
            # Generate anchor points for all scales
            strides = [8, 16, 32]
            anchor_points = []
            
            for stride in strides:
                grid_h = self.model_height // stride
                grid_w = self.model_width // stride
                
                # Create grid coordinates
                for y in range(grid_h):
                    for x in range(grid_w):
                        # Convert grid coordinates to pixel coordinates
                        anchor_x = (x + 0.5) * stride
                        anchor_y = (y + 0.5) * stride
                        anchor_points.append([anchor_x, anchor_y, stride])
            
            anchor_points = np.array(anchor_points)  # [21504, 3] - x, y, stride
            
            # Process each prediction
            for i in range(num_anchors):
                if max_class_probs[i] > self.conf_threshold:
                    # Get bbox prediction (already in normalized format)
                    bbox = bbox_predictions[i]  # [x, y, w, h]
                    
                    # Convert to absolute coordinates
                    center_x = bbox[0] * self.model_width
                    center_y = bbox[1] * self.model_height
                    width = bbox[2] * self.model_width
                    height = bbox[3] * self.model_height
                    
                    # Convert to corner coordinates and scale to original image
                    x1 = (center_x - width / 2) * x_scale
                    y1 = (center_y - height / 2) * y_scale
                    x2 = (center_x + width / 2) * x_scale
                    y2 = (center_y + height / 2) * y_scale
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = int(x2)
                    y2 = int(y2)
                    
                    # Only add if the box has reasonable dimensions
                    if x2 > x1 and y2 > y1 and (x2 - x1) > 5 and (y2 - y1) > 5:
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(float(max_class_probs[i]))
                        class_ids.append(int(best_class_ids[i]))
        
        return boxes, confidences, class_ids
    
    def apply_nms(self, boxes: List, confidences: List, class_ids: List) -> Tuple[List, List, List]:
        """Apply Non-Maximum Suppression to remove overlapping boxes"""
        if not boxes:
            return [], [], []
        
        # Convert to format expected by cv2.dnn.NMSBoxes [x, y, width, height]
        boxes_xywh = []
        for box in boxes:
            x1, y1, x2, y2 = box
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, confidences, self.conf_threshold, self.nms_threshold
        )
        
        if len(indices) == 0:
            return [], [], []
        
        # Extract surviving boxes
        final_boxes = []
        final_confidences = []
        final_class_ids = []
        
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_confidences.append(confidences[i])
            final_class_ids.append(class_ids[i])
        
        return final_boxes, final_confidences, final_class_ids
    
    def draw_detections(self, image: np.ndarray, boxes: List, confidences: List, 
                       class_ids: List, thickness: int = 2) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (128, 128, 128), (0, 128, 0), (128, 0, 0)
        ]
        
        for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with background
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"Class_{class_id}"
            
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                image, 
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            print(f"Detection {i+1}: {class_name} ({confidence:.3f}) at [{x1}, {y1}, {x2}, {y2}]")
        
        return image
    
    def predict(self, image_path: str, save_path: Optional[str] = None) -> Tuple[np.ndarray, List, List, List]:
        """
        Run inference on an image
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save result image
            
        Returns:
            result_image: Image with drawn bounding boxes
            boxes: List of bounding boxes
            confidences: List of confidence scores
            class_ids: List of class IDs
        """
        start_time = time.time()
        
        # Preprocess image
        input_tensor, original_image, x_scale, y_scale = self.preprocess_image(image_path)
        preprocess_time = time.time() - start_time
        
        # Run inference
        inference_start = time.time()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # Get outputs
        outputs = []
        for output_detail in self.output_details:
            output_data = self.interpreter.get_tensor(output_detail['index'])
            outputs.append(output_data)
        
        inference_time = time.time() - inference_start
        
        # Postprocess detections
        postprocess_start = time.time()
        boxes, confidences, class_ids = self.postprocess_detections(outputs, x_scale, y_scale)
        
        # Apply NMS
        boxes, confidences, class_ids = self.apply_nms(boxes, confidences, class_ids)
        postprocess_time = time.time() - postprocess_start
        
        # Draw results
        result_image = original_image.copy()
        if boxes:
            result_image = self.draw_detections(result_image, boxes, confidences, class_ids)
        
        total_time = time.time() - start_time
        
        # Print timing information
        print(f"\n=== Inference Results ===")
        print(f"Found {len(boxes)} detections")
        print(f"Preprocessing: {preprocess_time*1000:.1f}ms")
        print(f"Inference: {inference_time*1000:.1f}ms")
        print(f"Postprocessing: {postprocess_time*1000:.1f}ms")
        print(f"Total time: {total_time*1000:.1f}ms")
        
        # Save result if requested
        if save_path:
            cv2.imwrite(save_path, result_image)
            print(f"Result saved to: {save_path}")
        
        return result_image, boxes, confidences, class_ids

# Main execution
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = 'runs/detect/train/weights/best_saved_model/best_float32.tflite'
    CLASSES_PATH = '../data+label/classes.txt'
    IMAGE_PATH = '../better_daytime/images/20241203_144957.jpg'  # Updated path
    RESULT_PATH = 'result_optimized_tflite.jpg'
    
    # Initialize detector
    print("Initializing YOLOv11 TFLite detector...")
    detector = OptimizedYOLOv11TFLite(
        model_path=MODEL_PATH,
        classes_path=CLASSES_PATH,
        conf_threshold=0.60,  # Optimized for worker detection
        nms_threshold=0.45
    )
    
    # Run inference on test images
    test_images = [
        IMAGE_PATH,
        '../better_daytime/images/20241202_131017.jpg',
        '../better_daytime/images/20241203_102939.jpg'
    ]
    
    for i, img_path in enumerate(test_images):
        if os.path.exists(img_path):
            print(f"\n--- Processing image {i+1}: {os.path.basename(img_path)} ---")
            result_path = f'result_optimized_{i+1}.jpg'
            
            try:
                result_image, boxes, confidences, class_ids = detector.predict(img_path, result_path)
                print(f"✅ Successfully processed {img_path}")
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
        else:
            print(f"⚠️  Image not found: {img_path}")
    
    print(f"\n🎉 Processing complete! Check the result images.")