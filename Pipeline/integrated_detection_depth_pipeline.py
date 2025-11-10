#!/usr/bin/env python3
"""
Integrated YOLOv11 + Depth Anything V2 Pipeline
================================================

This pipeline combines object detection with depth estimation to:
1. Detect objects using YOLOv11
2. Estimate depth using Depth Anything V2
3. Calculate distances between detected objects
4. Provide comprehensive spatial analysis

Author: Pipeline Integration System
Date: November 2025
"""

import sys
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'YOLO'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Depth-Anything-V2'))

from tflite_yolo import OptimizedYOLOv11TFLite
from depth_anything_v2.dpt import DepthAnythingV2


class IntegratedPipeline:
    """
    Integrated pipeline combining YOLO object detection with depth estimation.
    
    ARCHITECTURE:
    -------------
    Input Image → YOLO Detection → Depth Estimation → Distance Calculation → Output
    
    PROCESS FLOW:
    -------------
    1. YOLO detects objects and returns bounding boxes
    2. Depth Anything V2 estimates depth map for entire image
    3. For each detected object, extract average depth from bbox region
    4. Calculate pairwise distances between all detected objects
    5. Generate comprehensive visualization and report
    """
    
    def __init__(self, 
                 yolo_model_path: str,
                 yolo_classes_path: str,
                 depth_encoder: str = 'vitb',
                 depth_weights_path: str = None,
                 yolo_conf_threshold: float = 0.60,
                 yolo_nms_threshold: float = 0.45):
        """
        Initialize the integrated pipeline.
        
        Args:
            yolo_model_path: Path to YOLOv11 TFLite model
            yolo_classes_path: Path to YOLO classes.txt
            depth_encoder: Depth model encoder type ('vits', 'vitb', 'vitl', 'vitg')
            depth_weights_path: Path to depth model weights (auto-detected if None)
            yolo_conf_threshold: YOLO confidence threshold
            yolo_nms_threshold: YOLO NMS threshold
        """
        print("=" * 80)
        print("INITIALIZING INTEGRATED DETECTION + DEPTH PIPELINE")
        print("=" * 80)
        
        # Initialize YOLO detector
        print("\n[1/2] Loading YOLOv11 Object Detector...")
        self.yolo = OptimizedYOLOv11TFLite(
            model_path=yolo_model_path,
            classes_path=yolo_classes_path,
            conf_threshold=yolo_conf_threshold,
            nms_threshold=yolo_nms_threshold
        )
        print("✓ YOLOv11 loaded successfully")
        
        # Initialize Depth Anything V2
        print(f"\n[2/2] Loading Depth Anything V2 ({depth_encoder} encoder)...")
        self.depth_encoder = depth_encoder
        
        # Model configurations
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # Auto-detect weights path if not provided
        if depth_weights_path is None:
            depth_weights_path = os.path.join(
                os.path.dirname(__file__), 
                '..', 
                'Depth-Anything-V2', 
                'checkpoints', 
                f'depth_anything_v2_{depth_encoder}.pth'
            )
        
        # Check CUDA availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load depth model
        self.depth_model = DepthAnythingV2(**model_configs[depth_encoder])
        self.depth_model.load_state_dict(torch.load(depth_weights_path, map_location='cpu'))
        self.depth_model = self.depth_model.to(self.device).eval()
        print("✓ Depth Anything V2 loaded successfully")
        
        print("\n" + "=" * 80)
        print("PIPELINE READY")
        print("=" * 80 + "\n")
    
    def process_image(self, image_path: str, output_dir: str = None) -> Dict:
        """
        Process a single image through the entire pipeline.
        
        PIPELINE STAGES:
        ----------------
        Stage 1: Object Detection (YOLO)
            - Detects objects in the image
            - Returns bounding boxes, classes, and confidence scores
        
        Stage 2: Depth Estimation (Depth Anything V2)
            - Generates depth map for entire image
            - Normalized depth values (0-255 range)
        
        Stage 3: Depth Extraction
            - For each detected object, extract depth from bbox region
            - Calculate average depth within each bounding box
        
        Stage 4: Distance Calculation
            - Compute pairwise distances between all objects
            - Based on depth values and pixel positions
        
        Stage 5: Visualization & Output
            - Generate annotated images
            - Create distance matrix
            - Save comprehensive report
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs (default: same as input)
        
        Returns:
            results: Dictionary containing all detection and depth data
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING: {os.path.basename(image_path)}")
        print(f"{'='*80}\n")
        
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Stage 1: YOLO Object Detection
        print("[Stage 1/5] Running YOLO Object Detection...")
        yolo_result, boxes, confidences, class_ids = self.yolo.predict(image_path)
        print(f"✓ Detected {len(boxes)} objects")
        
        # Stage 2: Depth Estimation
        print("\n[Stage 2/5] Running Depth Estimation...")
        original_image = cv2.imread(image_path)
        depth_map_raw = self.depth_model.infer_image(original_image, 518)
        
        # Apply logarithmic scaling for better depth perception
        # This compresses large depth values and expands small depth values
        # making near-far distinctions more visually apparent
        epsilon = 1e-6  # Small value to avoid log(0)
        depth_log = np.log1p(depth_map_raw - depth_map_raw.min() + epsilon)  # log1p(x) = log(1+x)
        
        # Normalize logarithmic depth map to 0-255 range for visualization
        depth_normalized = (depth_log - depth_log.min()) / (depth_log.max() - depth_log.min())
        depth_map = (depth_normalized * 255.0).astype(np.uint8)
        
        # Store raw logarithmic depth for accurate distance calculations
        self.depth_log_raw = depth_log
        
        print(f"✓ Depth map generated with logarithmic scaling (range: {depth_map.min()}-{depth_map.max()})")
        
        # Stage 3: Extract Depth for Each Detection
        print("\n[Stage 3/5] Extracting Depth Values for Detected Objects...")
        detections = []
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            class_name = self.yolo.class_names[class_id]
            
            # Extract depth values within bounding box (using normalized 0-255 for display)
            bbox_depth = depth_map[y1:y2, x1:x2]
            
            # Extract logarithmic depth values for accurate calculations
            bbox_depth_log = self.depth_log_raw[y1:y2, x1:x2]
            
            # Calculate statistics on logarithmic depth (more meaningful for distance)
            avg_depth_log = np.mean(bbox_depth_log)
            median_depth_log = np.median(bbox_depth_log)
            
            # Calculate statistics on normalized depth (for visualization)
            avg_depth = np.mean(bbox_depth)
            median_depth = np.median(bbox_depth)
            min_depth = np.min(bbox_depth)
            max_depth = np.max(bbox_depth)
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_depth = depth_map[center_y, center_x]
            center_depth_log = self.depth_log_raw[center_y, center_x]
            
            detection = {
                'id': i,
                'class': class_name,
                'class_id': class_id,
                'confidence': conf,
                'bbox': box,
                'center': (center_x, center_y),
                'depth_avg': avg_depth,
                'depth_median': median_depth,
                'depth_min': min_depth,
                'depth_max': max_depth,
                'depth_center': center_depth,
                'depth_avg_log': avg_depth_log,  # Logarithmic depth for calculations
                'depth_median_log': median_depth_log,
                'depth_center_log': center_depth_log,
                'area': (x2 - x1) * (y2 - y1)
            }
            
            detections.append(detection)
            print(f"  Object {i+1}: {class_name} - Avg Depth: {avg_depth:.1f} (log: {avg_depth_log:.3f}), Center: ({center_x}, {center_y})")
        
        # Stage 4: Calculate Pairwise Distances
        print("\n[Stage 4/5] Calculating Distances Between Objects...")
        distances = self._calculate_distances(detections)
        
        if distances:
            print(f"✓ Calculated {len(distances)} pairwise distances")
            for dist in distances[:5]:  # Show first 5
                print(f"  {dist['obj1_class']} ↔ {dist['obj2_class']}: "
                      f"Euclidean: {dist['euclidean']:.1f}px, "
                      f"Depth diff: {dist['depth_diff']:.1f} (log: {dist['depth_diff_log']:.3f})")
        
        # Stage 5: Generate Visualizations
        print("\n[Stage 5/5] Generating Visualizations...")
        
        # Create colored depth map
        cmap = plt.get_cmap('Spectral_r')
        depth_colored = (cmap(depth_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Save individual outputs
        yolo_output_path = os.path.join(output_dir, f"{base_name}_yolo_detections.jpg")
        depth_output_path = os.path.join(output_dir, f"{base_name}_depth_map.png")
        cv2.imwrite(yolo_output_path, yolo_result)
        cv2.imwrite(depth_output_path, depth_colored)
        
        # Create combined visualization
        combined_viz = self._create_combined_visualization(
            original_image, yolo_result, depth_colored, detections, distances
        )
        combined_output_path = os.path.join(output_dir, f"{base_name}_combined_analysis.jpg")
        cv2.imwrite(combined_output_path, combined_viz)
        
        # Create distance matrix visualization
        if len(detections) > 1:
            distance_matrix_path = os.path.join(output_dir, f"{base_name}_distance_matrix.png")
            self._create_distance_matrix(detections, distances, distance_matrix_path)
        
        # Save JSON report
        report = {
            'image': image_path,
            'total_detections': len(detections),
            'detections': detections,
            'distances': distances,
            'outputs': {
                'yolo': yolo_output_path,
                'depth': depth_output_path,
                'combined': combined_output_path
            }
        }
        
        report_path = os.path.join(output_dir, f"{base_name}_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)
        
        print(f"\n✓ All outputs saved to: {output_dir}")
        print(f"  - YOLO detections: {os.path.basename(yolo_output_path)}")
        print(f"  - Depth map: {os.path.basename(depth_output_path)}")
        print(f"  - Combined analysis: {os.path.basename(combined_output_path)}")
        print(f"  - Analysis report: {os.path.basename(report_path)}")
        
        return report
    
    def _calculate_distances(self, detections: List[Dict]) -> List[Dict]:
        """
        Calculate pairwise distances between all detected objects.
        
        DISTANCE METRICS:
        -----------------
        1. Euclidean Distance (pixels): sqrt((x2-x1)² + (y2-y1)²)
           - Measures pixel-space separation between object centers
        
        2. Depth Difference (Logarithmic): |log_depth1 - log_depth2|
           - Measures difference in logarithmic depth values
           - More perceptually meaningful than linear depth
        
        3. Horizontal/Vertical Separation:
           - X distance and Y distance components
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            List of distance measurements between object pairs
        """
        distances = []
        
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                obj1 = detections[i]
                obj2 = detections[j]
                
                # Calculate Euclidean distance between centers
                dx = obj2['center'][0] - obj1['center'][0]
                dy = obj2['center'][1] - obj1['center'][1]
                euclidean = np.sqrt(dx**2 + dy**2)
                
                # Calculate depth difference using logarithmic values (more accurate)
                depth_diff_log = abs(obj1['depth_avg_log'] - obj2['depth_avg_log'])
                
                # Also keep normalized depth difference for reference
                depth_diff = abs(obj1['depth_avg'] - obj2['depth_avg'])
                
                distance = {
                    'obj1_id': obj1['id'],
                    'obj1_class': obj1['class'],
                    'obj2_id': obj2['id'],
                    'obj2_class': obj2['class'],
                    'euclidean': float(euclidean),
                    'horizontal': abs(dx),
                    'vertical': abs(dy),
                    'depth_diff': float(depth_diff),  # Normalized (0-255)
                    'depth_diff_log': float(depth_diff_log),  # Logarithmic (more accurate)
                    'obj1_depth': float(obj1['depth_avg']),
                    'obj2_depth': float(obj2['depth_avg']),
                    'obj1_depth_log': float(obj1['depth_avg_log']),
                    'obj2_depth_log': float(obj2['depth_avg_log'])
                }
                
                distances.append(distance)
        
        # Sort by euclidean distance
        distances.sort(key=lambda x: x['euclidean'])
        
        return distances
    
    def _create_combined_visualization(self, original: np.ndarray, yolo_result: np.ndarray,
                                      depth_colored: np.ndarray, detections: List[Dict],
                                      distances: List[Dict]) -> np.ndarray:
        """
        Create a comprehensive visualization combining all pipeline outputs.
        
        LAYOUT:
        -------
        +------------------+------------------+
        |  YOLO Detection  |   Depth Map      |
        +------------------+------------------+
        |  Statistics Panel (text overlay)    |
        +-------------------------------------+
        
        Args:
            original: Original input image
            yolo_result: YOLO detection visualization
            depth_colored: Colored depth map
            detections: List of detections
            distances: List of distance measurements
        
        Returns:
            Combined visualization image
        """
        # Resize images to same height
        h = max(yolo_result.shape[0], depth_colored.shape[0])
        
        yolo_resized = cv2.resize(yolo_result, (int(yolo_result.shape[1] * h / yolo_result.shape[0]), h))
        depth_resized = cv2.resize(depth_colored, (int(depth_colored.shape[1] * h / depth_colored.shape[0]), h))
        
        # Combine side by side
        separator = np.ones((h, 20, 3), dtype=np.uint8) * 255
        combined = cv2.hconcat([yolo_resized, separator, depth_resized])
        
        # Add title bar
        title_bar = np.ones((80, combined.shape[1], 3), dtype=np.uint8) * 50
        cv2.putText(title_bar, "INTEGRATED DETECTION + DEPTH ANALYSIS", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Add statistics panel
        stats_height = 150
        stats_panel = np.ones((stats_height, combined.shape[1], 3), dtype=np.uint8) * 240
        
        # Draw statistics
        y_offset = 30
        cv2.putText(stats_panel, f"Total Detections: {len(detections)}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        if detections:
            y_offset += 30
            class_counts = {}
            for det in detections:
                class_counts[det['class']] = class_counts.get(det['class'], 0) + 1
            
            summary = ", ".join([f"{k}: {v}" for k, v in class_counts.items()])
            cv2.putText(stats_panel, f"Classes: {summary}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        if distances:
            y_offset += 35
            closest = distances[0]
            cv2.putText(stats_panel, 
                       f"Closest Pair: {closest['obj1_class']} ↔ {closest['obj2_class']} ({closest['euclidean']:.1f}px, log_depth_diff: {closest['depth_diff_log']:.3f})",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)
            
            y_offset += 30
            cv2.putText(stats_panel, f"Total Pairwise Distances: {len(distances)}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Stack all components
        final = cv2.vconcat([title_bar, combined, stats_panel])
        
        return final
    
    def _create_distance_matrix(self, detections: List[Dict], distances: List[Dict], output_path: str):
        """
        Create a visual distance matrix showing all pairwise distances.
        
        VISUALIZATION:
        --------------
        - Heatmap showing distances between all detected objects
        - Color-coded: Blue (close) → Red (far)
        - Includes both Euclidean and depth differences
        
        Args:
            detections: List of detections
            distances: List of distance measurements
            output_path: Path to save the matrix image
        """
        n = len(detections)
        
        # Create distance matrices
        euclidean_matrix = np.zeros((n, n))
        depth_matrix = np.zeros((n, n))
        
        for dist in distances:
            i, j = dist['obj1_id'], dist['obj2_id']
            euclidean_matrix[i, j] = dist['euclidean']
            euclidean_matrix[j, i] = dist['euclidean']
            depth_matrix[i, j] = dist['depth_diff']
            depth_matrix[j, i] = dist['depth_diff']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Euclidean distance heatmap
        im1 = ax1.imshow(euclidean_matrix, cmap='RdYlGn_r', aspect='auto')
        ax1.set_title('Euclidean Distance (pixels)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Object ID')
        ax1.set_ylabel('Object ID')
        
        # Add labels
        labels = [f"{d['id']}: {d['class']}" for d in detections]
        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_yticklabels(labels)
        
        # Add colorbar
        plt.colorbar(im1, ax=ax1, label='Distance (pixels)')
        
        # Depth difference heatmap
        im2 = ax2.imshow(depth_matrix, cmap='viridis', aspect='auto')
        ax2.set_title('Depth Difference', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Object ID')
        ax2.set_ylabel('Object ID')
        
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_yticklabels(labels)
        
        plt.colorbar(im2, ax=ax2, label='Depth Difference')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  - Distance matrix: {os.path.basename(output_path)}")
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)


def main():
    """
    Main execution function demonstrating pipeline usage.
    
    USAGE EXAMPLE:
    --------------
    python integrated_detection_depth_pipeline.py
    
    REQUIREMENTS:
    -------------
    1. YOLOv11 TFLite model trained on your dataset
    2. Depth Anything V2 weights (vitb or vits recommended)
    3. Input image for processing
    4. CUDA-enabled GPU (optional but recommended)
    """
    print("\n" + "="*80)
    print("INTEGRATED DETECTION + DEPTH PIPELINE - DEMO")
    print("="*80 + "\n")
    
    # Configuration
    YOLO_MODEL = os.path.join(os.path.dirname(__file__), '..', 'YOLO', 
                              'runs', 'detect', 'train', 'weights', 'best_saved_model', 'best_float32.tflite')
    YOLO_CLASSES = os.path.join(os.path.dirname(__file__), '..', 'YOLO', 'classes.txt')
    
    # Test image
    TEST_IMAGE = '/home/harshyy/Desktop/20250103_104457.jpg'
    OUTPUT_DIR = '/home/harshyy/Desktop/pipeline_output'
    
    # Check if files exist
    if not os.path.exists(YOLO_MODEL):
        print(f"❌ YOLO model not found: {YOLO_MODEL}")
        print("Please train your YOLO model first or update the path.")
        return
    
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Test image not found: {TEST_IMAGE}")
        print("Please update TEST_IMAGE path in the script.")
        return
    
    # Initialize pipeline
    pipeline = IntegratedPipeline(
        yolo_model_path=YOLO_MODEL,
        yolo_classes_path=YOLO_CLASSES,
        depth_encoder='vitb',  # Use 'vits' for faster inference
        yolo_conf_threshold=0.60,
        yolo_nms_threshold=0.45
    )
    
    # Process image
    results = pipeline.process_image(TEST_IMAGE, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*80)
    print(f"\nTotal objects detected: {results['total_detections']}")
    
    if results['detections']:
        print("\nDetected Objects:")
        for det in results['detections']:
            print(f"  {det['id']+1}. {det['class']} (confidence: {det['confidence']:.2f}, "
                  f"avg depth: {det['depth_avg']:.1f})")
    
    if results['distances']:
        print(f"\nClosest objects:")
        for i, dist in enumerate(results['distances'][:3], 1):
            print(f"  {i}. {dist['obj1_class']} ↔ {dist['obj2_class']}: "
                  f"{dist['euclidean']:.1f}px apart, "
                  f"depth diff: {dist['depth_diff']:.1f} (log: {dist['depth_diff_log']:.3f})")
    
    print(f"\n✅ All results saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
