#!/usr/bin/env python3
"""
Integrated YOLOv11 + Depth Anything V2 Pipeline (PyTorch Version)
==================================================================

This pipeline uses the YOLOv11 PyTorch model (best.pt) instead of TFLite.
Much simpler and more reliable!

Usage:
    python integrated_pipeline_pytorch.py

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

# Add Depth-Anything-V2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Depth-Anything-V2'))

from depth_anything_v2.dpt import DepthAnythingV2

# Try to import ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO


class IntegratedPipelinePyTorch:
    """
    Integrated pipeline using PyTorch YOLO model + Depth Anything V2.
    
    This version uses the native YOLOv11 PyTorch model (best.pt) which is:
    - Simpler to use (no TFLite conversion issues)
    - More accurate (native PyTorch inference)
    - Better supported (Ultralytics API)
    - Easier to debug
    """
    
    def __init__(self, 
                 yolo_model_path: str,
                 depth_encoder: str = 'vitb',
                 depth_weights_path: str = None,
                 yolo_conf_threshold: float = 0.51,
                 yolo_iou_threshold: float = 0.45,
                 depth_scale_factor: float = 3.0):
        """
        Initialize the integrated pipeline with PyTorch YOLO.
        
        Args:
            yolo_model_path: Path to YOLOv11 PyTorch model (best.pt)
            depth_encoder: Depth model encoder type ('vits', 'vitb', 'vitl', 'vitg')
            depth_weights_path: Path to depth model weights (auto-detected if None)
            yolo_conf_threshold: YOLO confidence threshold
            yolo_iou_threshold: YOLO IoU threshold for NMS
            depth_scale_factor: Scaling factor for depth to match real-world proportions
                              - Default 3.0 means depth is weighted 3x in distance calculations
                              - Increase for scenes with more depth variation
                              - Decrease for mostly planar scenes
        """
        print("=" * 80)
        print("INITIALIZING INTEGRATED PIPELINE (PyTorch Version)")
        print("=" * 80)
        
        # Store depth scale factor
        self.depth_scale_factor = depth_scale_factor
        
        # Initialize YOLO model
        print("\n[1/2] Loading YOLOv11 PyTorch Model...")
        self.yolo = YOLO(yolo_model_path)
        self.conf_threshold = yolo_conf_threshold
        self.iou_threshold = yolo_iou_threshold
        
        # Get class names
        self.class_names = self.yolo.names  # Dictionary {0: 'worker', 1: 'truck', ...}
        print(f"✓ YOLOv11 loaded with {len(self.class_names)} classes: {list(self.class_names.values())}")
        
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
        print(f"Depth Scale Factor: {self.depth_scale_factor}x")
        print("=" * 80 + "\n")
    
    def process_image(self, image_path: str, output_dir: str = None) -> Dict:
        """
        Process a single image through the entire pipeline.
        
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
        results = self.yolo.predict(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Extract detections
        result = results[0]  # First (and only) image
        boxes = result.boxes
        
        detections_list = []
        print(f"✓ Detected {len(boxes)} objects")
        
        for i, box in enumerate(boxes):
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            
            print(f"  {i+1}. {class_name} (confidence: {conf:.3f}) at [{x1}, {y1}, {x2}, {y2}]")
            
            detections_list.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class_id': class_id,
                'class': class_name
            })
        
        # Stage 2: Depth Estimation
        print("\n[Stage 2/5] Running Depth Estimation...")
        original_image = cv2.imread(image_path)
        depth_map_raw = self.depth_model.infer_image(original_image, 518)
        
        # Normalize depth map to 0-255 range
        depth_normalized = (depth_map_raw - depth_map_raw.min()) / (depth_map_raw.max() - depth_map_raw.min())
        depth_map = (depth_normalized * 255.0).astype(np.uint8)
        print(f"✓ Depth map generated (range: {depth_map.min()}-{depth_map.max()})")
        
        # Stage 3: Extract Depth for Each Detection
        print("\n[Stage 3/5] Extracting Depth Values for Detected Objects...")
        detections = []
        
        for i, det in enumerate(detections_list):
            x1, y1, x2, y2 = det['bbox']
            
            # Extract depth values within bounding box
            bbox_depth = depth_map[y1:y2, x1:x2]
            
            if bbox_depth.size == 0:
                continue
            
            # Calculate statistics
            avg_depth = np.mean(bbox_depth)
            median_depth = np.median(bbox_depth)
            min_depth = np.min(bbox_depth)
            max_depth = np.max(bbox_depth)
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_depth = depth_map[center_y, center_x]
            
            detection = {
                'id': i,
                'class': det['class'],
                'class_id': det['class_id'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'center': (center_x, center_y),
                'depth_avg': float(avg_depth),
                'depth_median': float(median_depth),
                'depth_min': float(min_depth),
                'depth_max': float(max_depth),
                'depth_center': float(center_depth),
                'area': (x2 - x1) * (y2 - y1)
            }
            
            detections.append(detection)
            print(f"  Object {i+1}: {det['class']} - Avg Depth: {avg_depth:.1f}, Center: ({center_x}, {center_y})")
        
        # Stage 4: Calculate Pairwise Distances
        print("\n[Stage 4/5] Calculating Distances Between Objects...")
        distances = self._calculate_distances(detections)
        
        if distances:
            print(f"✓ Calculated {len(distances)} pairwise distances")
            for dist in distances[:5]:  # Show first 5
                print(f"  {dist['obj1_class']} ↔ {dist['obj2_class']}: "
                      f"Euclidean: {dist['euclidean']:.1f}px, "
                      f"Depth diff: {dist['depth_diff']:.1f} (scaled: {dist['depth_diff_scaled']:.1f})")
        
        # Stage 5: Generate Visualizations
        print("\n[Stage 5/5] Generating Visualizations...")
        
        # Create colored depth map
        cmap = plt.get_cmap('Spectral_r')
        depth_colored = (cmap(depth_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Save YOLO detections
        yolo_output_path = os.path.join(output_dir, f"{base_name}_yolo_detections.jpg")
        annotated_frame = result.plot()  # Get annotated image from YOLO
        cv2.imwrite(yolo_output_path, annotated_frame)
        
        # Save depth map
        depth_output_path = os.path.join(output_dir, f"{base_name}_depth_map.png")
        cv2.imwrite(depth_output_path, depth_colored)
        
        # Create combined visualization
        combined_viz = self._create_combined_visualization(
            original_image, annotated_frame, depth_colored, detections, distances
        )
        combined_output_path = os.path.join(output_dir, f"{base_name}_combined_analysis.jpg")
        cv2.imwrite(combined_output_path, combined_viz)
        
        # Create distance matrix visualization
        if len(detections) > 1:
            distance_matrix_path = os.path.join(output_dir, f"{base_name}_distance_matrix.png")
            distance_table_path = os.path.join(output_dir, f"{base_name}_distance_table.txt")
            comparison_path = os.path.join(output_dir, f"{base_name}_depth_comparison.txt")
            comparison_chart_path = os.path.join(output_dir, f"{base_name}_depth_comparison_charts.png")
            self._create_distance_matrix(detections, distances, distance_matrix_path)
            self._save_distance_table(detections, distances, distance_table_path)
            self._save_depth_comparison(detections, distances, comparison_path)
            self._create_depth_comparison_charts(detections, distances, comparison_chart_path)
        
        # Create worker density heatmap
        if detections:
            worker_heatmap_path = os.path.join(output_dir, f"{base_name}_worker_heatmap.png")
            self._create_worker_heatmap(original_image, detections, worker_heatmap_path)
        
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
        if len(detections) > 1:
            print(f"  - Distance table: {base_name}_distance_table.txt")
            print(f"  - Depth comparison (text): {base_name}_depth_comparison.txt")
            print(f"  - Depth comparison (charts): {base_name}_depth_comparison_charts.png")
        if detections:
            print(f"  - Worker heatmap: {base_name}_worker_heatmap.png")
        print(f"  - Analysis report: {os.path.basename(report_path)}")
        
        return report
    
    def _calculate_distances(self, detections: List[Dict]) -> List[Dict]:
        """Calculate pairwise distances between all detected objects."""
        distances = []
        
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                obj1 = detections[i]
                obj2 = detections[j]
                
                # Calculate Euclidean distance between centers
                dx = obj2['center'][0] - obj1['center'][0]
                dy = obj2['center'][1] - obj1['center'][1]
                euclidean = np.sqrt(dx**2 + dy**2)
                
                # Calculate depth difference (raw, unscaled)
                depth_diff_raw = abs(obj1['depth_avg'] - obj2['depth_avg'])
                
                # Calculate SCALED depth difference for realistic 2.5D distance
                depth_diff_scaled = depth_diff_raw * self.depth_scale_factor
                
                distance = {
                    'obj1_id': obj1['id'],
                    'obj1_class': obj1['class'],
                    'obj2_id': obj2['id'],
                    'obj2_class': obj2['class'],
                    'euclidean': float(euclidean),
                    'horizontal': abs(dx),
                    'vertical': abs(dy),
                    'depth_diff': float(depth_diff_raw),  # Keep raw for reference
                    'depth_diff_scaled': float(depth_diff_scaled),  # Scaled for calculations
                    'obj1_depth': float(obj1['depth_avg']),
                    'obj2_depth': float(obj2['depth_avg'])
                }
                
                distances.append(distance)
        
        # Sort by euclidean distance
        distances.sort(key=lambda x: x['euclidean'])
        
        return distances
    
    def _create_combined_visualization(self, original: np.ndarray, yolo_result: np.ndarray,
                                      depth_colored: np.ndarray, detections: List[Dict],
                                      distances: List[Dict]) -> np.ndarray:
        """Create a comprehensive visualization combining all pipeline outputs."""
        # Resize images to same height
        h = max(yolo_result.shape[0], depth_colored.shape[0])
        
        yolo_resized = cv2.resize(yolo_result, (int(yolo_result.shape[1] * h / yolo_result.shape[0]), h))
        depth_resized = cv2.resize(depth_colored, (int(depth_colored.shape[1] * h / depth_colored.shape[0]), h))
        
        # Combine side by side
        separator = np.ones((h, 20, 3), dtype=np.uint8) * 255
        combined = cv2.hconcat([yolo_resized, separator, depth_resized])
        
        # Add title bar
        title_bar = np.ones((80, combined.shape[1], 3), dtype=np.uint8) * 50
        cv2.putText(title_bar, "INTEGRATED DETECTION + DEPTH ANALYSIS (PyTorch)", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
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
                       f"Closest Pair: {closest['obj1_class']} <-> {closest['obj2_class']} ({closest['euclidean']:.1f}px)",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
            
            y_offset += 30
            cv2.putText(stats_panel, f"Total Pairwise Distances: {len(distances)}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Stack all components
        final = cv2.vconcat([title_bar, combined, stats_panel])
        
        return final
    
    def _create_distance_matrix(self, detections: List[Dict], distances: List[Dict], output_path: str):
        """Create a visual distance matrix showing all pairwise distances with values."""
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Fixed color scale ranges for consistency across runs
        # Euclidean: typically 0-2000 pixels for construction site images
        # Depth: typically 0-200 for normalized depth values
        euclidean_vmin, euclidean_vmax = 0, 2000
        depth_vmin, depth_vmax = 0, 200
        
        # Euclidean distance heatmap with fixed scale
        im1 = ax1.imshow(euclidean_matrix, cmap='RdYlGn_r', aspect='auto',
                        vmin=euclidean_vmin, vmax=euclidean_vmax)
        ax1.set_title('Euclidean Distance (pixels)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Object ID')
        ax1.set_ylabel('Object ID')
        
        # Add labels
        labels = [f"{d['id']}: {d['class']}" for d in detections]
        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.set_yticklabels(labels, fontsize=9)
        
        # Add numerical values on cells
        for i in range(n):
            for j in range(n):
                if i != j:  # Don't show 0 on diagonal
                    text_color = 'white' if euclidean_matrix[i, j] > euclidean_vmax * 0.5 else 'black'
                    ax1.text(j, i, f'{euclidean_matrix[i, j]:.0f}',
                           ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im1, ax=ax1, label='Distance (pixels)')
        
        # Depth difference heatmap with fixed scale
        im2 = ax2.imshow(depth_matrix, cmap='viridis', aspect='auto',
                        vmin=depth_vmin, vmax=depth_vmax)
        ax2.set_title('Depth Difference', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Object ID')
        ax2.set_ylabel('Object ID')
        
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.set_yticklabels(labels, fontsize=9)
        
        # Add numerical values on cells
        for i in range(n):
            for j in range(n):
                if i != j:  # Don't show 0 on diagonal
                    text_color = 'white' if depth_matrix[i, j] > depth_vmax * 0.5 else 'black'
                    ax2.text(j, i, f'{depth_matrix[i, j]:.1f}',
                           ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, label='Depth Difference')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  - Distance matrix: {os.path.basename(output_path)}")
    
    def _save_distance_table(self, detections: List[Dict], distances: List[Dict], output_path: str):
        """Save distance measurements in a human-readable text table format."""
        with open(output_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("DISTANCE TABLE - Pairwise Distances Between Detected Objects\n")
            f.write("=" * 100 + "\n\n")
            
            # Summary of detections
            f.write(f"Total Objects Detected: {len(detections)}\n\n")
            f.write("Object List:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'ID':<5} {'Class':<15} {'Confidence':<12} {'Center (x,y)':<15} {'Avg Depth':<12} {'BBox Area':<10}\n")
            f.write("-" * 100 + "\n")
            
            for det in detections:
                f.write(f"{det['id']:<5} {det['class']:<15} {det['confidence']:<12.3f} "
                       f"({det['center'][0]},{det['center'][1]}){'':<6} {det['depth_avg']:<12.1f} "
                       f"{det['area']:<10}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"Pairwise Distances (Total: {len(distances)} pairs)\n")
            f.write("=" * 100 + "\n\n")
            
            # Column headers
            f.write(f"{'#':<4} {'Object 1':<20} {'Object 2':<20} {'Euclidean':<12} "
                   f"{'Horizontal':<12} {'Vertical':<12} {'Depth Diff':<12}\n")
            f.write("-" * 100 + "\n")
            
            # Distance data
            for idx, dist in enumerate(distances, 1):
                obj1_str = f"[{dist['obj1_id']}] {dist['obj1_class']}"
                obj2_str = f"[{dist['obj2_id']}] {dist['obj2_class']}"
                
                f.write(f"{idx:<4} {obj1_str:<20} {obj2_str:<20} "
                       f"{dist['euclidean']:<12.1f} {dist['horizontal']:<12.1f} "
                       f"{dist['vertical']:<12.1f} {dist['depth_diff']:<12.1f}\n")
            
            # Statistics section
            f.write("\n" + "=" * 100 + "\n")
            f.write("STATISTICS\n")
            f.write("=" * 100 + "\n\n")
            
            if distances:
                euclidean_dists = [d['euclidean'] for d in distances]
                depth_diffs = [d['depth_diff'] for d in distances]
                
                f.write(f"Euclidean Distances:\n")
                f.write(f"  Min:     {min(euclidean_dists):.1f} px\n")
                f.write(f"  Max:     {max(euclidean_dists):.1f} px\n")
                f.write(f"  Mean:    {np.mean(euclidean_dists):.1f} px\n")
                f.write(f"  Median:  {np.median(euclidean_dists):.1f} px\n\n")
                
                f.write(f"Depth Differences:\n")
                f.write(f"  Min:     {min(depth_diffs):.1f}\n")
                f.write(f"  Max:     {max(depth_diffs):.1f}\n")
                f.write(f"  Mean:    {np.mean(depth_diffs):.1f}\n")
                f.write(f"  Median:  {np.median(depth_diffs):.1f}\n\n")
                
                # Closest and farthest pairs
                f.write(f"Closest Pair:   {distances[0]['obj1_class']} ↔ {distances[0]['obj2_class']} "
                       f"({distances[0]['euclidean']:.1f} px)\n")
                f.write(f"Farthest Pair:  {distances[-1]['obj1_class']} ↔ {distances[-1]['obj2_class']} "
                       f"({distances[-1]['euclidean']:.1f} px)\n")
            
            f.write("\n" + "=" * 100 + "\n")
        
        print(f"  - Distance table: {os.path.basename(output_path)}")
    
    def _save_depth_comparison(self, detections: List[Dict], distances: List[Dict], output_path: str):
        """
        Compare distance measurements with and without depth information.
        
        This analysis shows:
        1. 2D distances (pixel-based only, ignoring depth)
        2. 2.5D distances (incorporating depth information)
        3. Impact of depth on distance ranking
        4. Cases where depth significantly changes interpretation
        """
        with open(output_path, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("DEPTH IMPACT ANALYSIS - Comparing 2D vs 2.5D Distance Measurements\n")
            f.write("=" * 120 + "\n\n")
            
            f.write("METHODOLOGY:\n")
            f.write("-" * 120 + "\n")
            f.write("• 2D Distance (Euclidean): Based only on pixel coordinates (x, y)\n")
            f.write("  Formula: sqrt((x2-x1)² + (y2-y1)²)\n")
            f.write("  - Ignores depth information\n")
            f.write("  - Objects at different depths appear equally distant if at same pixel location\n\n")
            
            f.write("• 2.5D Distance (Depth-Aware): Incorporates SCALED depth as third dimension\n")
            f.write(f"  Formula: sqrt((x2-x1)² + (y2-y1)² + (depth_diff × {self.depth_scale_factor})²)\n")
            f.write("  - Considers depth difference as spatial dimension\n")
            f.write(f"  - Depth scaled by factor {self.depth_scale_factor}x to match real-world proportions\n")
            f.write("  - More realistic representation of 3D spatial relationships\n\n")
            
            f.write("=" * 120 + "\n\n")
            
            # Calculate 2.5D distances
            distances_2d = []
            distances_25d = []
            
            for dist in distances:
                # 2D distance (already have this)
                dist_2d = dist['euclidean']
                
                # 2.5D distance (add SCALED depth as third dimension)
                dist_25d = np.sqrt(dist['euclidean']**2 + dist['depth_diff_scaled']**2)
                
                distances_2d.append({
                    'pair': f"{dist['obj1_class']}[{dist['obj1_id']}] ↔ {dist['obj2_class']}[{dist['obj2_id']}]",
                    'obj1_id': dist['obj1_id'],
                    'obj2_id': dist['obj2_id'],
                    'distance': dist_2d,
                    'depth_diff': dist['depth_diff'],
                    'depth_diff_scaled': dist['depth_diff_scaled']
                })
                
                distances_25d.append({
                    'pair': f"{dist['obj1_class']}[{dist['obj1_id']}] ↔ {dist['obj2_class']}[{dist['obj2_id']}]",
                    'obj1_id': dist['obj1_id'],
                    'obj2_id': dist['obj2_id'],
                    'distance': dist_25d,
                    'depth_diff': dist['depth_diff'],
                    'depth_diff_scaled': dist['depth_diff_scaled']
                })
            
            # Sort both by distance
            distances_2d_sorted = sorted(distances_2d, key=lambda x: x['distance'])
            distances_25d_sorted = sorted(distances_25d, key=lambda x: x['distance'])
            
            # SECTION 1: Side-by-side comparison
            f.write("SECTION 1: SIDE-BY-SIDE COMPARISON (Top 15 Closest Pairs)\n")
            f.write("=" * 120 + "\n\n")
            
            f.write(f"{'Rank':<6} {'2D Distance (Pixel Only)':<40} {'Distance':<12} {'2.5D Distance (With Depth)':<40} {'Distance':<12}\n")
            f.write("-" * 120 + "\n")
            
            max_display = min(15, len(distances))
            for i in range(max_display):
                d2d = distances_2d_sorted[i]
                d25d = distances_25d_sorted[i]
                
                f.write(f"{i+1:<6} {d2d['pair']:<40} {d2d['distance']:<12.1f} "
                       f"{d25d['pair']:<40} {d25d['distance']:<12.1f}\n")
            
            # SECTION 2: Ranking changes
            f.write("\n" + "=" * 120 + "\n")
            f.write("SECTION 2: RANKING CHANGES - Objects that changed position when depth is considered\n")
            f.write("=" * 120 + "\n\n")
            
            # Create ranking maps
            rank_2d = {d['pair']: idx for idx, d in enumerate(distances_2d_sorted)}
            rank_25d = {d['pair']: idx for idx, d in enumerate(distances_25d_sorted)}
            
            rank_changes = []
            for pair in rank_2d.keys():
                rank_change = rank_2d[pair] - rank_25d[pair]
                if rank_change != 0:
                    rank_changes.append({
                        'pair': pair,
                        'rank_2d': rank_2d[pair] + 1,
                        'rank_25d': rank_25d[pair] + 1,
                        'change': rank_change,
                        'dist_2d': next(d['distance'] for d in distances_2d if d['pair'] == pair),
                        'dist_25d': next(d['distance'] for d in distances_25d if d['pair'] == pair),
                        'depth_diff': next(d['depth_diff'] for d in distances_2d if d['pair'] == pair)
                    })
            
            rank_changes.sort(key=lambda x: abs(x['change']), reverse=True)
            
            if rank_changes:
                f.write(f"{'Object Pair':<45} {'2D Rank':<10} {'2.5D Rank':<12} {'Change':<10} {'Depth Diff':<12}\n")
                f.write("-" * 120 + "\n")
                
                for rc in rank_changes[:20]:  # Show top 20 changes
                    direction = "↑" if rc['change'] > 0 else "↓"
                    f.write(f"{rc['pair']:<45} {rc['rank_2d']:<10} {rc['rank_25d']:<12} "
                           f"{direction}{abs(rc['change']):<9} {rc['depth_diff']:<12.1f}\n")
                
                f.write(f"\nTotal pairs with rank changes: {len(rank_changes)}/{len(distances)}\n")
            else:
                f.write("No ranking changes - depth had minimal impact on distance ordering.\n")
            
            # SECTION 3: Statistical comparison
            f.write("\n" + "=" * 120 + "\n")
            f.write("SECTION 3: STATISTICAL COMPARISON\n")
            f.write("=" * 120 + "\n\n")
            
            dists_2d_values = [d['distance'] for d in distances_2d]
            dists_25d_values = [d['distance'] for d in distances_25d]
            depth_diffs_values = [d['depth_diff'] for d in distances_2d]
            
            # Calculate percentage differences
            pct_diffs = [(d25 - d2) / d2 * 100 for d2, d25 in zip(dists_2d_values, dists_25d_values)]
            
            f.write(f"{'Metric':<30} {'2D Distance':<20} {'2.5D Distance':<20} {'Difference':<15}\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Minimum Distance':<30} {min(dists_2d_values):<20.1f} {min(dists_25d_values):<20.1f} "
                   f"{min(dists_25d_values) - min(dists_2d_values):<15.1f}\n")
            f.write(f"{'Maximum Distance':<30} {max(dists_2d_values):<20.1f} {max(dists_25d_values):<20.1f} "
                   f"{max(dists_25d_values) - max(dists_2d_values):<15.1f}\n")
            f.write(f"{'Mean Distance':<30} {np.mean(dists_2d_values):<20.1f} {np.mean(dists_25d_values):<20.1f} "
                   f"{np.mean(dists_25d_values) - np.mean(dists_2d_values):<15.1f}\n")
            f.write(f"{'Median Distance':<30} {np.median(dists_2d_values):<20.1f} {np.median(dists_25d_values):<20.1f} "
                   f"{np.median(dists_25d_values) - np.median(dists_2d_values):<15.1f}\n")
            f.write(f"{'Std Deviation':<30} {np.std(dists_2d_values):<20.1f} {np.std(dists_25d_values):<20.1f} "
                   f"{np.std(dists_25d_values) - np.std(dists_2d_values):<15.1f}\n")
            
            f.write(f"\n{'Average % Increase':<30} {'-':<20} {'-':<20} {np.mean(pct_diffs):<15.1f}%\n")
            f.write(f"{'Max % Increase':<30} {'-':<20} {'-':<20} {max(pct_diffs):<15.1f}%\n")
            f.write(f"{'Min % Increase':<30} {'-':<20} {'-':<20} {min(pct_diffs):<15.1f}%\n")
            
            # SECTION 4: Depth impact categories
            f.write("\n" + "=" * 120 + "\n")
            f.write("SECTION 4: DEPTH IMPACT CATEGORIES\n")
            f.write("=" * 120 + "\n\n")
            
            # Categorize by depth impact
            low_impact = [d for d in pct_diffs if d < 5]
            medium_impact = [d for d in pct_diffs if 5 <= d < 20]
            high_impact = [d for d in pct_diffs if d >= 20]
            
            f.write(f"Low Impact (< 5% change):      {len(low_impact):<5} pairs ({len(low_impact)/len(pct_diffs)*100:.1f}%)\n")
            f.write(f"Medium Impact (5-20% change):  {len(medium_impact):<5} pairs ({len(medium_impact)/len(pct_diffs)*100:.1f}%)\n")
            f.write(f"High Impact (> 20% change):    {len(high_impact):<5} pairs ({len(high_impact)/len(pct_diffs)*100:.1f}%)\n")
            
            # SECTION 5: Cases where depth matters most
            f.write("\n" + "=" * 120 + "\n")
            f.write("SECTION 5: HIGH DEPTH IMPACT CASES - Pairs where depth significantly changes distance\n")
            f.write("=" * 120 + "\n\n")
            
            # Find pairs with high depth impact
            high_depth_impact = []
            for i, (d2d, d25d, pct) in enumerate(zip(distances_2d, distances_25d, pct_diffs)):
                if pct > 10:  # More than 10% increase
                    high_depth_impact.append({
                        'pair': d2d['pair'],
                        'dist_2d': d2d['distance'],
                        'dist_25d': d25d['distance'],
                        'depth_diff': d2d['depth_diff'],
                        'pct_increase': pct
                    })
            
            high_depth_impact.sort(key=lambda x: x['pct_increase'], reverse=True)
            
            if high_depth_impact:
                f.write(f"{'Object Pair':<45} {'2D Dist':<12} {'2.5D Dist':<12} {'Depth Diff':<12} {'% Change':<10}\n")
                f.write("-" * 120 + "\n")
                
                for hdi in high_depth_impact[:15]:
                    f.write(f"{hdi['pair']:<45} {hdi['dist_2d']:<12.1f} {hdi['dist_25d']:<12.1f} "
                           f"{hdi['depth_diff']:<12.1f} {hdi['pct_increase']:<10.1f}%\n")
            else:
                f.write("No pairs with significant depth impact (> 10% increase).\n")
            
            # SECTION 6: Key insights
            f.write("\n" + "=" * 120 + "\n")
            f.write("SECTION 6: KEY INSIGHTS & RECOMMENDATIONS\n")
            f.write("=" * 120 + "\n\n")
            
            f.write("INTERPRETATION:\n")
            f.write("-" * 120 + "\n")
            
            avg_pct_increase = np.mean(pct_diffs)
            
            if avg_pct_increase < 5:
                f.write("• LOW DEPTH IMPACT: Objects are mostly at similar depths.\n")
                f.write("  → 2D distances are generally sufficient for this scene.\n")
                f.write("  → Scene appears mostly planar or with minimal depth variation.\n")
            elif avg_pct_increase < 15:
                f.write("• MODERATE DEPTH IMPACT: Noticeable depth variations exist.\n")
                f.write("  → Consider depth information for accurate spatial analysis.\n")
                f.write("  → Some objects significantly closer/farther than they appear in 2D.\n")
            else:
                f.write("• HIGH DEPTH IMPACT: Significant depth variations across scene.\n")
                f.write("  → Depth information is CRITICAL for accurate distance measurement.\n")
                f.write("  → 2D distances alone would be highly misleading.\n")
                f.write("  → Scene has complex 3D structure.\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 120 + "\n")
            
            if len(high_impact) > len(distances) * 0.3:
                f.write("• Use 2.5D distances for safety analysis and spatial reasoning.\n")
                f.write("• Depth significantly affects object relationships in this scene.\n")
            else:
                f.write("• 2D distances may be acceptable for general analysis.\n")
                f.write("• Use 2.5D distances when precision is critical.\n")
            
            if rank_changes:
                f.write(f"• {len(rank_changes)} pairs changed ranking - verify closest pairs with depth.\n")
            
            f.write("\n" + "=" * 120 + "\n")
        
        print(f"  - Depth comparison: {os.path.basename(output_path)}")
    
    def _create_depth_comparison_charts(self, detections: List[Dict], distances: List[Dict], output_path: str):
        """
        Create graphical visualizations comparing 2D and 2.5D distances.
        
        Generates 4 charts:
        1. Bar chart comparing 2D vs 2.5D distances
        2. Scatter plot showing correlation
        3. Histogram of percentage differences
        4. Ranking changes visualization
        """
        # Calculate 2.5D distances
        distances_2d = []
        distances_25d = []
        labels = []
        
        for dist in distances:
            dist_2d = dist['euclidean']
            # Use SCALED depth difference for realistic impact
            dist_25d = np.sqrt(dist['euclidean']**2 + dist['depth_diff_scaled']**2)
            
            distances_2d.append(dist_2d)
            distances_25d.append(dist_25d)
            labels.append(f"{dist['obj1_class'][0]}{dist['obj1_id']}-{dist['obj2_class'][0]}{dist['obj2_id']}")
        
        # Calculate percentage differences
        pct_diffs = [(d25 - d2) / d2 * 100 for d2, d25 in zip(distances_2d, distances_25d)]
        
        # Create figure with 4 subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('2D vs 2.5D Distance Comparison Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # Chart 1: Top 15 Pairs Bar Comparison
        ax1 = fig.add_subplot(gs[0, :])
        n_show = min(15, len(distances))
        x_pos = np.arange(n_show)
        width = 0.35
        
        ax1.bar(x_pos - width/2, distances_2d[:n_show], width, label='2D Distance (Pixel Only)', 
                color='#3498db', alpha=0.8)
        ax1.bar(x_pos + width/2, distances_25d[:n_show], width, label='2.5D Distance (With Depth)', 
                color='#e74c3c', alpha=0.8)
        
        ax1.set_xlabel('Object Pairs (Sorted by 2D Distance)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Distance (pixels)', fontsize=11, fontweight='bold')
        ax1.set_title('Top 15 Closest Pairs: 2D vs 2.5D Comparison', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels[:n_show], rotation=45, ha='right', fontsize=9)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars with depth impact highlighting
        for i, (d2, d25) in enumerate(zip(distances_2d[:n_show], distances_25d[:n_show])):
            pct_change = ((d25 - d2) / d2 * 100)
            
            # Highlight high-impact pairs
            if pct_change >= 10:
                # High impact - add colored background
                ax1.text(i - width/2, d2, f'{d2:.0f}', ha='center', va='bottom', fontsize=8)
                ax1.text(i + width/2, d25, f'{d25:.0f}', ha='center', va='bottom', fontsize=8, 
                        fontweight='bold', color='red')
                # Add impact label
                ax1.text(i, max(d2, d25) * 1.05, f'+{pct_change:.1f}%', ha='center', va='bottom', 
                        fontsize=7, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            elif pct_change >= 5:
                # Medium impact
                ax1.text(i - width/2, d2, f'{d2:.0f}', ha='center', va='bottom', fontsize=8)
                ax1.text(i + width/2, d25, f'{d25:.0f}', ha='center', va='bottom', fontsize=8, 
                        fontweight='bold', color='orange')
                ax1.text(i, max(d2, d25) * 1.05, f'+{pct_change:.1f}%', ha='center', va='bottom', 
                        fontsize=7, color='orange', fontweight='bold')
            else:
                # Low impact
                ax1.text(i - width/2, d2, f'{d2:.0f}', ha='center', va='bottom', fontsize=8)
                ax1.text(i + width/2, d25, f'{d25:.0f}', ha='center', va='bottom', fontsize=8)
        
        # Chart 2: Scatter Plot - Correlation between 2D and 2.5D
        ax2 = fig.add_subplot(gs[1, 0])
        
        scatter = ax2.scatter(distances_2d, distances_25d, c=pct_diffs, cmap='RdYlGn_r', 
                             s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add diagonal line (y=x) showing where they'd be equal
        max_val = max(max(distances_2d), max(distances_25d))
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal Distance Line')
        
        ax2.set_xlabel('2D Distance (pixels)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('2.5D Distance (pixels)', fontsize=11, fontweight='bold')
        ax2.set_title('2D vs 2.5D Distance Correlation', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend(loc='upper left', fontsize=9)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('% Increase', fontsize=10)
        
        # Add statistics text
        correlation = np.corrcoef(distances_2d, distances_25d)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.4f}\nAvg % Increase: {np.mean(pct_diffs):.2f}%', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Chart 3: Histogram of Percentage Differences
        ax3 = fig.add_subplot(gs[1, 1])
        
        n_bins = min(20, len(pct_diffs) // 2 + 1)
        counts, bins, patches = ax3.hist(pct_diffs, bins=n_bins, color='#2ecc71', 
                                         alpha=0.7, edgecolor='black', linewidth=1.2)
        
        # Color code bins
        for i, patch in enumerate(patches):
            if bins[i] < 5:
                patch.set_facecolor('#2ecc71')  # Green - low impact
            elif bins[i] < 20:
                patch.set_facecolor('#f39c12')  # Orange - medium impact
            else:
                patch.set_facecolor('#e74c3c')  # Red - high impact
        
        ax3.axvline(np.mean(pct_diffs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pct_diffs):.2f}%')
        ax3.axvline(np.median(pct_diffs), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(pct_diffs):.2f}%')
        
        ax3.set_xlabel('Percentage Increase (%)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Number of Object Pairs', fontsize=11, fontweight='bold')
        ax3.set_title('Distribution of Depth Impact', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add impact categories
        low = sum(1 for p in pct_diffs if p < 5)
        medium = sum(1 for p in pct_diffs if 5 <= p < 20)
        high = sum(1 for p in pct_diffs if p >= 20)
        
        ax3.text(0.65, 0.95, f'Impact Categories:\n'
                             f'Low (<5%): {low} pairs\n'
                             f'Medium (5-20%): {medium} pairs\n'
                             f'High (>20%): {high} pairs',
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Chart 4: Ranking Changes
        ax4 = fig.add_subplot(gs[2, :])
        
        # Sort and create ranking maps
        distances_2d_sorted_idx = np.argsort(distances_2d)
        distances_25d_sorted_idx = np.argsort(distances_25d)
        
        rank_2d = {idx: rank for rank, idx in enumerate(distances_2d_sorted_idx)}
        rank_25d = {idx: rank for rank, idx in enumerate(distances_25d_sorted_idx)}
        
        rank_changes = []
        indices = []
        for i in range(len(distances_2d)):
            change = rank_2d[i] - rank_25d[i]
            if change != 0:
                rank_changes.append(change)
                indices.append(i)
        
        if rank_changes:
            # Sort by magnitude of change
            sorted_pairs = sorted(zip(indices, rank_changes), key=lambda x: abs(x[1]), reverse=True)
            top_n = min(20, len(sorted_pairs))
            indices_sorted = [x[0] for x in sorted_pairs[:top_n]]
            changes_sorted = [x[1] for x in sorted_pairs[:top_n]]
            labels_sorted = [labels[i] for i in indices_sorted]
            
            colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in changes_sorted]
            
            bars = ax4.barh(range(len(changes_sorted)), changes_sorted, color=colors, alpha=0.7, edgecolor='black')
            
            ax4.set_yticks(range(len(changes_sorted)))
            ax4.set_yticklabels(labels_sorted, fontsize=9)
            ax4.set_xlabel('Ranking Change (Positive = Moved Closer with Depth)', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Object Pairs', fontsize=11, fontweight='bold')
            ax4.set_title(f'Ranking Changes When Depth Added (Top {top_n} Changes)', fontsize=13, fontweight='bold')
            ax4.axvline(0, color='black', linewidth=1)
            ax4.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, changes_sorted)):
                x_pos = val + (0.2 if val > 0 else -0.2)
                ax4.text(x_pos, bar.get_y() + bar.get_height()/2, f'{abs(val):.0f}', 
                        ha='left' if val > 0 else 'right', va='center', fontsize=9, fontweight='bold')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#2ecc71', label='Closer in 2.5D'),
                             Patch(facecolor='#e74c3c', label='Farther in 2.5D')]
            ax4.legend(handles=legend_elements, loc='lower right', fontsize=10)
            
            # Add summary text
            ax4.text(0.02, 0.98, f'Total rank changes: {len(rank_changes)}/{len(distances)} pairs',
                    transform=ax4.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        else:
            ax4.text(0.5, 0.5, 'No Ranking Changes\nDepth had minimal impact on distance ordering',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=ax4.transAxes)
            ax4.set_xlim([0, 1])
            ax4.set_ylim([0, 1])
            ax4.axis('off')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  - Depth comparison charts: {os.path.basename(output_path)}")
    
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
    
    def _create_worker_heatmap(self, image: np.ndarray, detections: List[Dict], output_path: str):
        """
        Create a worker density heatmap showing concentration of workers across the image.
        
        VISUALIZATION:
        ---------------
        - Gaussian heatmap overlay showing worker locations and density
        - Warmer colors (red) = higher worker concentration
        - Cooler colors (blue) = lower worker concentration
        - Includes original image as background for reference
        
        Args:
            image: Original input image
            detections: List of detected objects
            output_path: Path to save the heatmap
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Extract worker positions
        worker_positions = []
        worker_depths = []
        
        for det in detections:
            if det['class'].lower() == 'worker':
                center_x, center_y = det['center']
                worker_positions.append([center_x, center_y])
                worker_depths.append(det['depth_avg'])
        
        h, w = image.shape[:2]
        
        # Subplot 1: Original image with worker positions marked
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Worker Locations in Image', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        
        if worker_positions:
            positions_array = np.array(worker_positions)
            # Plot workers as dots
            scatter = ax1.scatter(positions_array[:, 0], positions_array[:, 1], 
                                 c=worker_depths, cmap='RdYlGn_r', s=200, 
                                 edgecolors='white', linewidth=2, zorder=5, alpha=0.7)
            plt.colorbar(scatter, ax=ax1, label='Avg Depth')
            
            # Add labels
            for i, (x, y) in enumerate(positions_array):
                ax1.text(x+10, y+10, f'W{i}', fontsize=9, color='white', 
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Subplot 2: Density heatmap
        if worker_positions:
            # Create empty heatmap
            heatmap = np.zeros((h, w), dtype=np.float32)
            
            # Apply Gaussian at each worker position
            sigma = 50  # Gaussian standard deviation (in pixels)
            for x, y in worker_positions:
                y_grid, x_grid = np.ogrid[:h, :w]
                gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
                heatmap += gaussian
            
            # Normalize heatmap
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            # Display heatmap
            im = ax2.imshow(heatmap, cmap='hot', interpolation='bilinear')
            ax2.set_title('Worker Density Heatmap', fontsize=12, fontweight='bold')
            ax2.set_xlabel('X Position (pixels)')
            ax2.set_ylabel('Y Position (pixels)')
            
            plt.colorbar(im, ax=ax2, label='Worker Density')
            
            # Add statistics
            max_density_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            max_density_pos = (max_density_idx[1], max_density_idx[0])
            
            stats_text = f'Total Workers: {len(worker_positions)}\n'
            stats_text += f'Max Density at: ({max_density_pos[0]}, {max_density_pos[1]})\n'
            stats_text += f'Coverage Area: {np.sum(heatmap > 0.1):.0f} pixels'
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No workers detected', ha='center', va='center',
                    fontsize=14, transform=ax2.transAxes)
            ax2.set_title('Worker Density Heatmap', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  - Worker heatmap: {os.path.basename(output_path)}")



def main():
    """
    Main execution function demonstrating pipeline usage.
    """
    print("\n" + "="*80)
    print("INTEGRATED DETECTION + DEPTH PIPELINE - PyTorch Version")
    print("="*80 + "\n")
    
    # Configuration
    YOLO_MODEL = os.path.join(os.path.dirname(__file__), '..', 'YOLO', 
                              'runs', 'detect', 'train', 'weights', 'best.pt')
    
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
    pipeline = IntegratedPipelinePyTorch(
        yolo_model_path=YOLO_MODEL,
        depth_encoder='vitb',  # Use 'vits' for faster inference
        yolo_conf_threshold=0.51,
        yolo_iou_threshold=0.45,
        depth_scale_factor=3.0  # Adjust based on scene depth variation (1.0-10.0)
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
            print(f"  {i}. {dist['obj1_class']} <-> {dist['obj2_class']}: "
                  f"{dist['euclidean']:.1f}px apart, depth diff: {dist['depth_diff']:.1f}")
    
    print(f"\n✅ All results saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
