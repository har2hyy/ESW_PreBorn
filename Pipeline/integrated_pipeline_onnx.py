#!/usr/bin/env python3
"""
Integrated YOLOv11 + Depth Anything V2 Pipeline (ONNX Version)
===============================================================

Production pipeline for QIDK NPU deployment using ONNX detector.
Combines object detection with depth estimation for comprehensive spatial analysis.

This pipeline uses OptimizedYOLOv11ONNX for detection, optimized for:
- QIDK NPU deployment via DLC conversion
- Low-memory embedded environments
- imgsz=1024 matching training resolution

Usage (default paths configured inside the script):
    conda activate pipeline
    python integrated_pipeline_onnx.py --help

Author: Pipeline Integration System
Date: November 2025
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure repository modules can be imported
PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent
sys.path.append(str(ROOT_DIR / "YOLO"))
sys.path.append(str(ROOT_DIR / "Depth-Anything-V2"))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

# Import the optimized ONNX detector (works better than TFLite for QIDK)
sys.path.append(str(ROOT_DIR / "YOLO" / "scripts" / "testing"))
from optimized_yolo_onnx import OptimizedYOLOv11ONNX  # noqa: E402


class IntegratedPipelineONNX:
    """Full detection + depth pipeline using ONNX detector for QIDK NPU deployment."""

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(
        self,
        yolo_model_path: Path,
        yolo_classes_path: Path,
        depth_encoder: str = "vitb",
        depth_weights_path: Path | None = None,
        yolo_conf_threshold: float = 0.51,
        yolo_nms_threshold: float = 0.45,
        depth_scale_factor: float = 3.0,
        output_suffix: str = "_tflite",
    ) -> None:
        if depth_encoder not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported depth encoder '{depth_encoder}'. Choose from {list(self.MODEL_CONFIGS)}")

        self.depth_scale_factor = depth_scale_factor
        self.output_suffix = output_suffix if output_suffix.startswith("_") else f"_{output_suffix}"

        print("=" * 80)
        print("INITIALIZING INTEGRATED PIPELINE (ONNX Version)")
        print("=" * 80)

        print("\n[1/2] Loading YOLOv11 ONNX model...")
        self.detector = OptimizedYOLOv11ONNX(
            model_path=str(yolo_model_path)
        )
        # Note: The ONNX detector has built-in class names and thresholds
        # Override if needed
        self.detector.conf_threshold = yolo_conf_threshold
        self.detector.iou_threshold = yolo_nms_threshold
        
        self.model_variant = Path(yolo_model_path).name
        print(f"✓ Detector ready: {self.model_variant}")
        print(f"  Confidence threshold: {self.detector.conf_threshold}")
        print(f"  IoU threshold: {self.detector.iou_threshold}")

        print(f"\n[2/2] Loading Depth Anything V2 ({depth_encoder})...")
        weights = depth_weights_path or ROOT_DIR / "Depth-Anything-V2" / "checkpoints" / f"depth_anything_v2_{depth_encoder}.pth"
        if not weights.exists():
            raise FileNotFoundError(f"Depth weights not found at {weights}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.depth_model = DepthAnythingV2(**self.MODEL_CONFIGS[depth_encoder])
        state_dict = torch.load(weights, map_location="cpu")
        self.depth_model.load_state_dict(state_dict)
        self.depth_model = self.depth_model.to(self.device).eval()
        print("✓ Depth Anything V2 loaded successfully")

        print("\n" + "=" * 80)
        print("PIPELINE READY")
        print(f"YOLO model: {self.model_variant}")
        print(f"Depth scale factor: {self.depth_scale_factor}x")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_image(self, image_path: Path, output_dir: Path | None = None) -> Dict:
        image_path = Path(image_path)
        if output_dir is None:
            output_dir = image_path.parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"PROCESSING: {image_path.name}")
        print(f"{'=' * 80}\n")

        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        base_name = image_path.stem

        # Stage 1 - YOLO detections
        print("[Stage 1/5] YOLO object detection (ONNX)...")
        det_start = time.time()
        
        # Load image for detection
        boxes, confidences, class_ids = self.detector.detect(original_image)
        
        det_time = (time.time() - det_start) * 1000
        print(f"✓ {len(boxes)} detections in {det_time:.1f} ms")

        # Class names mapping
        class_names_map = {0: 'worker', 1: 'truck', 2: 'bike', 3: 'bulldozer', 4: 'car'}
        
        detections_raw = []
        for idx, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names_map[int(class_id)]
            print(f"  {idx + 1}. {class_name} ({conf:.3f}) @ [{x1}, {y1}, {x2}, {y2}]")
            detections_raw.append({"bbox": [x1, y1, x2, y2], "confidence": float(conf), "class_id": int(class_id), "class": class_name})
        
        # Create annotated image
        annotated_image = original_image.copy()
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names_map[int(class_id)]
            
            # Draw bbox
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_image, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


        # Stage 2 - Depth estimation
        print("\n[Stage 2/5] Depth estimation...")
        depth_start = time.time()
        depth_map_raw = self.depth_model.infer_image(original_image, 518)
        depth_time = (time.time() - depth_start) * 1000
        depth_norm = (depth_map_raw - depth_map_raw.min()) / (depth_map_raw.max() - depth_map_raw.min() + 1e-8)
        depth_map = (depth_norm * 255.0).astype(np.uint8)
        print(f"✓ Depth map ready in {depth_time:.1f} ms (range {depth_map.min()}-{depth_map.max()})")

        # Stage 3 - Per-detection depth stats
        print("\n[Stage 3/5] Aggregating depth statistics per object...")
        detections = self._attach_depth_stats(detections_raw, depth_map)

        # Stage 4 - Pairwise distances
        print("\n[Stage 4/5] Calculating spatial relationships...")
        distances = self._calculate_distances(detections)
        if distances:
            print(f"✓ Computed {len(distances)} distance pairs")
            for dist in distances[:5]:
                print(
                    f"  {dist['obj1_class']} ↔ {dist['obj2_class']}: "
                    f"{dist['euclidean']:.1f}px, depth Δ {dist['depth_diff']:.1f} (scaled {dist['depth_diff_scaled']:.1f})"
                )

        # Stage 5 - Visualizations & reports
        print("\n[Stage 5/5] Rendering outputs...")
        outputs = self._render_outputs(base_name, original_image, annotated_image, depth_map, detections, distances, output_dir)

        report = {
            "image": str(image_path),
            "yolo_model": self.model_variant,
            "depth_encoder": self.depth_model.encoder if hasattr(self.depth_model, "encoder") else "DepthAnythingV2",
            "inference_times_ms": {"yolo": det_time, "depth": depth_time},
            "total_detections": len(detections),
            "detections": detections,
            "distances": distances,
            "outputs": outputs,
        }

        report_path = output_dir / f"{base_name}_analysis_report{self.output_suffix}.json"
        with report_path.open("w") as f:
            json.dump(report, f, indent=2, default=self._json_serializer)
        print(f"  - Analysis report: {report_path.name}")

        print(f"\n✓ All outputs saved to: {output_dir}")
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _attach_depth_stats(self, detections: List[Dict], depth_map: np.ndarray) -> List[Dict]:
        h, w = depth_map.shape[:2]
        enriched: List[Dict] = []

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            roi = depth_map[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            detection = {
                "id": len(enriched),
                "class": det["class"],
                "class_id": det["class_id"],
                "confidence": float(det["confidence"]),
                "bbox": [x1, y1, x2, y2],
                "center": (center_x, center_y),
                "depth_avg": float(np.mean(roi)),
                "depth_median": float(np.median(roi)),
                "depth_min": float(np.min(roi)),
                "depth_max": float(np.max(roi)),
                "depth_center": float(depth_map[center_y, center_x]),
                "area": int((x2 - x1) * (y2 - y1)),
            }
            enriched.append(detection)
            print(
                f"  Object {detection['id'] + 1}: {detection['class']} | "
                f"avg depth {detection['depth_avg']:.1f} at ({center_x}, {center_y})"
            )

        return enriched

    def _calculate_distances(self, detections: List[Dict]) -> List[Dict]:
        distances: List[Dict] = []
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                obj1 = detections[i]
                obj2 = detections[j]
                dx = obj2["center"][0] - obj1["center"][0]
                dy = obj2["center"][1] - obj1["center"][1]
                euclidean = float(np.sqrt(dx ** 2 + dy ** 2))
                depth_diff = float(abs(obj1["depth_avg"] - obj2["depth_avg"]))
                depth_diff_scaled = depth_diff * self.depth_scale_factor
                distances.append(
                    {
                        "obj1_id": obj1["id"],
                        "obj1_class": obj1["class"],
                        "obj2_id": obj2["id"],
                        "obj2_class": obj2["class"],
                        "euclidean": euclidean,
                        "horizontal": float(abs(dx)),
                        "vertical": float(abs(dy)),
                        "depth_diff": depth_diff,
                        "depth_diff_scaled": float(depth_diff_scaled),
                        "obj1_depth": obj1["depth_avg"],
                        "obj2_depth": obj2["depth_avg"],
                    }
                )

        distances.sort(key=lambda d: d["euclidean"])
        return distances

    def _render_outputs(
        self,
        base_name: str,
        original_image: np.ndarray,
        annotated_image: np.ndarray,
        depth_map: np.ndarray,
        detections: List[Dict],
        distances: List[Dict],
        output_dir: Path,
    ) -> Dict[str, str]:
        suffix = self.output_suffix
        outputs: Dict[str, str] = {}

        cmap = plt.get_cmap("Spectral_r")
        depth_colored = (cmap(depth_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        yolo_path = output_dir / f"{base_name}_yolo_detections{suffix}.jpg"
        depth_path = output_dir / f"{base_name}_depth_map{suffix}.png"
        combined_path = output_dir / f"{base_name}_combined_analysis{suffix}.jpg"

        cv2.imwrite(str(yolo_path), annotated_image)
        cv2.imwrite(str(depth_path), depth_colored)
        combined_viz = self._create_combined_visualization(annotated_image, depth_colored, detections, distances)
        cv2.imwrite(str(combined_path), combined_viz)

        outputs.update({"yolo": str(yolo_path), "depth": str(depth_path), "combined": str(combined_path)})
        print(f"  - YOLO detections: {yolo_path.name}")
        print(f"  - Depth map: {depth_path.name}")
        print(f"  - Combined analysis: {combined_path.name}")

        if len(detections) > 1:
            matrix_path = output_dir / f"{base_name}_distance_matrix{suffix}.png"
            table_path = output_dir / f"{base_name}_distance_table{suffix}.txt"
            comparison_txt = output_dir / f"{base_name}_depth_comparison{suffix}.txt"
            comparison_chart = output_dir / f"{base_name}_depth_comparison_charts{suffix}.png"
            self._create_distance_matrix(detections, distances, matrix_path)
            self._save_distance_table(detections, distances, table_path)
            self._save_depth_comparison(detections, distances, comparison_txt)
            self._create_depth_comparison_charts(detections, distances, comparison_chart)
            outputs.update(
                {
                    "distance_matrix": str(matrix_path),
                    "distance_table": str(table_path),
                    "depth_comparison": str(comparison_txt),
                    "depth_comparison_charts": str(comparison_chart),
                }
            )

        if detections:
            heatmap_path = output_dir / f"{base_name}_worker_heatmap{suffix}.png"
            self._create_worker_heatmap(original_image, detections, heatmap_path)
            outputs["worker_heatmap"] = str(heatmap_path)

        return outputs

    def _create_combined_visualization(
        self,
        yolo_image: np.ndarray,
        depth_image: np.ndarray,
        detections: List[Dict],
        distances: List[Dict],
    ) -> np.ndarray:
        h = max(yolo_image.shape[0], depth_image.shape[0])
        yolo_resized = cv2.resize(yolo_image, (int(yolo_image.shape[1] * h / yolo_image.shape[0]), h))
        depth_resized = cv2.resize(depth_image, (int(depth_image.shape[1] * h / depth_image.shape[0]), h))
        combined = cv2.hconcat([yolo_resized, np.ones((h, 20, 3), dtype=np.uint8) * 255, depth_resized])

        title_bar = np.ones((80, combined.shape[1], 3), dtype=np.uint8) * 50
        cv2.putText(
            title_bar,
            "INTEGRATED DETECTION + DEPTH ANALYSIS (ONNX)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        stats_panel = np.ones((150, combined.shape[1], 3), dtype=np.uint8) * 240
        y_offset = 30
        cv2.putText(stats_panel, f"Total detections: {len(detections)}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        if detections:
            class_counts: Dict[str, int] = {}
            for det in detections:
                class_counts[det["class"]] = class_counts.get(det["class"], 0) + 1
            y_offset += 30
            summary = ", ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
            cv2.putText(stats_panel, f"Classes: {summary}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        if distances:
            closest = distances[0]
            y_offset += 35
            cv2.putText(
                stats_panel,
                f"Closest pair: {closest['obj1_class']} ↔ {closest['obj2_class']} ({closest['euclidean']:.1f}px)",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 64, 0),
                2,
            )
            y_offset += 30
            cv2.putText(stats_panel, f"Pairwise distances: {len(distances)}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return cv2.vconcat([title_bar, combined, stats_panel])

    def _create_distance_matrix(self, detections: List[Dict], distances: List[Dict], output_path: Path) -> None:
        n = len(detections)
        euclidean_matrix = np.zeros((n, n))
        depth_matrix = np.zeros((n, n))
        for dist in distances:
            i, j = dist["obj1_id"], dist["obj2_id"]
            euclidean_matrix[i, j] = euclidean_matrix[j, i] = dist["euclidean"]
            depth_matrix[i, j] = depth_matrix[j, i] = dist["depth_diff"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        labels = [f"{d['id']}: {d['class']}" for d in detections]

        euclidean_vmax = max(2000, float(euclidean_matrix.max())) if euclidean_matrix.size else 2000
        im1 = ax1.imshow(euclidean_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=euclidean_vmax)
        ax1.set_title("Euclidean Distance (px)", fontsize=12, fontweight="bold")
        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax1.set_yticklabels(labels, fontsize=9)
        for i in range(n):
            for j in range(n):
                if i != j and euclidean_matrix[i, j] > 0:
                    color = "white" if euclidean_matrix[i, j] > np.max(euclidean_matrix) * 0.5 else "black"
                    ax1.text(j, i, f"{euclidean_matrix[i, j]:.0f}", ha="center", va="center", color=color, fontsize=8)
        plt.colorbar(im1, ax=ax1, label="pixels")

        depth_vmax = max(200, float(depth_matrix.max())) if depth_matrix.size else 200
        im2 = ax2.imshow(depth_matrix, cmap="viridis", aspect="auto", vmin=0, vmax=depth_vmax)
        ax2.set_title("Depth Difference", fontsize=12, fontweight="bold")
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax2.set_yticklabels(labels, fontsize=9)
        for i in range(n):
            for j in range(n):
                if i != j and depth_matrix[i, j] > 0:
                    color = "white" if depth_matrix[i, j] > np.max(depth_matrix) * 0.6 else "black"
                    ax2.text(j, i, f"{depth_matrix[i, j]:.1f}", ha="center", va="center", color=color, fontsize=8)
        plt.colorbar(im2, ax=ax2, label="depth units")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  - Distance matrix: {output_path.name}")

    def _save_distance_table(self, detections: List[Dict], distances: List[Dict], output_path: Path) -> None:
        with output_path.open("w") as f:
            f.write("=" * 100 + "\n")
            f.write("DISTANCE TABLE - ONNX Pipeline\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Total objects: {len(detections)}\n\n")
            f.write(f"{'ID':<5}{'Class':<15}{'Conf':<10}{'Center':<15}{'Avg Depth':<15}{'Area':<10}\n")
            f.write("-" * 100 + "\n")
            for det in detections:
                f.write(
                    f"{det['id']:<5}{det['class']:<15}{det['confidence']:<10.2f}"
                    f"({det['center'][0]},{det['center'][1]}){'':<5}{det['depth_avg']:<15.1f}{det['area']:<10}\n"
                )

            f.write("\nPairwise distances (2D vs depth-aware)\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'#':<4}{'Pair':<35}{'Euclidean':<12}{'Horiz':<10}{'Vert':<10}{'Depth Δ':<10}\n")
            f.write("-" * 100 + "\n")
            for idx, dist in enumerate(distances, 1):
                pair = f"[{dist['obj1_id']}] {dist['obj1_class']} ↔ [{dist['obj2_id']}] {dist['obj2_class']}"
                f.write(
                    f"{idx:<4}{pair:<35}{dist['euclidean']:<12.1f}{dist['horizontal']:<10.1f}"
                    f"{dist['vertical']:<10.1f}{dist['depth_diff']:<10.1f}\n"
                )

        print(f"  - Distance table: {output_path.name}")

    def _save_depth_comparison(self, detections: List[Dict], distances: List[Dict], output_path: Path) -> None:
        with output_path.open("w") as f:
            f.write("=" * 120 + "\n")
            f.write("DEPTH IMPACT ANALYSIS (2D vs 2.5D)\n")
            f.write("=" * 120 + "\n\n")

            d2 = [d["euclidean"] for d in distances]
            d25 = [np.sqrt(d["euclidean"] ** 2 + d["depth_diff_scaled"] ** 2) for d in distances]
            pct = [((d25_i - d2_i) / d2_i * 100) if d2_i else 0 for d2_i, d25_i in zip(d2, d25)]

            f.write(f"Average % increase: {np.mean(pct):.2f}%\n")
            f.write(f"Median % increase: {np.median(pct):.2f}%\n")
            f.write(f"Max % increase: {np.max(pct):.2f}%\n\n")

            f.write(f"{'Pair':<40}{'2D Dist':<12}{'2.5D Dist':<12}{'%Δ':<8}{'Depth Δ':<10}\n")
            f.write("-" * 120 + "\n")
            for dist, d25_i, pct_i in zip(distances[:25], d25[:25], pct[:25]):
                pair = f"[{dist['obj1_id']}] {dist['obj1_class']} ↔ [{dist['obj2_id']}] {dist['obj2_class']}"
                f.write(
                    f"{pair:<40}{dist['euclidean']:<12.1f}{d25_i:<12.1f}{pct_i:<8.1f}{dist['depth_diff']:<10.1f}\n"
                )

        print(f"  - Depth comparison (text): {output_path.name}")

    def _create_depth_comparison_charts(self, detections: List[Dict], distances: List[Dict], output_path: Path) -> None:
        if not distances:
            return

        d2 = np.array([d["euclidean"] for d in distances])
        d25 = np.sqrt(d2 ** 2 + np.array([d["depth_diff_scaled"] for d in distances]) ** 2)
        pct = ((d25 - d2) / np.maximum(d2, 1e-6)) * 100

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        fig.suptitle("Depth Impact Visualization", fontsize=18, fontweight="bold")

        ax1 = fig.add_subplot(gs[0, 0])
        top = min(15, len(distances))
        ax1.bar(np.arange(top) - 0.2, d2[:top], width=0.4, label="2D", color="#3498db")
        ax1.bar(np.arange(top) + 0.2, d25[:top], width=0.4, label="2.5D", color="#e74c3c")
        ax1.set_title("Closest pairs: 2D vs 2.5D")
        ax1.set_ylabel("Distance (px)")
        ax1.set_xticks(range(top))
        ax1.set_xticklabels([f"{distances[i]['obj1_class'][0]}{distances[i]['obj1_id']}↔{distances[i]['obj2_class'][0]}{distances[i]['obj2_id']}" for i in range(top)], rotation=45, ha="right")
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(d2, d25, c=pct, cmap="RdYlGn_r", edgecolors="black", linewidth=0.4)
        ax2.plot([0, max(d2.max(), d25.max())], [0, max(d2.max(), d25.max())], "k--", alpha=0.4)
        ax2.set_title("Correlation between 2D and 2.5D distances")
        ax2.set_xlabel("2D distance (px)")
        ax2.set_ylabel("2.5D distance (px)")
        plt.colorbar(scatter, ax=ax2, label="% increase")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(pct, bins=min(20, len(pct)), color="#2ecc71", edgecolor="black", alpha=0.8)
        ax3.set_title("Distribution of depth impact")
        ax3.set_xlabel("% increase when depth added")
        ax3.set_ylabel("# pairs")

        ax4 = fig.add_subplot(gs[1, 1])
        rank_2d = np.argsort(d2)
        rank_25d = np.argsort(d25)
        deltas = rank_2d - rank_25d
        ax4.bar(range(len(deltas)), deltas, color="#f1c40f")
        ax4.set_title("Ranking shifts after depth integration")
        ax4.set_xlabel("Pair index (sorted by 2D distance)")
        ax4.set_ylabel("Rank change")

        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  - Depth comparison (charts): {output_path.name}")

    def _create_worker_heatmap(self, image: np.ndarray, detections: List[Dict], output_path: Path) -> None:
        workers = [det for det in detections if det["class"].lower() == "worker"]
        if not workers:
            return

        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        sigma = 50
        for det in workers:
            cx, cy = det["center"]
            y_grid, x_grid = np.ogrid[:h, :w]
            heatmap += np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
        heatmap = heatmap / (heatmap.max() + 1e-8)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Worker detections")
        for idx, det in enumerate(workers):
            ax1.scatter(*det["center"], c="red", s=80, edgecolors="white")
            ax1.text(det["center"][0] + 10, det["center"][1] + 10, f"W{idx}", color="white", fontsize=10,
                     bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

        ax2.imshow(heatmap, cmap="hot", interpolation="bilinear")
        ax2.set_title("Worker density heatmap")
        plt.colorbar(ax2.images[0], ax=ax2, label="Relative density")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  - Worker heatmap: {output_path.name}")

    @staticmethod
    def _json_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)


def parse_args() -> argparse.Namespace:
    default_image = Path("/home/harshyy/Desktop/20250103_104457.jpg")
    default_output = Path("/home/harshyy/Desktop/pipeline_output_onnx")
    default_model = ROOT_DIR / "YOLO" / "models" / "onnx" / "best_simplified.onnx"
    default_classes = ROOT_DIR / "YOLO" / "data" / "classes.txt"
    default_depth_weights = ROOT_DIR / "Depth-Anything-V2" / "checkpoints" / "depth_anything_v2_vitb.pth"

    parser = argparse.ArgumentParser(description="Integrated YOLOv11 + Depth Anything pipeline (ONNX for QIDK)")
    parser.add_argument("--image", type=Path, default=default_image, help="Path to the input image")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory for pipeline outputs")
    parser.add_argument("--yolo-model", type=Path, default=default_model, help="Path to the ONNX model file")
    parser.add_argument("--classes", type=Path, default=default_classes, help="Path to classes.txt (optional, built-in)")
    parser.add_argument("--depth-weights", type=Path, default=default_depth_weights, help="Path to depth weights")
    parser.add_argument("--depth-encoder", choices=list(IntegratedPipelineONNX.MODEL_CONFIGS), default="vitb")
    parser.add_argument("--depth-scale", type=float, default=3.0, help="Scaling factor applied to depth differences")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="YOLO NMS threshold")
    parser.add_argument("--suffix", type=str, default="_onnx", help="Suffix appended to generated files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path_attr in ["image", "yolo_model", "classes", "depth_weights"]:
        value = getattr(args, path_attr)
        if not value.exists():
            raise FileNotFoundError(f"Required resource missing: {value}")

    pipeline = IntegratedPipelineONNX(
        yolo_model_path=args.yolo_model,
        yolo_classes_path=args.classes,
        depth_encoder=args.depth_encoder,
        depth_weights_path=args.depth_weights,
        yolo_conf_threshold=args.confidence,
        yolo_nms_threshold=args.nms,
        depth_scale_factor=args.depth_scale,
        output_suffix=args.suffix,
    )

    results = pipeline.process_image(args.image, args.output_dir)

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nTotal objects detected: {results['total_detections']}")
    if results["detections"]:
        print("\nDetected objects:")
        for det in results["detections"]:
            print(
                f"  {det['id'] + 1}. {det['class']} (conf {det['confidence']:.2f}, "
                f"avg depth {det['depth_avg']:.1f})"
            )
    if results["distances"]:
        print("\nClosest pairs:")
        for idx, dist in enumerate(results["distances"][:3], 1):
            print(
                f"  {idx}. {dist['obj1_class']} ↔ {dist['obj2_class']}: "
                f"{dist['euclidean']:.1f}px apart, depth Δ {dist['depth_diff']:.1f}"
            )
    print(f"\nOutputs written to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
