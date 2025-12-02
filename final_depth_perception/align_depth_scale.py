#!/usr/bin/env python3
"""
Scale a relative depth map (e.g., from Depth Anything / MiDaS) to metric meters using a known-distance ROI.

Usage:
  python align_depth_scale.py --raw depth_result_raw.png --roi 100 200 300 300 --known-distance 8.5 --output depth_metric.png

Arguments:
  --raw: path to 16-bit raw depth PNG saved by run_tflite_depth.py
  --roi: x y w h region where the scene is approximately planar at a known distance
  --known-distance: distance in meters for that ROI

Effect:
  Computes the median depth value in ROI (d0). Scales the entire map by s = Z0 / d0.
  Writes a 32-bit floating-point single-channel TIFF/PNG (depending on pillow support) and a colorized PNG.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def load_raw_16(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im, dtype=np.uint16)
    # Convert back to [0,1] float based on 16-bit scaling used in run_tflite_depth.py
    f = arr.astype(np.float32) / 65535.0
    return f


def colorize(depth: np.ndarray) -> Image.Image:
    try:
        import matplotlib.pyplot as plt
        d = depth.copy()
        d -= np.nanmin(d)
        d /= (np.nanmax(d) + 1e-6)
        cmap = plt.get_cmap("viridis")
        rgb = (cmap(d)[..., :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
    except Exception:
        d = depth.copy()
        d -= np.nanmin(d)
        d /= (np.nanmax(d) + 1e-6)
        d8 = (d * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(d8, mode="L")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--roi", type=int, nargs=4, metavar=("x", "y", "w", "h"), required=True)
    ap.add_argument("--known-distance", type=float, required=True)
    ap.add_argument("--output", type=Path, default=Path("depth_metric.png"))
    args = ap.parse_args()

    d = load_raw_16(args.raw)  # [0,1] range
    x, y, w, h = args.roi
    roi = d[y:y+h, x:x+w]
    if roi.size == 0:
        raise ValueError("ROI is empty; check coordinates")
    d0 = float(np.median(roi))
    if d0 <= 0:
        raise ValueError("Median depth in ROI is non-positive; choose a different region")

    # Scale to meters
    s = args.known_distance / d0
    depth_m = d * s

    # Save metric depth (float32 TIFF/PNG fallback)
    metric_path = args.output.with_suffix(".tiff")
    Image.fromarray(depth_m.astype(np.float32)).save(metric_path)

    # Save colorized preview
    color = colorize(depth_m)
    color.save(args.output)

    print(f"Scaled with s={s:.6f}. Saved metric depth: {metric_path} and colorized: {args.output}")


if __name__ == "__main__":
    main()
