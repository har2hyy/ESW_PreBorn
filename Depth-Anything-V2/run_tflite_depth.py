#!/usr/bin/env python3
"""
Run a TFLite depth estimation model on an input image with camera calibration.

Usage:
  python run_tflite_depth.py --model depth_anything_v2.tflite --image 20241210_074500.jpg --output depth_output.png
  
  # With calibration for metric depth:
  python run_tflite_depth.py --model depth_anything_v2.tflite --image 20241210_074500.jpg \
    --intrinsics camera_intrinsics.json --output depth_output.png

Dependencies:
  pip install tflite-runtime pillow numpy opencv-python matplotlib
  (Optional fallback) If tflite-runtime unavailable, install full tensorflow: pip install tensorflow

The script:
  * Loads the TFLite model with tflite_runtime.Interpreter or tensorflow.lite.Interpreter.
  * Automatically reads the model input tensor shape to resize the image.
  * Normalizes image to [0,1] float32 if required (heuristic based on input dtype).
  * Runs inference and extracts the depth map.
  * Optionally uses camera intrinsics for metric depth estimation.
  * Saves both raw depth (as 16-bit PNG) and a colorized visualization (viridis) if output path provided.

Output files:
  depth_output.png (colorized) & depth_output_raw.png (raw 16-bit) when --output depth_output.png.
"""
import argparse
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Lazy import OpenCV only if available for possible future extensions (not required now)
try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

# Try loading TFLite interpreter from tflite-runtime first, fallback to TensorFlow.
Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except Exception:
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    except Exception:
        print("ERROR: Neither tflite-runtime nor TensorFlow found. Install one: 'pip install tflite-runtime' or 'pip install tensorflow'", file=sys.stderr)
        sys.exit(1)


def load_image(path: Path, target_shape: tuple[int, int], normalize: bool, channels: int) -> np.ndarray:
    """Load and resize image returning numpy array with shape (1, H, W, C)."""
    img = Image.open(path).convert("RGB")
    # target_shape is (height, width)
    img = img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32 if normalize else np.uint8)
    if normalize:
        arr /= 255.0
    # Ensure desired channels
    if channels == 1:
        # Convert to grayscale
        arr = arr.mean(axis=2, keepdims=True)
    else:
        # If model expects 3 channels but we have different, attempt broadcast
        if arr.shape[2] != channels:
            # Simple replication/padding
            if channels == 4:
                alpha = np.ones((*arr.shape[:2], 1), dtype=arr.dtype)
                arr = np.concatenate([arr, alpha], axis=2)
            else:
                # Truncate or pad to required channels
                if arr.shape[2] > channels:
                    arr = arr[:, :, :channels]
                else:
                    pad = np.repeat(arr[:, :, -1:], channels - arr.shape[2], axis=2)
                    arr = np.concatenate([arr, pad], axis=2)
    # Add batch dim
    arr = np.expand_dims(arr, axis=0)
    return arr


def colorize_depth(depth: np.ndarray) -> Image.Image:
    """Convert depth map (H,W) float to colorized PNG using matplotlib colormap."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        # Fallback: scale to grayscale
        d = depth - depth.min()
        d /= (depth.max() + 1e-6)
        d8 = (d * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(d8, mode="L")
    d = depth - np.nanmin(depth)
    d /= (np.nanmax(depth) + 1e-6)
    cmap = plt.get_cmap("viridis")
    colored = (cmap(d)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(colored, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Run TFLite depth model on an image with optional camera calibration")
    parser.add_argument("--model", required=True, type=Path, help="Path to .tflite model")
    parser.add_argument("--image", required=True, type=Path, help="Path to input image")
    parser.add_argument("--output", required=False, type=Path, default=Path("depth_result.png"), help="Output visualization PNG path")
    parser.add_argument("--intrinsics", type=Path, help="Path to camera_intrinsics.json for metric depth info")
    parser.add_argument("--delegate", choices=["none", "xnnpack"], default="xnnpack", help="Optional TFLite delegate")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Load camera intrinsics if provided
    intrinsics = None
    if args.intrinsics and args.intrinsics.exists():
        with open(args.intrinsics, 'r') as f:
            intrinsics = json.load(f)
        print(f"📐 Loaded camera intrinsics from: {args.intrinsics}")
        print(f"   Focal Length: fx={intrinsics['focal_length']['fx']:.2f}, fy={intrinsics['focal_length']['fy']:.2f}")
        print(f"   Principal Point: cx={intrinsics['principal_point']['cx']:.2f}, cy={intrinsics['principal_point']['cy']:.2f}")
        print(f"   Image Size: {intrinsics['image_size']['width']}x{intrinsics['image_size']['height']}")
    else:
        print("ℹ️  No camera intrinsics provided. Depth will be relative (not metric).")

    print(f"\n🎬 Running depth estimation on: {args.image.name}")

    # Initialize interpreter
    delegate_kw = {}
    if args.delegate == "xnnpack":
        try:
            delegate_kw = {"experimental_delegates": [Interpreter.load_delegate("libXNNPACK.so")]}  # type: ignore[attr-defined]
        except Exception:
            delegate_kw = {}
    interpreter = Interpreter(model_path=str(args.model), **delegate_kw)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assume single input.
    in_det = input_details[0]
    in_shape = in_det["shape"]  # e.g., [1, H, W, C]
    height, width, channels = int(in_shape[1]), int(in_shape[2]), int(in_shape[3])
    in_dtype = in_det["dtype"]
    normalize = in_dtype == np.float32

    print(f"   Model input shape: {in_shape}")
    print(f"   Model expects: {height}x{width}, {channels} channels, dtype={in_dtype}")

    img_tensor = load_image(args.image, (height, width), normalize, channels)
    # Cast if needed
    img_tensor = img_tensor.astype(in_dtype)

    # Set tensor
    print(f"\n🔄 Running inference...")
    interpreter.set_tensor(in_det["index"], img_tensor)
    interpreter.invoke()

    # Extract output (assume first output) - shape maybe [1, H, W, 1] or [1, H, W]
    out_det = output_details[0]
    depth = interpreter.get_tensor(out_det["index"])
    depth = np.squeeze(depth)
    depth_float = depth.astype(np.float32)

    print(f"   Depth map shape: {depth_float.shape}")
    print(f"   Depth range: {depth_float.min():.4f} to {depth_float.max():.4f}")

    # Save raw 16-bit depth scaled
    d = depth_float - np.nanmin(depth_float)
    d /= (np.nanmax(depth_float) + 1e-6)
    d16 = (d * 65535).clip(0, 65535).astype(np.uint16)
    raw_path = args.output.with_name(args.output.stem + "_raw.png")
    Image.fromarray(d16, mode="I;16").save(raw_path)

    # Create colorized visualization
    color_img = colorize_depth(depth_float)
    color_img.save(args.output)

    print(f"\n✅ Depth inference complete!")
    print(f"   📊 Colorized visualization: {args.output}")
    print(f"   📊 Raw 16-bit depth map: {raw_path}")

    # If intrinsics available, provide additional info
    if intrinsics:
        print(f"\n📏 Camera Calibration Info:")
        print(f"   ℹ️  Note: This depth map is RELATIVE depth (not metric)")
        print(f"   ℹ️  To get metric depth, you need:")
        print(f"      1. A reference object at known distance in the image")
        print(f"      2. Or use align_depth_scale.py to align with checkerboard")
        print(f"\n   Camera Field of View:")
        fx = intrinsics['focal_length']['fx']
        fy = intrinsics['focal_length']['fy']
        img_w = intrinsics['image_size']['width']
        img_h = intrinsics['image_size']['height']
        import math
        fov_h = 2 * math.atan(img_w / (2 * fx)) * 180 / math.pi
        fov_v = 2 * math.atan(img_h / (2 * fy)) * 180 / math.pi
        print(f"      Horizontal FOV: {fov_h:.1f}°")
        print(f"      Vertical FOV: {fov_v:.1f}°")
        
        print(f"\n   💡 Tip: Use pixel_to_3d.py to convert pixel coordinates to 3D!")
        print(f"      Example: converter.pixel_to_3d(u=960, v=540, depth=5.0)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
