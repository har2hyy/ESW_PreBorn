#!/usr/bin/env python3
"""
Camera Calibration from Checkerboard Videos

This script extracts camera intrinsic parameters (focal length, principal point,
distortion coefficients) from videos containing a checkerboard pattern.

Usage:
    python calibrate_camera.py --videos video1.mp4 video2.mp4 \
        --pattern-cols 8 --pattern-rows 6 \
        --square-size 0.02 \
        --output calibration_results.json
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
from datetime import datetime


def extract_frames_from_video(video_path, frame_skip=30, max_frames=None):
    """
    Extract frames from video at regular intervals.
    
    Args:
        video_path: Path to video file
        frame_skip: Extract every Nth frame
        max_frames: Maximum number of frames to extract (None = all)
    
    Returns:
        List of (frame_number, frame_image) tuples
    """
    print(f"\n📹 Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   Total frames: {total_frames}, FPS: {fps:.2f}")
    
    frames = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frames.append((frame_count, frame.copy()))
            extracted_count += 1
            
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"   ✓ Extracted {len(frames)} frames")
    
    return frames


def find_checkerboard_corners(frame, pattern_size, show_progress=True):
    """
    Find checkerboard corners in a frame.
    
    Args:
        frame: Input image
        pattern_size: (cols, rows) - number of internal corners
        show_progress: Whether to print detection status
    
    Returns:
        Tuple of (success, corners) or (False, None)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    
    if ret:
        # Refine corner positions to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners
    
    return False, None


def calibrate_camera_from_videos(video_paths, pattern_size, square_size, 
                                 frame_skip=30, max_frames_per_video=100,
                                 save_debug_images=False, debug_dir=None):
    """
    Perform camera calibration from multiple videos containing checkerboard patterns.
    
    Args:
        video_paths: List of paths to calibration videos
        pattern_size: (cols, rows) - number of internal corners in checkerboard
        square_size: Size of each checkerboard square in meters
        frame_skip: Extract every Nth frame from videos
        max_frames_per_video: Maximum frames to extract per video
        save_debug_images: Whether to save frames with detected corners
        debug_dir: Directory to save debug images
    
    Returns:
        Dictionary containing calibration results
    """
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...., (cols-1, rows-1, 0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale by actual square size
    
    # Arrays to store object points and image points from all frames
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    successful_frames = []
    image_size = None
    
    # Create debug directory if needed
    if save_debug_images and debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each video
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"⚠️  Warning: Video not found: {video_path}")
            continue
        
        frames = extract_frames_from_video(video_path, frame_skip, max_frames_per_video)
        
        print(f"\n🔍 Detecting checkerboard in {len(frames)} frames from {Path(video_path).name}...")
        detected_count = 0
        
        for frame_num, frame in frames:
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])
            
            ret, corners = find_checkerboard_corners(frame, pattern_size, show_progress=False)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                successful_frames.append((video_path, frame_num))
                detected_count += 1
                
                # Save debug image if requested
                if save_debug_images and debug_dir:
                    debug_frame = frame.copy()
                    cv2.drawChessboardCorners(debug_frame, pattern_size, corners, ret)
                    debug_filename = f"{Path(video_path).stem}_frame_{frame_num:06d}.jpg"
                    cv2.imwrite(os.path.join(debug_dir, debug_filename), debug_frame)
        
        print(f"   ✓ Detected checkerboard in {detected_count}/{len(frames)} frames")
    
    # Perform calibration
    if len(objpoints) < 10:
        print(f"\n❌ Error: Not enough valid frames for calibration (found {len(objpoints)}, need at least 10)")
        print("   Try:")
        print("   - Reducing --frame-skip to extract more frames")
        print("   - Ensuring the checkerboard is clearly visible")
        print("   - Checking that pattern size is correct (internal corners)")
        return None
    
    print(f"\n🎯 Calibrating camera using {len(objpoints)} valid frames...")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    if not ret:
        print("❌ Calibration failed!")
        return None
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    
    # Extract calibration parameters
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    # Prepare results
    results = {
        'calibration_date': datetime.now().isoformat(),
        'pattern_size': {
            'cols': pattern_size[0],
            'rows': pattern_size[1]
        },
        'square_size_meters': square_size,
        'image_size': {
            'width': image_size[0],
            'height': image_size[1]
        },
        'frames_used': len(objpoints),
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist()[0],
        'focal_length': {
            'fx': fx,
            'fy': fy,
            'fx_pixels': fx,
            'fy_pixels': fy
        },
        'principal_point': {
            'cx': cx,
            'cy': cy
        },
        'reprojection_error_pixels': mean_error,
        'videos_used': [str(Path(v).name) for v in video_paths],
        'successful_frames': [
            {'video': str(Path(v).name), 'frame': f} 
            for v, f in successful_frames
        ]
    }
    
    return results


def print_calibration_summary(results):
    """Print a formatted summary of calibration results."""
    
    print("\n" + "="*70)
    print("📊 CAMERA CALIBRATION RESULTS")
    print("="*70)
    
    print(f"\n📐 Pattern Information:")
    print(f"   Checkerboard size: {results['pattern_size']['cols']} x {results['pattern_size']['rows']} internal corners")
    print(f"   Square size: {results['square_size_meters']*100:.1f} cm ({results['square_size_meters']*1000:.1f} mm)")
    
    print(f"\n🖼️  Image Information:")
    print(f"   Resolution: {results['image_size']['width']} x {results['image_size']['height']} pixels")
    print(f"   Frames used: {results['frames_used']}")
    
    print(f"\n🎯 Camera Intrinsics:")
    print(f"   Focal Length (fx): {results['focal_length']['fx']:.2f} pixels")
    print(f"   Focal Length (fy): {results['focal_length']['fy']:.2f} pixels")
    print(f"   Principal Point (cx): {results['principal_point']['cx']:.2f} pixels")
    print(f"   Principal Point (cy): {results['principal_point']['cy']:.2f} pixels")
    
    print(f"\n📏 Distortion Coefficients:")
    dist = results['distortion_coefficients']
    print(f"   k1: {dist[0]:.6f}")
    print(f"   k2: {dist[1]:.6f}")
    print(f"   p1: {dist[2]:.6f}")
    print(f"   p2: {dist[3]:.6f}")
    print(f"   k3: {dist[4]:.6f}")
    
    print(f"\n✅ Quality Metrics:")
    print(f"   Reprojection Error: {results['reprojection_error_pixels']:.4f} pixels")
    
    if results['reprojection_error_pixels'] < 0.5:
        quality = "Excellent ✨"
    elif results['reprojection_error_pixels'] < 1.0:
        quality = "Good ✓"
    elif results['reprojection_error_pixels'] < 2.0:
        quality = "Acceptable ⚠️"
    else:
        quality = "Poor - Consider recalibrating ❌"
    
    print(f"   Quality Assessment: {quality}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Camera calibration from checkerboard videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # For a 7x9 checkerboard (internal corners) with 2cm squares:
  python calibrate_camera.py --videos calib1.mp4 calib2.mp4 \\
      --pattern-cols 8 --pattern-rows 6 --square-size 0.02
  
  # Note: If your checkerboard has 7x9 SQUARES, it has 6x8 INTERNAL CORNERS
  # A 7x9 checkerboard means 7 columns and 9 rows of squares
  # So internal corners = (7-1, 9-1) = (6, 8)
        """
    )
    
    parser.add_argument('--videos', nargs='+', required=True,
                       help='Paths to calibration videos')
    parser.add_argument('--pattern-cols', type=int, required=True,
                       help='Number of internal corners in checkerboard (columns)')
    parser.add_argument('--pattern-rows', type=int, required=True,
                       help='Number of internal corners in checkerboard (rows)')
    parser.add_argument('--square-size', type=float, required=True,
                       help='Size of each checkerboard square in METERS (e.g., 0.02 for 2cm)')
    parser.add_argument('--frame-skip', type=int, default=30,
                       help='Extract every Nth frame (default: 30)')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Maximum frames to extract per video (default: 100)')
    parser.add_argument('--output', type=str, default='camera_intrinsics.json',
                       help='Output JSON file for calibration results')
    parser.add_argument('--debug-images', action='store_true',
                       help='Save frames with detected corners for debugging')
    parser.add_argument('--debug-dir', type=str, default='calibration_debug',
                       help='Directory for debug images (default: calibration_debug)')
    
    args = parser.parse_args()
    
    print("🎬 Camera Calibration Tool")
    print(f"   Videos: {', '.join([Path(v).name for v in args.videos])}")
    print(f"   Pattern: {args.pattern_cols} x {args.pattern_rows} internal corners")
    print(f"   Square size: {args.square_size*100:.1f} cm")
    print(f"   Frame skip: {args.frame_skip}")
    print(f"   Max frames per video: {args.max_frames}")
    
    # Run calibration
    results = calibrate_camera_from_videos(
        video_paths=args.videos,
        pattern_size=(args.pattern_cols, args.pattern_rows),
        square_size=args.square_size,
        frame_skip=args.frame_skip,
        max_frames_per_video=args.max_frames,
        save_debug_images=args.debug_images,
        debug_dir=args.debug_dir if args.debug_images else None
    )
    
    if results is None:
        print("\n❌ Calibration failed!")
        return 1
    
    # Print summary
    print_calibration_summary(results)
    
    # Save results
    output_path = args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Also save camera matrix in OpenCV format for easy loading
    cv_format_path = output_path.replace('.json', '_opencv_format.npz')
    np.savez(cv_format_path,
             camera_matrix=np.array(results['camera_matrix']),
             dist_coeffs=np.array(results['distortion_coefficients']))
    print(f"💾 OpenCV format saved to: {cv_format_path}")
    
    print("\n✨ Calibration complete!")
    print("\nNext steps:")
    print(f"  1. Use the calibration data in {output_path}")
    print("  2. Run undistort_images.py to undistort your construction site images")
    print("  3. Use the intrinsics for depth estimation and 3D reconstruction")
    
    return 0


if __name__ == '__main__':
    exit(main())
