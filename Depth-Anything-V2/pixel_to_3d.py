#!/usr/bin/env python3
"""
Pixel to 3D Coordinate Converter

This script provides utilities to convert pixel coordinates and depth values
to real-world 3D coordinates using camera intrinsics from calibration.

Usage as a module:
    from pixel_to_3d import PixelTo3DConverter
    
    converter = PixelTo3DConverter('camera_intrinsics.json')
    x, y, z = converter.pixel_to_3d(u=320, v=240, depth=5.0)
    distance = converter.distance_between_points(point1_3d, point2_3d)
"""

import json
import numpy as np
import cv2
from pathlib import Path


class PixelTo3DConverter:
    """Convert pixel coordinates to 3D world coordinates using camera intrinsics."""
    
    def __init__(self, intrinsics_path):
        """
        Initialize converter with camera intrinsics.
        
        Args:
            intrinsics_path: Path to camera_intrinsics.json file
        """
        with open(intrinsics_path, 'r') as f:
            self.intrinsics = json.load(f)
        
        # Extract camera parameters
        self.fx = self.intrinsics['focal_length']['fx']
        self.fy = self.intrinsics['focal_length']['fy']
        self.cx = self.intrinsics['principal_point']['cx']
        self.cy = self.intrinsics['principal_point']['cy']
        
        self.camera_matrix = np.array(self.intrinsics['camera_matrix'])
        self.dist_coeffs = np.array(self.intrinsics['distortion_coefficients'])
        
        self.image_width = self.intrinsics['image_size']['width']
        self.image_height = self.intrinsics['image_size']['height']
    
    def pixel_to_3d(self, u, v, depth):
        """
        Convert pixel coordinates and depth to 3D world coordinates.
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            depth: Depth value in meters
        
        Returns:
            Tuple of (X, Y, Z) in meters
        """
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        
        return X, Y, Z
    
    def bbox_to_3d(self, bbox, depth_map):
        """
        Convert bounding box to 3D coordinates using depth map.
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box coordinates
            depth_map: 2D numpy array of depth values in meters
        
        Returns:
            Dictionary with 3D coordinates:
            {
                'center_3d': (X, Y, Z),
                'corners_3d': [(X, Y, Z), ...],
                'mean_depth': float,
                'bbox_2d': (x1, y1, x2, y2)
            }
        """
        x1, y1, x2, y2 = bbox
        
        # Get center of bbox
        center_u = (x1 + x2) / 2
        center_v = (y1 + y2) / 2
        
        # Extract depth values from bbox region
        y1_int, y2_int = int(y1), int(y2)
        x1_int, x2_int = int(x1), int(x2)
        
        # Ensure coordinates are within image bounds
        y1_int = max(0, min(y1_int, depth_map.shape[0] - 1))
        y2_int = max(0, min(y2_int, depth_map.shape[0] - 1))
        x1_int = max(0, min(x1_int, depth_map.shape[1] - 1))
        x2_int = max(0, min(x2_int, depth_map.shape[1] - 1))
        
        bbox_depth_region = depth_map[y1_int:y2_int, x1_int:x2_int]
        
        # Calculate mean depth (excluding zeros/invalid depths)
        valid_depths = bbox_depth_region[bbox_depth_region > 0]
        if len(valid_depths) > 0:
            mean_depth = np.median(valid_depths)  # Use median to avoid outliers
        else:
            mean_depth = 0.0
        
        # Convert center to 3D
        center_3d = self.pixel_to_3d(center_u, center_v, mean_depth)
        
        # Convert corners to 3D
        corners_2d = [
            (x1, y1), (x2, y1),  # Top corners
            (x1, y2), (x2, y2)   # Bottom corners
        ]
        
        corners_3d = [
            self.pixel_to_3d(u, v, mean_depth)
            for u, v in corners_2d
        ]
        
        return {
            'center_3d': center_3d,
            'corners_3d': corners_3d,
            'mean_depth': mean_depth,
            'bbox_2d': bbox
        }
    
    def distance_between_points(self, point1, point2):
        """
        Calculate Euclidean distance between two 3D points.
        
        Args:
            point1: (X1, Y1, Z1) tuple
            point2: (X2, Y2, Z2) tuple
        
        Returns:
            Distance in meters
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        return np.linalg.norm(p2 - p1)
    
    def horizontal_distance(self, point1, point2):
        """
        Calculate horizontal (X-Z plane) distance between two 3D points.
        Useful for measuring ground-level distances between workers.
        
        Args:
            point1: (X1, Y1, Z1) tuple
            point2: (X2, Y2, Z2) tuple
        
        Returns:
            Horizontal distance in meters
        """
        dx = point2[0] - point1[0]
        dz = point2[2] - point1[2]
        return np.sqrt(dx**2 + dz**2)
    
    def undistort_point(self, u, v):
        """
        Undistort a single pixel point using camera distortion coefficients.
        
        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
        
        Returns:
            (u_undistorted, v_undistorted) tuple
        """
        point = np.array([[[u, v]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(
            point, self.camera_matrix, self.dist_coeffs, 
            P=self.camera_matrix
        )
        return undistorted[0][0]
    
    def undistort_image(self, image):
        """
        Undistort an entire image using camera distortion coefficients.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Undistorted image
        """
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs, 
            None, new_camera_matrix
        )
        
        # Crop the image
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def get_field_of_view(self):
        """
        Calculate camera field of view.
        
        Returns:
            Dictionary with horizontal and vertical FOV in degrees
        """
        fov_x = 2 * np.arctan(self.image_width / (2 * self.fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(self.image_height / (2 * self.fy)) * 180 / np.pi
        
        return {
            'horizontal_fov_degrees': fov_x,
            'vertical_fov_degrees': fov_y
        }
    
    def create_point_cloud(self, depth_map, rgb_image=None):
        """
        Create a point cloud from a depth map.
        
        Args:
            depth_map: 2D numpy array of depth values in meters
            rgb_image: Optional RGB image for coloring points
        
        Returns:
            Numpy array of shape (N, 3) or (N, 6) with XYZ or XYZRGB
        """
        h, w = depth_map.shape
        
        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert to 3D
        X = (u - self.cx) * depth_map / self.fx
        Y = (v - self.cy) * depth_map / self.fy
        Z = depth_map
        
        # Stack coordinates
        points = np.stack([X, Y, Z], axis=-1)
        points = points.reshape(-1, 3)
        
        # Filter out invalid points
        valid_mask = points[:, 2] > 0
        points = points[valid_mask]
        
        # Add color if RGB image is provided
        if rgb_image is not None:
            colors = rgb_image.reshape(-1, 3)[valid_mask]
            points = np.hstack([points, colors])
        
        return points
    
    def save_point_cloud(self, points, output_path):
        """
        Save point cloud to PLY file.
        
        Args:
            points: Numpy array (N, 3) or (N, 6) with XYZ or XYZRGB
            output_path: Output .ply file path
        """
        has_color = points.shape[1] == 6
        
        with open(output_path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            if has_color:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            f.write("end_header\n")
            
            # Write points
            for point in points:
                if has_color:
                    f.write(f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} "
                           f"{int(point[3])} {int(point[4])} {int(point[5])}\n")
                else:
                    f.write(f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f}\n")


def example_usage():
    """Example demonstrating how to use the converter."""
    
    # Initialize converter
    converter = PixelTo3DConverter('camera_intrinsics.json')
    
    # Example 1: Convert single pixel to 3D
    u, v = 320, 240  # Pixel coordinates
    depth = 5.0      # Depth in meters
    x, y, z = converter.pixel_to_3d(u, v, depth)
    print(f"Pixel ({u}, {v}) at depth {depth}m -> 3D: ({x:.2f}, {y:.2f}, {z:.2f}) meters")
    
    # Example 2: Distance between two workers
    worker1_pixel = (320, 240)
    worker1_depth = 5.0
    
    worker2_pixel = (450, 280)
    worker2_depth = 5.5
    
    worker1_3d = converter.pixel_to_3d(*worker1_pixel, worker1_depth)
    worker2_3d = converter.pixel_to_3d(*worker2_pixel, worker2_depth)
    
    distance = converter.distance_between_points(worker1_3d, worker2_3d)
    h_distance = converter.horizontal_distance(worker1_3d, worker2_3d)
    
    print(f"\nWorker 1 at: {worker1_3d}")
    print(f"Worker 2 at: {worker2_3d}")
    print(f"3D Distance: {distance:.2f} meters")
    print(f"Horizontal Distance: {h_distance:.2f} meters")
    
    # Example 3: Camera field of view
    fov = converter.get_field_of_view()
    print(f"\nCamera Field of View:")
    print(f"  Horizontal: {fov['horizontal_fov_degrees']:.1f}°")
    print(f"  Vertical: {fov['vertical_fov_degrees']:.1f}°")


if __name__ == '__main__':
    # Check if intrinsics file exists
    if Path('camera_intrinsics.json').exists():
        example_usage()
    else:
        print("❌ camera_intrinsics.json not found!")
        print("Please run calibrate_camera.py first to generate calibration data.")
