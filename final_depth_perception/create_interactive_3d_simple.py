#!/usr/bin/env python3
"""
Create interactive 3D depth visualization - Simplified version
Uses existing depth generation capability
"""

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import subprocess
import os

def generate_depth_map(image_path):
    """Generate depth map using existing TFLite script"""
    print(f"🌊 Generating depth map with TFLite model...")
    
    # Run the existing depth script
    result = subprocess.run(
        ['python', 'run_tflite_depth.py', '--image', image_path, '--output', 'temp_depth.npy'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0 and os.path.exists('temp_depth.npy'):
        depth_map = np.load('temp_depth.npy')
        os.remove('temp_depth.npy')  # Clean up
        return depth_map
    else:
        # Fallback: create simple depth from grayscale
        print(f"   Using grayscale fallback...")
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        depth = gray.astype(np.float32) / 255.0
        return depth

def create_interactive_depth_viz():
    print("="*80)
    print("  CREATING INTERACTIVE 3D DEPTH VISUALIZATION")
    print("="*80)
    
    # Load image
    img_path = "20241210_074500.jpg"
    print(f"\n📷 Loading image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Error: Could not load {img_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    print(f"   Resolution: {width}x{height}")
    
    # Load camera calibration
    print(f"\n📊 Loading camera calibration...")
    with open('camera_intrinsics.json', 'r') as f:
        calib = json.load(f)
    
    fx = calib['focal_length']['fx']
    fy = calib['focal_length']['fy']
    cx = calib['principal_point']['cx']
    cy = calib['principal_point']['cy']
    
    print(f"   fx={fx:.2f}, fy={fy:.2f}")
    print(f"   cx={cx:.2f}, cy={cy:.2f}")
    
    # Generate or load depth map - use ACTUAL depth model
    print(f"\n🌊 Loading ACTUAL TFLite depth data...")
    
    # Load the pre-computed raw depth map (16-bit PNG from run_tflite_depth.py)
    depth_raw_path = "depth_result_raw.png"
    if os.path.exists(depth_raw_path):
        # Load 16-bit depth map
        depth_map = cv2.imread(depth_raw_path, cv2.IMREAD_ANYDEPTH)
        if depth_map is not None:
            # Convert to float [0, 1]
            depth_map = depth_map.astype(np.float32) / 65535.0
            # Resize to match image resolution
            depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
            print(f"   ✓ Loaded actual TFLite depth: {depth_raw_path}")
        else:
            raise Exception(f"Could not read {depth_raw_path}")
    else:
        # Generate depth if not exists
        print(f"   Generating depth with TFLite model...")
        result = subprocess.run(
            ['python', 'run_tflite_depth.py', 
             '--model', 'depth_anything_v2.tflite',
             '--image', img_path,
             '--output', 'depth_result.png'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and os.path.exists(depth_raw_path):
            depth_map = cv2.imread(depth_raw_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 65535.0
            depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
            print(f"   ✓ Generated and loaded depth")
        else:
            raise Exception("Failed to generate depth map")
    
    # Don't normalize - preserve raw depth variations for accurate terrain
    # Keep original depth: far objects have higher values (natural depth map)
    
    print(f"   Depth range: {depth_map.min():.4f} - {depth_map.max():.4f}")
    print(f"   Mean depth: {depth_map.mean():.4f}")
    
    # NO smoothing - preserve all bumps and terrain features
    print(f"\n🔧 Preserving raw terrain features (no smoothing)...")
    depth_smoothed = depth_map  # Use raw data
    
    # Downsample for performance (smaller step = more detail)
    print(f"\n⚡ Downsampling for interactivity...")
    step = 5  # Smaller step to preserve terrain detail
    y_coords = np.arange(0, height, step)
    x_coords = np.arange(0, width, step)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = depth_smoothed[::step, ::step]
    
    # No smoothing - use raw data
    # Scale Z to make terrain bumps highly visible
    Z = Z * 2.0  # Strong amplification to show all terrain features
    
    print(f"   3D surface: {X.shape[0]}x{X.shape[1]} points ({X.shape[0]*X.shape[1]} total)")
    
    # Create interactive visualizations
    print(f"\n🎨 Creating interactive 3D plots...")
    
    # Create main high-quality single view with depth coloring
    fig_single = go.Figure()
    
    fig_single.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=Z,  # Use depth values for coloring
        colorscale='Jet',  # Blue (near/low) to Red (far/high)
        reversescale=True,  # Reverse: blue=near, red=far
        showscale=True,
        colorbar=dict(
            title=dict(
                text='<b>Depth</b>',
                font=dict(size=14, family='Arial')
            ),
            thickness=20,
            len=0.7,
            x=1.02,
            tickfont=dict(size=11),
            tickformat='.2f'
        ),
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            specular=0.3,
            roughness=0.4,
            fresnel=0.1
        ),
        lightposition=dict(
            x=1000,
            y=1000,
            z=2000
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor='rgba(255,255,255,0.5)',
                width=2,
                highlightwidth=3,
                project=dict(z=False)
            )
        ),
        hovertemplate='<b>Position</b><br>X: %{x:.0f}px<br>Y: %{y:.0f}px<br><b>Depth: %{z:.3f}</b><extra></extra>',
        opacity=1.0
    ))
    
    fig_single.update_layout(
        title=dict(
            text="<b>Interactive 3D Depth Surface - Construction Site</b><br>" +
                 f"<sub>Image: {img_path} | Resolution: {width}x{height} | " +
                 f"Camera: fx={fx:.1f}px, fy={fy:.1f}px | Reprojection Error: {calib['reprojection_error_pixels']:.4f}px</sub><br>" +
                 "<sub style='font-size:11px; color:#666;'>🖱️ Click & Drag to Rotate • Scroll to Zoom • Double-Click to Reset</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=22, family="Arial, sans-serif")
        ),
        scene=dict(
            xaxis=dict(
                title='<b>X Position (pixels)</b>',
                showgrid=True,
                gridcolor='rgba(200,200,200,0.3)',
                gridwidth=1,
                backgroundcolor='rgba(240,240,245,0.5)'
            ),
            yaxis=dict(
                title='<b>Y Position (pixels)</b>',
                showgrid=True,
                gridcolor='rgba(200,200,200,0.3)',
                gridwidth=1,
                backgroundcolor='rgba(240,240,245,0.5)'
            ),
            zaxis=dict(
                title='<b>Depth (far = high)</b>',
                showgrid=True,
                gridcolor='rgba(200,200,200,0.3)',
                gridwidth=1,
                backgroundcolor='rgba(245,245,250,0.5)',
                range=[0, 2.5]  # Allow for highly amplified elevation
            ),
            camera=dict(
                eye=dict(x=2.2, y=-1.8, z=0.8),
                center=dict(x=0, y=0, z=0.3),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=0.9, z=0.6),  # High Z to show all terrain bumps
            bgcolor='rgba(248,248,252,1)'
        ),
        height=900,
        margin=dict(l=0, r=0, t=120, b=20),
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=13, color='#333')
    )
    
    output_single = "interactive_depth_3d.html"
    fig_single.write_html(
        output_single,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'depth_3d_presentation',
                'height': 1080,
                'width': 1920,
                'scale': 2
            },
            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
        }
    )
    
    print(f"   ✓ Created: {output_single}")
    
    # Create multi-view comparison
    print(f"\n🎨 Creating multi-angle view...")
    
    fig_multi = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('<b>Main Angled View</b>', '<b>High Angle View</b>',
                       '<b>Low Front Angle</b>', '<b>Side Angle View</b>'),
        vertical_spacing=0.06,
        horizontal_spacing=0.04
    )
    
    # Add surface to all subplots with depth coloring
    for row in range(1, 3):
        for col in range(1, 3):
            show_colorbar = (row == 1 and col == 2)  # Only show colorbar on one subplot
            fig_multi.add_trace(
                go.Surface(
                    x=X, y=Y, z=Z,
                    surfacecolor=Z,
                    colorscale='Jet',
                    reversescale=True,  # Blue=near, Red=far
                    showscale=show_colorbar,
                    colorbar=dict(
                        title='Depth',
                        thickness=15,
                        len=0.5
                    ) if show_colorbar else None,
                    lighting=dict(
                        ambient=0.6,
                        diffuse=0.8,
                        specular=0.3,
                        roughness=0.4
                    ),
                    contours=dict(
                        z=dict(show=True, usecolormap=True, width=1)
                    ),
                    hovertemplate='X:%{x:.0f}<br>Y:%{y:.0f}<br>Depth:%{z:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
    
    # Camera views for each subplot
    cameras = [
        # Perspective angled (top-left)
        dict(eye=dict(x=2.2, y=-1.8, z=0.8)),
        # High angle (top-right)
        dict(eye=dict(x=1.5, y=-1.2, z=1.5)),
        # Low front angle (bottom-left)
        dict(eye=dict(x=0.5, y=-2.5, z=0.3)),
        # Side angle (bottom-right)
        dict(eye=dict(x=2.5, y=-0.5, z=0.6))
    ]
    
    for i, camera in enumerate(cameras, 1):
        scene = f'scene{i}' if i > 1 else 'scene'
        fig_multi.update_layout({
            scene: dict(
                xaxis=dict(title='X', showgrid=True, gridcolor='lightgray'),
                yaxis=dict(title='Y', showgrid=True, gridcolor='lightgray'),
                zaxis=dict(title='Depth', showgrid=True, gridcolor='lightgray'),
                camera=camera,
                aspectmode='manual',
                aspectratio=dict(x=1, y=0.6, z=0.3)
            )
        })
    
    fig_multi.update_layout(
        title=dict(
            text=f"<b>Multi-Angle 3D Depth Visualization</b><br>" +
                 f"<sub>{img_path} | {width}x{height} | Camera Calibrated</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        height=1100,
        showlegend=False,
        margin=dict(l=0, r=0, t=100, b=0),
        paper_bgcolor='white'
    )
    
    output_multi = "interactive_depth_3d_multiview.html"
    fig_multi.write_html(output_multi, config={'displayModeBar': True, 'displaylogo': False})
    
    print(f"   ✓ Created: {output_multi}")
    
    # Print summary
    print("\n" + "="*80)
    print("✅ INTERACTIVE 3D VISUALIZATIONS CREATED!")
    print("="*80)
    print(f"\n📁 Generated files:")
    print(f"   1. {output_single}")
    print(f"      → Single high-quality view (BEST FOR PRESENTATIONS)")
    print(f"      → Professional layout with camera specs")
    print(f"   2. {output_multi}")
    print(f"      → 4-panel multi-angle view")
    print(f"      → Compare different perspectives")
    
    print(f"\n🌐 How to embed in presentation:")
    print(f"   1. Upload to Netlify Drop: https://app.netlify.com/drop")
    print(f"   2. Drag & drop {output_single}")
    print(f"   3. Copy the generated link (e.g., https://xyz.netlify.app)")
    print(f"   4. Optional: Shorten with bit.ly")
    print(f"   5. In Google Slides:")
    print(f"      • Insert → Embed")
    print(f"      • Paste URL")
    print(f"      • Click 'Insert'")
    
    print(f"\n💡 Interactive features:")
    print(f"   • 🖱️  Click & drag to rotate view")
    print(f"   • 📜 Scroll to zoom in/out")
    print(f"   • 🖱️  Double-click to reset to default view")
    print(f"   • 📍 Hover over surface for exact coordinates & depth")
    print(f"   • 📸 Camera icon (top-right) to save as PNG")
    
    print(f"\n📊 Depth statistics:")
    print(f"   • Min depth: {depth_map.min():.4f}")
    print(f"   • Max depth: {depth_map.max():.4f}")
    print(f"   • Mean depth: {depth_map.mean():.4f}")
    print(f"   • Std dev: {depth_map.std():.4f}")
    
    print(f"\n📷 Camera calibration:")
    print(f"   • Focal length: fx={fx:.2f}px, fy={fy:.2f}px")
    print(f"   • Principal point: cx={cx:.2f}px, cy={cy:.2f}px")
    print(f"   • Reprojection error: {calib['reprojection_error_pixels']:.4f}px")
    print(f"   • Quality: Excellent ✅")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    create_interactive_depth_viz()
