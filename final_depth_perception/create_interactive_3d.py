#!/usr/bin/env python3
"""
Create interactive 3D depth visualization for presentation
Uses Plotly for full interactivity (rotate, zoom, pan)
"""

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def create_interactive_depth_viz():
    print("="*80)
    print("  CREATING INTERACTIVE 3D DEPTH VISUALIZATION")
    print("="*80)
    
    # Load image
    img_path = "20241210_074500.jpg"
    print(f"\n📷 Loading image: {img_path}")
    img = cv2.imread(img_path)
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
    
    # Load or generate depth map
    print(f"\n🌊 Generating depth map...")
    import tensorflow as tf
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path="depth_anything_v2.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    
    img_resized = cv2.resize(img_rgb, (input_width, input_height))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    depth_map = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Resize depth to original image size
    depth_map = cv2.resize(depth_map, (width, height))
    
    # Normalize depth
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    print(f"   Depth range: {depth_normalized.min():.4f} - {depth_normalized.max():.4f}")
    print(f"   Mean depth: {depth_normalized.mean():.4f}")
    
    # Downsample for performance (every 10th pixel)
    print(f"\n⚡ Downsampling for performance...")
    step = 10
    y_coords = np.arange(0, height, step)
    x_coords = np.arange(0, width, step)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = depth_normalized[::step, ::step]
    
    # Get colors from original image
    colors = img_rgb[::step, ::step]
    
    print(f"   3D surface: {X.shape[0]}x{X.shape[1]} points")
    
    # Create interactive 3D plot
    print(f"\n🎨 Creating interactive visualization...")
    
    # Create figure with multiple views
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('Perspective View', 'Top-Down View',
                       'Side View', 'Front View'),
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )
    
    # Prepare surface color from RGB image
    surfacecolor = np.zeros((Z.shape[0], Z.shape[1]))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            r, g, b = colors[i, j]
            # Convert to grayscale for surface coloring
            surfacecolor[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Create surface trace
    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=surfacecolor,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Depth", len=0.45, y=0.75),
        lighting=dict(
            ambient=0.4,
            diffuse=0.8,
            specular=0.3,
            roughness=0.5
        ),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Depth: %{z:.3f}<extra></extra>'
    )
    
    # Add to all subplots
    for i in range(1, 5):
        fig.add_trace(surface, row=(i-1)//2 + 1, col=(i-1)%2 + 1)
    
    # Camera views for each subplot
    camera_views = [
        # Perspective (top-left)
        dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
        # Top-down (top-right)
        dict(
            eye=dict(x=0, y=0, z=2.5),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)
        ),
        # Side view (bottom-left)
        dict(
            eye=dict(x=2.5, y=0, z=0.5),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
        # Front view (bottom-right)
        dict(
            eye=dict(x=0, y=2.5, z=0.5),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
    ]
    
    # Update layout for each scene
    for i in range(1, 5):
        scene_name = f'scene{i}' if i > 1 else 'scene'
        fig.update_layout({
            scene_name: dict(
                xaxis=dict(title='X (pixels)', showgrid=True, gridcolor='lightgray'),
                yaxis=dict(title='Y (pixels)', showgrid=True, gridcolor='lightgray'),
                zaxis=dict(title='Depth (normalized)', showgrid=True, gridcolor='lightgray'),
                camera=camera_views[i-1],
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            )
        })
    
    # Update overall layout
    fig.update_layout(
        title=dict(
            text="<b>Interactive 3D Depth Map - Construction Site</b><br>" +
                 f"<sub>Image: {img_path} | Resolution: {width}x{height} | " +
                 f"Camera: fx={fx:.1f}px, fy={fy:.1f}px</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        height=1000,
        showlegend=False,
        margin=dict(l=0, r=0, t=100, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Save as interactive HTML
    output_path = "interactive_depth_3d.html"
    fig.write_html(
        output_path,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['toImage'],
            'modeBarButtonsToAdd': ['hoverclosest', 'hovercompare']
        }
    )
    
    print(f"   ✓ Saved: {output_path}")
    
    # Also create a single high-quality view
    print(f"\n🎨 Creating single high-quality view...")
    
    fig_single = go.Figure(data=[surface])
    
    fig_single.update_layout(
        title=dict(
            text="<b>3D Depth Surface - Construction Site</b><br>" +
                 f"<sub>Drag to rotate • Scroll to zoom • Double-click to reset</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=24)
        ),
        scene=dict(
            xaxis=dict(title='X Position (pixels)', showgrid=True, gridcolor='lightgray'),
            yaxis=dict(title='Y Position (pixels)', showgrid=True, gridcolor='lightgray'),
            zaxis=dict(title='Depth Value (normalized)', showgrid=True, gridcolor='lightgray'),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4),
            bgcolor='rgba(240,240,240,0.9)'
        ),
        height=800,
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    output_single = "interactive_depth_3d_single.html"
    fig_single.write_html(
        output_single,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'depth_3d_view',
                'height': 1080,
                'width': 1920,
                'scale': 2
            }
        }
    )
    
    print(f"   ✓ Saved: {output_single}")
    
    # Print instructions
    print("\n" + "="*80)
    print("✅ INTERACTIVE 3D VISUALIZATIONS CREATED!")
    print("="*80)
    print(f"\n📁 Generated files:")
    print(f"   1. {output_path}")
    print(f"      → 4-panel view (Perspective, Top, Side, Front)")
    print(f"   2. {output_single}")
    print(f"      → Single high-quality view (best for presentations)")
    
    print(f"\n🌐 How to use:")
    print(f"   1. Upload HTML file to Netlify Drop (https://app.netlify.com/drop)")
    print(f"   2. Drag & drop the HTML file")
    print(f"   3. Get instant link (e.g., https://xyz.netlify.app)")
    print(f"   4. Shorten with bit.ly if needed")
    print(f"   5. Embed in Google Slides: Insert → Embed → Paste URL")
    
    print(f"\n💡 Interactive controls:")
    print(f"   • Click & drag to rotate")
    print(f"   • Scroll to zoom in/out")
    print(f"   • Double-click to reset view")
    print(f"   • Hover over surface to see coordinates & depth")
    
    print(f"\n📊 Depth statistics:")
    print(f"   • Min depth: {depth_normalized.min():.4f}")
    print(f"   • Max depth: {depth_normalized.max():.4f}")
    print(f"   • Mean depth: {depth_normalized.mean():.4f}")
    print(f"   • Std dev: {depth_normalized.std():.4f}")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    create_interactive_depth_viz()
