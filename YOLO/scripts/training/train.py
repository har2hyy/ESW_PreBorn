#!/usr/bin/env python3
"""
YOLO Training Script - Construction Site Worker Safety
Training from scratch with 300 manually annotated images

Optimized for:
- Small dataset (300 images)
- High-quality manual annotations
- Construction site worker detection
- CPU/GPU training
"""
from ultralytics import YOLO
import torch

def main():
    print("="*80)
    print("  YOLO11 TRAINING - Construction Site Worker Safety (300 Images)")
    print("="*80)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Training Device: {device.upper()}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load pre-trained YOLO11n model
    model = YOLO('yolo11n.pt')
    print("\n✅ Loaded pre-trained YOLO11n model")
    
    # Training configuration
    print("\n📋 Training Configuration:")
    print("   Dataset: 300 manually annotated images")
    print("   Epochs: 200 (with early stopping)")
    print("   Image size: 1024x1024")
    print("   Batch size: 4 (optimized for 7.6GB VRAM)")
    print("   Patience: 50 epochs (early stopping)")
    print("   Data augmentation: Enhanced")
    
    print("\n" + "="*80)
    print("🚀 Starting Training...")
    print("="*80 + "\n")
    
    # Train the model with optimized parameters for 300 images
    results = model.train(
        # Dataset
        data='construction_data.yaml',
        
        # Training duration
        epochs=200,                    # More epochs for small dataset
        patience=50,                   # Early stopping after 50 epochs without improvement
        
        # Image settings
        imgsz=1024,                    # High resolution for detailed worker detection
        
        # Batch settings
        batch=4,                       # Reduced from 16 to prevent OOM errors
        
        # Device
        device=device,
        
        # Optimizer settings
        optimizer='AdamW',             # AdamW is better for small datasets
        lr0=0.01,                      # Initial learning rate
        lrf=0.01,                      # Final learning rate (lr0 * lrf)
        momentum=0.937,                # SGD momentum
        weight_decay=0.0005,           # Weight decay for regularization
        warmup_epochs=5,               # Warmup for stable start
        warmup_momentum=0.8,           # Initial warmup momentum
        warmup_bias_lr=0.1,            # Warmup initial bias lr
        
        # Data Augmentation (Enhanced for small dataset)
        hsv_h=0.015,                   # HSV-Hue augmentation
        hsv_s=0.7,                     # HSV-Saturation augmentation
        hsv_v=0.4,                     # HSV-Value augmentation
        degrees=10.0,                  # Rotation (+/- deg)
        translate=0.1,                 # Translation (+/- fraction)
        scale=0.5,                     # Scale (+/- gain)
        shear=2.0,                     # Shear (+/- deg)
        perspective=0.0001,            # Perspective distortion
        flipud=0.0,                    # Flip up-down probability
        fliplr=0.5,                    # Flip left-right probability
        mosaic=1.0,                    # Mosaic augmentation probability
        mixup=0.1,                     # MixUp augmentation probability
        copy_paste=0.1,                # Copy-paste augmentation probability
        
        # Regularization
        dropout=0.0,                   # Dropout probability
        
        # Loss weights
        box=7.5,                       # Box loss weight
        cls=0.5,                       # Class loss weight
        dfl=1.5,                       # DFL loss weight
        
        # Validation
        val=True,                      # Validate during training
        
        # Saving
        save=True,                     # Save checkpoints
        save_period=10,                # Save checkpoint every 10 epochs
        
        # Plotting
        plots=True,                    # Generate training plots
        
        # Performance
        workers=4,                     # Reduced workers to save memory
        cache=False,                   # Disable caching to save VRAM
        
        # Multi-scale training
        rect=False,                    # Rectangular training (faster but less augmentation)
        cos_lr=True,                   # Cosine learning rate scheduler
        close_mosaic=10,               # Disable mosaic in last N epochs
        
        # AMP (Automatic Mixed Precision)
        amp=True if device == 'cuda' else False,  # GPU only
        
        # Project organization
        project='runs/train',
        name='construction_300_images',
        exist_ok=False,                # Don't overwrite existing
        
        # Miscellaneous
        deterministic=False,           # Ensure deterministic results
        single_cls=False,              # Train as multi-class (workers, equipment, etc.)
        verbose=True,                  # Verbose output
    )
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    
    # Results summary
    print(f"\n📁 Results Location:")
    print(f"   Training results: runs/train/construction_300_images/")
    print(f"   Best weights: runs/train/construction_300_images/weights/best.pt")
    print(f"   Last weights: runs/train/construction_300_images/weights/last.pt")
    
    print(f"\n📊 Generated Files:")
    print(f"   ✓ Training curves (losses, metrics)")
    print(f"   ✓ Confusion matrix")
    print(f"   ✓ Precision-Recall curves")
    print(f"   ✓ F1-Confidence curves")
    print(f"   ✓ Prediction examples")
    
    # Run validation on best model
    print("\n" + "="*80)
    print("🔍 Running Final Validation on Best Model...")
    print("="*80 + "\n")
    
    best_model = YOLO('runs/train/construction_300_images/weights/best.pt')
    metrics = best_model.val()
    
    # Print metrics
    print("\n" + "="*80)
    print("📈 FINAL VALIDATION METRICS")
    print("="*80)
    print(f"   mAP50:        {metrics.box.map50:.4f}  (Detection at IoU=0.5)")
    print(f"   mAP50-95:     {metrics.box.map:.4f}  (Detection at IoU=0.5:0.95)")
    print(f"   Precision:    {metrics.box.mp:.4f}  (True positives / All detections)")
    print(f"   Recall:       {metrics.box.mr:.4f}  (True positives / All ground truth)")
    print(f"   F1-Score:     {2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6):.4f}")
    print("="*80)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"   Use best.pt for inference and deployment")
    print(f"   Check plots in runs/train/construction_300_images/ for detailed analysis")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()