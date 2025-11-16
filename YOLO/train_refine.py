#!/usr/bin/env python3
"""
Fine-tune YOLO model on construction site dataset.

This script:
1. Loads your pre-trained best.pt model
2. Fine-tunes it on the 1_200_proper_dataset
3. Saves results to runs/train/construction_finetune
"""
from ultralytics import YOLO

def main():
    print("="*70)
    print("YOLO FINE-TUNING - Construction Site Detection")
    print("="*70)
    
    # Load pre-trained model
    model = YOLO('best.pt')
    print("\n✓ Loaded pre-trained model: best.pt")
    
    # Training parameters
    print("\nTraining Configuration:")
    print("  Dataset: construction_data.yaml")
    print("  Epochs: 50 (fine-tuning)")
    print("  Image size: 1024x1024")
    print("  Batch size: 8")
    print("  Device: CPU")
    print("  Train images: 160")
    print("  Val images: 40")
    
    # Start training
    print("\n" + "="*70)
    print("Starting fine-tuning...")
    print("="*70 + "\n")
    
    results = model.train(
        data='construction_data.yaml',
        epochs=50,
        imgsz=1024,
        batch=8,
        device='cpu',
        project='runs/train',
        name='construction_finetune',
        patience=10,  # Early stopping
        save=True,
        plots=True,
        val=True,
        cache=False,  # Don't cache on CPU to save RAM
        workers=4,
        optimizer='Adam',
        lr0=0.001,  # Lower learning rate for fine-tuning
        warmup_epochs=3,
        amp=False  # Disable AMP on CPU
    )
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nResults saved to: runs/train/construction_finetune")
    print(f"Best model: runs/train/construction_finetune/weights/best.pt")
    print(f"Last model: runs/train/construction_finetune/weights/last.pt")
    
    # Validation
    print("\n" + "="*70)
    print("Running validation on best model...")
    print("="*70 + "\n")
    
    best_model = YOLO('runs/train/construction_finetune/weights/best.pt')
    metrics = best_model.val()
    
    print("\n" + "="*70)
    print("VALIDATION METRICS")
    print("="*70)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall: {metrics.box.r.mean():.4f}")
    print("="*70)

if __name__ == "__main__":
    main()