"""
Step 4: Train UI Component Detection Model (YOLOv8 Nano)
-------------------------------------------------------
Trains YOLOv8 Nano on the verified, clean sketch UI dataset:
- Zero empty images
- 19,273 labeled UI component boxes
- mosaic=0.0 (Preserves document UI structure)
- rect=True (Preserves natural width-to-height aspect ratios)
- RAM Caching enabled for fast training on CPU/GPU
- Saves weights to data/prepared/models/drawx_detector/weights/best.pt
"""

import os
import glob
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 65)
    print(" STEP 4: TRAINING DRAWX UI DETECTION MODEL ")
    print("=" * 65)

    from ultralytics import YOLO
    import torch

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Acceleration: {'GPU (CUDA)' if device == '0' else 'CPU'}")

    yaml_path = os.path.join(BASE_DIR, "data", "prepared", "yolo", "dataset.yaml")
    print(f"Dataset Configuration: {yaml_path}")

    # Initialize YOLOv8 Nano
    weights_path = os.path.join(BASE_DIR, "yolov8n.pt")
    if not os.path.exists(weights_path):
        weights_path = "yolov8n.pt"
    print(f"Loading weights from: {weights_path}")
    model = YOLO(weights_path)

    project_dir = os.path.join(BASE_DIR, "data", "prepared", "models")
    os.makedirs(project_dir, exist_ok=True)

    print("\nStarting training...")
    print("Settings: epochs=25, imgsz=416, batch=16, rect=True, mosaic=0.0")
    print("-" * 65)

    results = model.train(
        data=yaml_path,
        epochs=25,
        imgsz=416,
        batch=16,
        device=device,
        project=project_dir,
        name="drawx_detector",
        exist_ok=True,
        cache=True,
        rect=True,          # Preserves natural UI aspect ratios
        mosaic=0.0,         # Disables mosaic tiling for clean UI documents
        degrees=5.0,
        hsv_v=0.3,
        scale=0.1,
        fliplr=0.0,
        workers=0,
        save=True,
        verbose=True
    )

    print("\n" + "=" * 65)
    print(" TRAINING COMPLETE! ")
    print("=" * 65)

    best_weights = os.path.join(project_dir, "drawx_detector", "weights", "best.pt")
    print(f"Best Model Saved to: {best_weights}")

    # Evaluate on Test Set
    print("\nEvaluating on Test Set (106 sketches)...")
    eval_model = YOLO(best_weights)
    metrics = eval_model.val(
        data=yaml_path,
        split="test",
        project=project_dir,
        name="drawx_eval",
        exist_ok=True
    )

    print(f"\nTest Set Metrics:")
    print(f"  - Precision (Mean):  {metrics.box.mp * 100:.1f}%")
    print(f"  - Recall (Mean):     {metrics.box.mr * 100:.1f}%")
    print(f"  - mAP50:             {metrics.box.map50 * 100:.1f}%")
    print(f"  - mAP50-95:          {metrics.box.map * 100:.1f}%")

    # Generate test predictions
    test_imgs_dir = os.path.join(BASE_DIR, "data", "prepared", "yolo", "images", "test")
    preds_dir = os.path.join(BASE_DIR, "data", "prepared", "test_predictions")
    os.makedirs(preds_dir, exist_ok=True)

    test_samples = sorted(glob.glob(os.path.join(test_imgs_dir, "*.png")))[:10]
    for p in test_samples:
        res = eval_model.predict(source=p, conf=0.15, save=False, verbose=False)[0]
        im_arr = res.plot()
        im = Image.fromarray(im_arr[..., ::-1])
        im.save(os.path.join(preds_dir, f"pred_{os.path.basename(p)}"))

    print(f"Visual predictions saved to: {preds_dir}")
    print("=" * 65)

if __name__ == "__main__":
    main()
