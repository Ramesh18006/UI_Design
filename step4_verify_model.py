"""
Step 4 Verification: Check and Test the Trained Model
----------------------------------------------------
This script lets you check if the model is trained well:
1. --eval: Runs full test set evaluation and shows Precision, Recall, and mAP50 for each class.
2. Default / --image: Runs live detection on a sketch and lists every detected component + confidence score.
3. Saves the visual prediction image with colored bounding boxes.

Usage examples:
  python step4_verify_model.py
  python step4_verify_model.py --eval
  python step4_verify_model.py --image data/prepared/yolo/images/test/1009_0.png
  python step4_verify_model.py --image path/to/student_phone_sketch.jpg --conf 0.20
"""

import os
import sys
import argparse
import glob
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="Verify DrawX UI Detection Model")
    parser.add_argument("--image", default=None, help="Path to any sketch image to test")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (default: 0.20)")
    parser.add_argument("--eval", action="store_true", help="Run full evaluation on the unseen Test Set (106 images)")
    args = parser.parse_args()

    from ultralytics import YOLO

    weights_path = os.path.join(BASE_DIR, "data", "prepared", "models", "drawx_detector", "weights", "best.pt")
    if not os.path.exists(weights_path):
        print("=" * 65)
        print(" ERROR: Model weights file not found!")
        print("=" * 65)
        print(f"Looked for: {weights_path}")
        print("Please train the model first by running:")
        print("  python step4_train_model.py")
        print("=" * 65)
        return

    print("=" * 65)
    print(" VERIFYING DRAWX UI DETECTION MODEL ")
    print("=" * 65)
    print(f"Loaded weights: {weights_path}")
    model = YOLO(weights_path)

    # Mode 1: Full Test Set Evaluation
    if args.eval:
        yaml_path = os.path.join(BASE_DIR, "data", "prepared", "yolo", "dataset.yaml")
        print(f"\nEvaluating on Unseen Test Set (106 sketches) using: {yaml_path}")
        print("-" * 65)
        metrics = model.val(data=yaml_path, split="test", verbose=True)

        print("\n" + "=" * 65)
        print(" OVERALL TEST METRICS:")
        print("=" * 65)
        print(f"  - Precision (Mean Accuracy):       {metrics.box.mp * 100:.1f}%")
        print(f"  - Recall (Detection Rate):         {metrics.box.mr * 100:.1f}%")
        print(f"  - mAP50 (Mean AP @ 50% IoU):       {metrics.box.map50 * 100:.1f}%")
        print(f"  - mAP50-95 (Strict Overlap Score): {metrics.box.map * 100:.1f}%")

        print("\nPER-CLASS ACCURACY BREAKDOWN:")
        print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10}")
        print("-" * 48)
        for i, name in model.names.items():
            p_val = metrics.box.p[i] * 100 if i < len(metrics.box.p) else 0.0
            r_val = metrics.box.r[i] * 100 if i < len(metrics.box.r) else 0.0
            map_val = metrics.box.ap50[i] * 100 if i < len(metrics.box.ap50) else 0.0
            print(f"{name:<12} | {p_val:>8.1f}% | {r_val:>8.1f}% | {map_val:>8.1f}%")
        print("=" * 65)
        return

    # Mode 2: Live Inference on a Sketch Image
    if args.image:
        test_img = args.image
    else:
        test_images = sorted(glob.glob(os.path.join(BASE_DIR, "data", "prepared", "yolo", "images", "test", "*.png")))
        if not test_images:
            print("No test images found in data/prepared/yolo/images/test/")
            return
        test_img = test_images[0]

    if not os.path.exists(test_img):
        print(f"Error: Image not found -> {test_img}")
        return

    print(f"\nRunning detection on: {test_img}")
    print(f"Confidence threshold: {args.conf * 100:.0f}%")
    print("-" * 65)

    results = model.predict(source=test_img, conf=args.conf, verbose=False)[0]
    boxes = results.boxes

    print(f"Total UI Components Detected: {len(boxes)}\n")
    if len(boxes) == 0:
        print("  No components detected at this threshold.")
        print("  Tip: Try lowering the threshold: python step4_verify_model.py --conf 0.10")
    else:
        print(f"{'#':<3} | {'Component':<12} | {'Confidence':<12} | {'Bounding Box [x1, y1, x2, y2]'}")
        print("-" * 65)
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            print(f"[{i+1:>2}] {cls_name.upper():<12} | {conf * 100:>9.1f}% | {xyxy}")

    # Save visualization
    out_dir = os.path.join(BASE_DIR, "data", "prepared", "verification_output")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"verified_{os.path.basename(test_img)}")

    im_arr = results.plot()
    Image.fromarray(im_arr[..., ::-1]).save(out_file)
    print("\n" + "-" * 65)
    print(f"Visual prediction image saved to:\n  -> {out_file}")
    print("=" * 65)

if __name__ == "__main__":
    main()
