"""
Step 3: YOLO Dataset Formatting & Train/Val/Test Splitting
---------------------------------------------------------
This script prepares the annotated dataset so it is 100% ready for training
an object detection model (like YOLOv8/YOLOv11).

What it does:
1. Groups sketches by webpage design ID so that sketches of the same page
   never leak across training and test splits.
2. Splits data into:
   - 70% Training   (The study material)
   - 15% Validation (Practice exams during training)
   - 15% Testing    (Final real-world exam)
3. Converts all bounding boxes into YOLO format:
   - Class ID (0 to 4)
   - Normalized coordinates: [x_center, y_center, width, height] between 0.0 and 1.0
4. Generates standard YOLO folders:
   data/prepared/yolo/images/{train,val,test}
   data/prepared/yolo/labels/{train,val,test}
5. Creates the master dataset.yaml configuration file.
"""

import os
import shutil
import random
import json
from collections import Counter

RAW_DATASET_DIR = os.path.join("data", "raw", "sketch2code_dataset_v1_cleaned")
ANNOTATIONS_FILE = os.path.join("data", "annotations", "annotations.json")
YOLO_DIR = os.path.join("data", "prepared", "yolo")

CLASSES = ["button", "heading", "text", "image", "input"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def main():
    print("=" * 65)
    print(" STEP 3: PREPARING YOLO DATASET & SPLITTING ")
    print("=" * 65)

    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Error: {ANNOTATIONS_FILE} not found! Please run Step 2 first.")
        return

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    # 1. Group sketches by webpage ID to prevent data leakage
    pages_to_sketches = {}
    for sketch_name, sample in master_data.items():
        page_id = sample["page_id"]
        if page_id not in pages_to_sketches:
            pages_to_sketches[page_id] = []
        pages_to_sketches[page_id].append(sketch_name)

    all_page_ids = sorted(list(pages_to_sketches.keys()))
    print(f"Total Unique Webpage Designs: {len(all_page_ids)}")
    print(f"Total Sketch Images:          {len(master_data)}")

    # Set random seed for reproducible splits
    random.seed(42)
    random.shuffle(all_page_ids)

    n_total = len(all_page_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    train_pages = set(all_page_ids[:n_train])
    val_pages = set(all_page_ids[n_train:n_train + n_val])
    test_pages = set(all_page_ids[n_train + n_val:])

    print(f"\nWebpage Split Distribution:")
    print(f"  - Train: {len(train_pages)} pages ({TRAIN_RATIO * 100:.0f}%)")
    print(f"  - Val:   {len(val_pages)} pages ({VAL_RATIO * 100:.0f}%)")
    print(f"  - Test:  {len(test_pages)} pages ({TEST_RATIO * 100:.0f}%)")

    # 2. Setup destination folders
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(YOLO_DIR, "images", s), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, "labels", s), exist_ok=True)

    split_image_counts = Counter()
    split_box_counts = {s: Counter() for s in splits}

    print("\nFormatting images and YOLO labels...")
    for sketch_name, sample in master_data.items():
        page_id = sample["page_id"]
        img_w, img_h = sample["image_size"]
        annotations = sample["annotations"]

        # Determine target split
        if page_id in train_pages:
            split = "train"
        elif page_id in val_pages:
            split = "val"
        else:
            split = "test"

        split_image_counts[split] += 1

        # A. Copy sketch image to images/<split>/
        src_img_path = os.path.join(RAW_DATASET_DIR, sketch_name)
        dst_img_path = os.path.join(YOLO_DIR, "images", split, sketch_name)
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)

        # B. Generate YOLO label file labels/<split>/<sketch_name>.txt
        txt_filename = os.path.splitext(sketch_name)[0] + ".txt"
        dst_txt_path = os.path.join(YOLO_DIR, "labels", split, txt_filename)

        yolo_lines = []
        for ann in annotations:
            cat = ann["category"]
            if cat not in CLASS_TO_ID:
                continue
            cid = CLASS_TO_ID[cat]
            bx, by, bw, bh = ann["box"]

            # Convert to YOLO center format normalized to [0, 1]
            x_center = (bx + (bw / 2.0)) / img_w
            y_center = (by + (bh / 2.0)) / img_h
            norm_w = bw / img_w
            norm_h = bh / img_h

            # Clip values between 0.0 and 1.0 for safety
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.001, min(1.0, norm_w))
            norm_h = max(0.001, min(1.0, norm_h))

            yolo_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            split_box_counts[split][cat] += 1

        with open(dst_txt_path, "w", encoding="utf-8") as f_lbl:
            f_lbl.write("\n".join(yolo_lines) + "\n")

    # 3. Create dataset.yaml configuration file
    yaml_path = os.path.join(YOLO_DIR, "dataset.yaml")
    abs_yolo_dir = os.path.abspath(YOLO_DIR).replace("\\", "/")
    
    yaml_content = f"""# DrawX: Hand-Drawn Sketch to UI Object Detection Dataset
path: {abs_yolo_dir}
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names:
"""
    for i, name in enumerate(CLASSES):
        yaml_content += f"  {i}: {name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f_yaml:
        f_yaml.write(yaml_content)

    # 4. Display Results Summary Table
    print("\n" + "=" * 65)
    print(" YOLO DATASET PREPARATION COMPLETED! ")
    print("=" * 65)
    print(f"YOLO Root Directory: {YOLO_DIR}")
    print(f"YOLO YAML Config:   {yaml_path}")
    print("\nFinal Split Counts (Sketch Images):")
    for s in splits:
        print(f"  - {s.upper():<6}: {split_image_counts[s]:>4} images")

    print("\nLabeled Boxes per Component Across Splits:")
    header = f"{'Category':<12} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}"
    print(header)
    print("-" * len(header))
    grand_total = 0
    for cat in CLASSES:
        tr = split_box_counts["train"][cat]
        va = split_box_counts["val"][cat]
        te = split_box_counts["test"][cat]
        tot = tr + va + te
        grand_total += tot
        print(f"{cat:<12} | {tr:<8} | {va:<8} | {te:<8} | {tot:<8}")
    print("-" * len(header))
    print(f"{'TOTAL':<12} | {sum(split_box_counts['train'].values()):<8} | {sum(split_box_counts['val'].values()):<8} | {sum(split_box_counts['test'].values()):<8} | {grand_total:<8}")
    print("=" * 65)

if __name__ == "__main__":
    main()

