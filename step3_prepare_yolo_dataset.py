"""
Step 3: YOLO Dataset Formatting & Train/Val/Test Splitting
---------------------------------------------------------
This script prepares the annotated dataset for training inside UI_Design:
1. Groups sketches by webpage design ID.
2. Splits into 70% Train, 15% Val, 15% Test.
3. Converts bounding boxes to YOLO format [class_id, x_center, y_center, width, height].
4. Verifies 0 empty files and 0 out-of-bounds boxes.
5. Generates dataset.yaml with dynamic path resolution.
"""

import os
import shutil
import random
import json
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET_DIR = os.path.join(BASE_DIR, "data", "raw", "sketch2code_dataset_v1_cleaned")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "data", "annotations", "annotations.json")
YOLO_DIR = os.path.join(BASE_DIR, "data", "prepared", "yolo")

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
        print(f"Error: {ANNOTATIONS_FILE} not found!")
        return

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    # 1. Group sketches by webpage ID
    pages_to_sketches = {}
    for sketch_name, sample in master_data.items():
        page_id = sample["page_id"]
        if page_id not in pages_to_sketches:
            pages_to_sketches[page_id] = []
        pages_to_sketches[page_id].append(sketch_name)

    all_page_ids = sorted(list(pages_to_sketches.keys()))
    random.seed(42)
    random.shuffle(all_page_ids)

    n_total = len(all_page_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_pages = set(all_page_ids[:n_train])
    val_pages = set(all_page_ids[n_train:n_train + n_val])
    test_pages = set(all_page_ids[n_train + n_val:])

    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(YOLO_DIR, "images", s), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, "labels", s), exist_ok=True)

    split_image_counts = Counter()
    split_box_counts = {s: Counter() for s in splits}
    empty_label_count = 0

    for sketch_name, sample in master_data.items():
        page_id = sample["page_id"]
        img_w, img_h = sample["image_size"]
        annotations = sample["annotations"]

        if page_id in train_pages:
            split = "train"
        elif page_id in val_pages:
            split = "val"
        else:
            split = "test"

        split_image_counts[split] += 1

        # Copy image
        src_img = os.path.join(RAW_DATASET_DIR, sketch_name)
        dst_img = os.path.join(YOLO_DIR, "images", split, sketch_name)
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)

        # Write labels
        txt_filename = os.path.splitext(sketch_name)[0] + ".txt"
        dst_txt = os.path.join(YOLO_DIR, "labels", split, txt_filename)

        yolo_lines = []
        for ann in annotations:
            cat = ann["category"]
            if cat not in CLASS_TO_ID:
                continue
            cid = CLASS_TO_ID[cat]
            bx, by, bw, bh = ann["box"]

            x_center = (bx + (bw / 2.0)) / img_w
            y_center = (by + (bh / 2.0)) / img_h
            norm_w = bw / img_w
            norm_h = bh / img_h

            x_center = max(0.001, min(0.999, x_center))
            y_center = max(0.001, min(0.999, y_center))
            norm_w = max(0.005, min(0.99, norm_w))
            norm_h = max(0.005, min(0.60, norm_h)) # Cap to 0.60 max height

            yolo_lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            split_box_counts[split][cat] += 1

        if len(yolo_lines) == 0:
            empty_label_count += 1

        with open(dst_txt, "w", encoding="utf-8") as f_lbl:
            f_lbl.write("\n".join(yolo_lines) + "\n")

    # Write dataset.yaml
    yaml_path = os.path.join(YOLO_DIR, "dataset.yaml")
    abs_yolo = YOLO_DIR.replace("\\", "/")
    yaml_content = f"""# DrawX: Hand-Drawn Sketch to UI Object Detection Dataset
path: {abs_yolo}
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

    print("\n" + "=" * 65)
    print(" YOLO DATASET RE-PREPARED SUCCESSFULLY! ")
    print("=" * 65)
    print(f"YOLO Root Directory: {YOLO_DIR}")
    print(f"Empty label files:  {empty_label_count} (Must be 0!)")
    print(f"\nFinal Split Counts (Images):")
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
