"""
Step 2 (Improved): Screenshot-Guided Data Annotation for Sketch UI Images
-----------------------------------------------------------------------
This updated script fixes previous misclassifications by using:
1. Ground-Truth Webpage Screenshot (1280px): Finds exact UI component positions.
2. HTML Element Semantics: Accurately identifies headings, buttons, text, images, and inputs.
3. Proportional Projection: Maps screenshot components to the hand-drawn sketch (600px).
4. Ink-Stroke Snapping: Snaps the bounding box tightly to the hand-drawn sketch ink lines.
5. Visual Verification: Saves 15 visual samples with colored boxes in data/annotated_samples/.
"""

import os
import re
import json
from collections import Counter
from html.parser import HTMLParser
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, find_objects, binary_dilation

DATASET_DIR = os.path.join("data", "raw", "sketch2code_dataset_v1_cleaned")
OUTPUT_ANNOTATIONS_DIR = os.path.join("data", "annotations")
OUTPUT_SAMPLES_DIR = os.path.join("data", "annotated_samples")

CLASSES = ["button", "heading", "text", "image", "input"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

CLASS_COLORS = {
    "button": (46, 204, 113),   # Green
    "heading": (52, 152, 219),  # Blue
    "text": (243, 156, 18),     # Orange
    "image": (231, 76, 60),     # Red
    "input": (155, 89, 182)     # Purple
}

class SemanticHTMLExtractor(HTMLParser):
    """Extracts UI elements and classification clues from HTML."""
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        style = attrs_dict.get("style", "").lower()
        css_class = attrs_dict.get("class", "").lower()
        tag_lower = tag.lower()

        if "display: none" in style or "display:none" in style:
            return

        if tag_lower == "button" or (tag_lower == "a" and ("btn" in css_class or "button" in css_class)):
            self.elements.append("button")
        elif tag_lower == "input" and attrs_dict.get("type", "").lower() in ["submit", "button"]:
            self.elements.append("button")
        elif tag_lower == "input" and attrs_dict.get("type", "text").lower() in ["text", "email", "password", "search", "tel", "url"]:
            self.elements.append("input")
        elif tag_lower == "textarea":
            self.elements.append("input")
        elif tag_lower == "img":
            self.elements.append("image")
        elif tag_lower in ["h1", "h2", "h3", "h4"]:
            self.elements.append("heading")
        elif tag_lower in ["p", "label", "li"]:
            self.elements.append("text")

def extract_screenshot_blocks(screenshot_path):
    """Detects separated UI component regions on the clean browser screenshot."""
    with Image.open(screenshot_path) as im:
        rgb = im.convert("RGB")
        sw, sh = rgb.size
        arr = np.array(rgb)

    # Detect non-white content
    diff = np.abs(arr.astype(int) - [255, 255, 255]).sum(axis=2)
    content_mask = diff > 25

    # Group words/elements into UI blocks
    struct = np.ones((14, 26), dtype=bool)
    dilated = binary_dilation(content_mask, structure=struct)
    labeled, _ = label(dilated)

    boxes = []
    for s in find_objects(labeled):
        ymin, ymax = s[0].start, s[0].stop
        xmin, xmax = s[1].start, s[1].stop
        bw = xmax - xmin
        bh = ymax - ymin

        if bw > 25 and bh > 14 and (bw < 0.96 * sw or bh < 0.96 * sh):
            # Check visual features of the block on the screenshot
            sub_arr = arr[ymin:ymax, xmin:xmax]
            boxes.append((xmin, ymin, bw, bh, sub_arr))

    # Sort blocks top-to-bottom, left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes, sw, sh

def classify_screenshot_block(block, html_elements, block_idx, total_blocks):
    """Accurately classifies a screenshot block using pixel properties and HTML semantics."""
    xmin, ymin, bw, bh, sub_arr = block

    # Feature 1: Color richness (colored button or photo image vs black text on white)
    std_color = sub_arr.std(axis=(0, 1))
    is_colorful = (std_color > 25).any()
    
    # Feature 2: Dark background fill (common for buttons)
    mean_val = sub_arr.mean()
    is_dark_fill = mean_val < 180

    # Feature 3: Size and aspect ratio
    aspect_ratio = bw / max(bh, 1)

    # If HTML elements are available, use progress-based alignment
    if html_elements:
        prog = block_idx / max(total_blocks - 1, 1)
        target_idx = min(int(prog * len(html_elements)), len(html_elements) - 1)
        html_cat = html_elements[target_idx]
    else:
        html_cat = "text"

    # Strict Decision Tree:
    if is_colorful and bh > 45 and 0.5 <= aspect_ratio <= 2.2:
        return "image"
    
    if (is_dark_fill or html_cat == "button") and (1.5 <= aspect_ratio <= 9.0) and (20 <= bh <= 65):
        return "button"

    if html_cat == "input" or (2.5 <= aspect_ratio <= 12.0 and 22 <= bh <= 50 and not is_colorful and mean_val > 220):
        # Single-line light box often represents an input field
        if html_cat == "input":
            return "input"

    if bh > 32 and (html_cat == "heading" or aspect_ratio < 7.0 and bh > 36):
        return "heading"

    if html_cat in ["heading", "image", "button", "input"]:
        return html_cat

    return "text"

def snap_box_to_sketch_strokes(box, ink_mask, pad=12):
    """Snaps the projected bounding box tightly to the hand-drawn ink strokes."""
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img_h, img_w = ink_mask.shape

    x1 = int(max(0, x - pad))
    y1 = int(max(0, y - pad))
    x2 = int(min(img_w, x + w + pad))
    y2 = int(min(img_h, y + h + pad))

    sub = ink_mask[y1:y2, x1:x2]
    ys, xs = np.where(sub)

    if len(xs) > 0 and len(ys) > 0:
        snapped_x1 = int(x1 + xs.min())
        snapped_x2 = int(x1 + xs.max() + 1)
        snapped_y1 = int(y1 + ys.min())
        snapped_y2 = int(y1 + ys.max() + 1)
        snapped_w = max(snapped_x2 - snapped_x1, 14)
        snapped_h = max(snapped_y2 - snapped_y1, 10)
        return [snapped_x1, snapped_y1, snapped_w, snapped_h]

    return [int(x), int(y), int(w), int(h)]

def draw_visual_sample(image_path, annotations, output_path):
    """Draws color-coded bounding boxes and tags on sample sketches."""
    with Image.open(image_path) as img:
        annotated_img = img.convert("RGB")
        draw = ImageDraw.Draw(annotated_img)

        for ann in annotations:
            x, y, w, h = ann["box"]
            cat = ann["category"]
            color = CLASS_COLORS.get(cat, (0, 0, 0))

            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            label_text = cat.upper()
            tag_w = len(label_text) * 7 + 4
            tag_h = 13
            draw.rectangle([x, max(0, y - tag_h), x + tag_w, max(0, y)], fill=color)
            draw.text((x + 2, max(0, y - tag_h)), label_text, fill=(255, 255, 255))

        annotated_img.save(output_path)

def main():
    print("=" * 65)
    print(" STEP 2 (IMPROVED): SCREENSHOT-GUIDED DATA ANNOTATION ")
    print("=" * 65)

    os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SAMPLES_DIR, exist_ok=True)

    all_files = os.listdir(DATASET_DIR)
    sketch_files = sorted([f for f in all_files if re.match(r"^\d+_\d+\.png$", f)])
    print(f"Total sketches to annotate: {len(sketch_files)}")

    master_annotations = {}
    class_counter = Counter()
    visual_sample_count = 0
    max_visual_samples = 15

    # Cache screenshot extraction per webpage to run fast
    screenshot_cache = {}

    for idx, sketch_file in enumerate(sketch_files):
        sketch_path = os.path.join(DATASET_DIR, sketch_file)
        page_id = sketch_file.split("_")[0]

        # 1. Load or process screenshot and HTML for this page
        if page_id not in screenshot_cache:
            screen_file = f"{page_id}.png"
            screen_path = os.path.join(DATASET_DIR, screen_file)
            html_file = f"{page_id}.html"
            html_path = os.path.join(DATASET_DIR, html_file)

            html_elements = []
            if os.path.exists(html_path):
                try:
                    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                        parser = SemanticHTMLExtractor()
                        parser.feed(f.read())
                        html_elements = parser.elements
                except Exception:
                    pass

            if os.path.exists(screen_path):
                screen_blocks, sw, sh = extract_screenshot_blocks(screen_path)
                classified_blocks = []
                for b_i, blk in enumerate(screen_blocks):
                    cat = classify_screenshot_block(blk, html_elements, b_i, len(screen_blocks))
                    classified_blocks.append((blk[0], blk[1], blk[2], blk[3], cat))
                screenshot_cache[page_id] = (classified_blocks, sw, sh)
            else:
                screenshot_cache[page_id] = ([], 1280, 720)

        screen_blocks, sw, sh = screenshot_cache[page_id]

        # 2. Open sketch and load ink mask
        with Image.open(sketch_path) as sk_img:
            kw, kh = sk_img.size
            gray_arr = np.array(sk_img.convert("L"))
            ink_mask = gray_arr < 220

        # Scale factor from screenshot to sketch
        scale_x = kw / sw
        scale_y = kh / sh

        annotations = []
        for sx, sy, sbw, sbh, cat in screen_blocks:
            # Project box to sketch
            proj_x = sx * scale_x
            proj_y = sy * scale_y
            proj_w = sbw * scale_x
            proj_h = sbh * scale_y

            # Snap to hand-drawn ink strokes
            snapped_box = snap_box_to_sketch_strokes([proj_x, proj_y, proj_w, proj_h], ink_mask)

            # Ensure inside sketch boundary
            bx, by, bw, bh = snapped_box
            bx = max(0, min(bx, kw - 14))
            by = max(0, min(by, kh - 10))
            bw = min(bw, kw - bx)
            bh = min(bh, kh - by)

            if bw >= 14 and bh >= 10:
                annotations.append({
                    "box": [bx, by, bw, bh],
                    "category": cat
                })
                class_counter[cat] += 1

        master_annotations[sketch_file] = {
            "image_size": [kw, kh],
            "page_id": page_id,
            "annotations": annotations
        }

        # 3. Save visual preview samples
        if visual_sample_count < max_visual_samples:
            sample_out = os.path.join(OUTPUT_SAMPLES_DIR, f"annotated_{sketch_file}")
            draw_visual_sample(sketch_path, annotations, sample_out)
            visual_sample_count += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == len(sketch_files):
            print(f"  Annotated {idx + 1}/{len(sketch_files)} sketches...")

    # Save Master JSON
    master_json_path = os.path.join(OUTPUT_ANNOTATIONS_DIR, "annotations.json")
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master_annotations, f, indent=2)

    print("\n" + "=" * 65)
    print(" IMPROVED ANNOTATIONS COMPLETED SUCCESSFULLY! ")
    print("=" * 65)
    print(f"Master Annotations JSON: {master_json_path}")
    print(f"Visual Samples Saved To: {OUTPUT_SAMPLES_DIR}")
    print("\nUpdated Component Distribution Across All Sketches:")
    total = sum(class_counter.values())
    for cat in CLASSES:
        count = class_counter[cat]
        pct = (count / max(total, 1)) * 100
        print(f"  - {cat:<10}: {count:>5,} boxes ({pct:>5.1f}%)")
    print(f"  -------------------------------------")
    print(f"  TOTAL     : {total:>5,} labeled boxes")
    print("=" * 65)

if __name__ == "__main__":
    main()
