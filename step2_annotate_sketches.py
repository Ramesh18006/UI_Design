"""
Step 2: Data Annotation for Sketch UI Images
--------------------------------------------
This script creates bounding boxes and component labels for the sketch images.

How it works:
1. Extracts visible UI elements from each HTML file (button, heading, text, image, input).
2. Detects the hand-drawn component boxes on each sketch image using image processing.
3. Aligns the detected boxes with the semantic HTML elements.
4. Saves all annotations in JSON format (ready for YOLO/COCO conversion in Step 3).
5. Draws colorful boxes on 15 sample sketches and saves them in `data/annotated_samples/` 
   so you can open and visually inspect the results!
"""

import os
import re
import json
from collections import Counter
from html.parser import HTMLParser
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label, find_objects, binary_dilation

# Directory paths
DATASET_DIR = os.path.join("data", "raw", "sketch2code_dataset_v1_cleaned")
OUTPUT_ANNOTATIONS_DIR = os.path.join("data", "annotations")
OUTPUT_SAMPLES_DIR = os.path.join("data", "annotated_samples")

# The 6 UI Component Classes
CLASSES = ["button", "heading", "text", "image", "input", "container"]

# Colors for visual samples (RGB)
CLASS_COLORS = {
    "button": (46, 204, 113),     # Green
    "heading": (52, 152, 219),    # Blue
    "text": (243, 156, 18),       # Orange / Amber
    "image": (231, 76, 60),       # Red
    "input": (155, 89, 182),      # Purple
    "container": (127, 140, 141)  # Gray
}

class HTMLElementsParser(HTMLParser):
    """Extracts UI elements in the order they appear on the page."""
    def __init__(self):
        super().__init__()
        self.elements = []
        self.in_hidden_block = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        style = attrs_dict.get("style", "").lower()
        css_class = attrs_dict.get("class", "").lower()
        tag_lower = tag.lower()

        # Skip hidden sections
        if "display: none" in style or "display:none" in style:
            return

        # 1. Buttons
        if tag_lower == "button" or (tag_lower == "a" and ("btn" in css_class or "button" in css_class)):
            self.elements.append({"type": "button", "tag": tag_lower})
        elif tag_lower == "input" and attrs_dict.get("type", "").lower() in ["submit", "button"]:
            self.elements.append({"type": "button", "tag": tag_lower})

        # 2. Text Input fields
        elif tag_lower == "input" and attrs_dict.get("type", "text").lower() in ["text", "email", "password", "search", "tel", "url"]:
            self.elements.append({"type": "input", "tag": tag_lower})
        elif tag_lower == "textarea":
            self.elements.append({"type": "input", "tag": tag_lower})

        # 3. Images
        elif tag_lower == "img":
            self.elements.append({"type": "image", "tag": tag_lower})

        # 4. Headings
        elif tag_lower in ["h1", "h2", "h3", "h4"]:
            self.elements.append({"type": "heading", "tag": tag_lower})

        # 5. Body Text
        elif tag_lower in ["p", "label", "li"]:
            self.elements.append({"type": "text", "tag": tag_lower})

def detect_sketch_boxes(image_path):
    """
    Detects drawn UI components (boxes, buttons, text blocks) on a sketch image.
    Uses stroke thresholding and morphological dilation to group drawn lines.
    """
    with Image.open(image_path) as img:
        img_gray = img.convert("L")
        img_w, img_h = img.size
        arr = np.array(img_gray)

    # Dark pixels (less than 220) represent drawn ink strokes
    stroke_mask = arr < 220

    # Dilation structure: merges nearby strokes (e.g. text inside a button border)
    # Horizontally 15px, Vertically 8px
    struct = np.ones((8, 15), dtype=bool)
    dilated_mask = binary_dilation(stroke_mask, structure=struct)

    # Find connected components
    labeled_array, num_features = label(dilated_mask)
    slices = find_objects(labeled_array)

    boxes = []
    for s in slices:
        ymin, ymax = s[0].start, s[0].stop
        xmin, xmax = s[1].start, s[1].stop
        w = xmax - xmin
        h = ymax - ymin

        # Filter out tiny noise (specks < 14px width or < 8px height)
        # and filter out full page borders (> 95% of image width and height)
        if w < 14 or h < 8:
            continue
        if w > 0.96 * img_w and h > 0.96 * img_h:
            continue

        boxes.append([int(xmin), int(ymin), int(w), int(h)])

    # Sort boxes in reading order: top-to-bottom, left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes, img_w, img_h

def match_boxes_with_elements(detected_boxes, html_elements):
    """
    Assigns each detected box on the sketch to the most matching UI element from HTML.
    Uses top-to-bottom flow and shape properties (aspect ratio, height).
    """
    annotations = []
    num_boxes = len(detected_boxes)
    num_elems = len(html_elements)

    if num_boxes == 0:
        return annotations

    # If no HTML elements were parsed, classify purely by visual geometry
    if num_elems == 0:
        for box in detected_boxes:
            xmin, ymin, w, h = box
            aspect_ratio = w / max(h, 1)
            if aspect_ratio > 3.0 and h < 45:
                cat = "button"
            elif h > 50 and aspect_ratio < 2.5:
                cat = "image"
            elif h > 25:
                cat = "heading"
            else:
                cat = "text"
            annotations.append({"box": box, "category": cat})
        return annotations

    # Align sketch boxes with HTML elements sequentially
    elem_idx = 0
    for i, box in enumerate(detected_boxes):
        xmin, ymin, w, h = box
        aspect_ratio = w / max(h, 1)

        # Estimate the corresponding HTML element index based on relative vertical progress
        progress = i / max(num_boxes - 1, 1)
        target_elem_idx = min(int(progress * num_elems), num_elems - 1)
        candidate_elem = html_elements[target_elem_idx]
        cat = candidate_elem["type"]

        # Geometric refinement:
        # If the shape is distinctly square/tall with large area, it's likely an image
        if h > 55 and 0.6 <= aspect_ratio <= 1.8:
            cat = "image"
        # If it's a wide rectangle with medium height, it's a button or input
        elif 2.2 <= aspect_ratio <= 7.0 and 18 <= h <= 55 and cat not in ["button", "input"]:
            cat = "button"

        annotations.append({
            "box": box,        # [x, y, w, h]
            "category": cat
        })

    return annotations

def draw_visual_sample(image_path, annotations, output_path):
    """Draws colored bounding boxes with category labels on a sample sketch."""
    with Image.open(image_path) as img:
        annotated_img = img.convert("RGB")
        draw = ImageDraw.Draw(annotated_img)

        for ann in annotations:
            x, y, w, h = ann["box"]
            cat = ann["category"]
            color = CLASS_COLORS.get(cat, (0, 0, 0))

            # Draw rectangle border
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

            # Draw background tag pill for the label
            label_text = cat.upper()
            tag_w = len(label_text) * 7 + 4
            tag_h = 13
            draw.rectangle([x, max(0, y - tag_h), x + tag_w, max(0, y)], fill=color)
            draw.text((x + 2, max(0, y - tag_h)), label_text, fill=(255, 255, 255))

        annotated_img.save(output_path)

def main():
    print("=" * 60)
    print(" STEP 2: GENERATING BOUNDING BOXES & DATA ANNOTATION ")
    print("=" * 60)

    os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SAMPLES_DIR, exist_ok=True)

    all_files = os.listdir(DATASET_DIR)
    sketch_files = sorted([f for f in all_files if re.match(r"^\d+_\d+\.png$", f)])
    print(f"Total sketch images to annotate: {len(sketch_files)}")

    master_annotations = {}
    class_counter = Counter()
    visual_sample_count = 0
    max_visual_samples = 15

    for idx, sketch_file in enumerate(sketch_files):
        sketch_path = os.path.join(DATASET_DIR, sketch_file)
        
        # Determine the parent webpage ID (e.g. 1002 from 1002_0.png)
        page_id = sketch_file.split("_")[0]
        html_file = f"{page_id}.html"
        html_path = os.path.join(DATASET_DIR, html_file)

        # 1. Parse HTML elements if available
        html_elements = []
        if os.path.exists(html_path):
            try:
                with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                parser = HTMLElementsParser()
                parser.feed(content)
                html_elements = parser.elements
            except Exception:
                pass

        # 2. Detect sketch bounding boxes
        boxes, img_w, img_h = detect_sketch_boxes(sketch_path)

        # 3. Match boxes with semantic labels
        annotations = match_boxes_with_elements(boxes, html_elements)

        # Save to master collection
        master_annotations[sketch_file] = {
            "image_size": [img_w, img_h],
            "page_id": page_id,
            "annotations": annotations
        }

        for ann in annotations:
            class_counter[ann["category"]] += 1

        # 4. Save visual sample images for the first 15 sketches
        if visual_sample_count < max_visual_samples:
            sample_out_path = os.path.join(OUTPUT_SAMPLES_DIR, f"annotated_{sketch_file}")
            draw_visual_sample(sketch_path, annotations, sample_out_path)
            visual_sample_count += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == len(sketch_files):
            print(f"  Processed {idx + 1}/{len(sketch_files)} sketches...")

    # Save the master JSON annotation file
    master_json_path = os.path.join(OUTPUT_ANNOTATIONS_DIR, "annotations.json")
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(master_annotations, f, indent=2)

    print("\n" + "=" * 60)
    print(" ANNOTATION COMPLETED SUCCESSFULLY! ")
    print("=" * 60)
    print(f"Master annotations saved to: {master_json_path}")
    print(f"Visual sample images saved to: {OUTPUT_SAMPLES_DIR}")
    print(f"Total sketches annotated: {len(master_annotations):,}")
    print("\nComponent Distribution Across All Sketches:")
    total_components = sum(class_counter.values())
    for cat in CLASSES:
        count = class_counter[cat]
        pct = (count / max(total_components, 1)) * 100
        print(f"  - {cat:<10}: {count:>6,} boxes ({pct:>5.1f}%)")
    print(f"  -------------------------------------")
    print(f"  TOTAL     : {total_components:>6,} labeled boxes")
    print("=" * 60)

if __name__ == "__main__":
    main()

