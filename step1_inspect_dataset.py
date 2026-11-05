"""
Step 1: Dataset Inspection & Tag Analysis
-----------------------------------------
This script does three simple things:
1. Counts the total number of HTML files, rendered screenshots, and sketch images.
2. Checks image resolutions (width and height) to see if sketches match screenshots.
3. Scans all HTML files to find what UI elements exist (buttons, headings, inputs, images, etc.).
"""

import os
import re
from collections import Counter
from html.parser import HTMLParser
from PIL import Image

# Path where your dataset files are stored
DATASET_DIR = os.path.join("data", "raw", "sketch2code_dataset_v1_cleaned")

class SimpleHTMLTagExtractor(HTMLParser):
    """A simple HTML parser to count tags and collect interactive elements."""
    def __init__(self):
        super().__init__()
        self.tags = []
        self.interactive_elements = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag.lower())
        attrs_dict = dict(attrs)

        # Check if an <a> tag looks like a button
        css_classes = attrs_dict.get("class", "").lower()
        if tag.lower() == "a" and ("btn" in css_classes or "button" in css_classes):
            self.interactive_elements.append("button_link")
        elif tag.lower() == "button":
            self.interactive_elements.append("button")
        elif tag.lower() == "input":
            input_type = attrs_dict.get("type", "text").lower()
            self.interactive_elements.append(f"input_{input_type}")

def inspect_dataset():
    print("=" * 60)
    print(" STEP 1: INSPECTING DATASET & HTML TAGS ")
    print("=" * 60)

    if not os.path.exists(DATASET_DIR):
        print(f"Error: Directory not found -> {DATASET_DIR}")
        return

    all_files = os.listdir(DATASET_DIR)
    
    html_files = []
    screenshot_files = []
    sketch_files = []
    other_files = []

    # Pattern: <id>.html, <id>.png (rendered), <id>_<idx>.png (sketch)
    sketch_pattern = re.compile(r"^(\d+)_(\d+)\.png$")
    screenshot_pattern = re.compile(r"^(\d+)\.png$")
    html_pattern = re.compile(r"^(\d+)\.html$")

    for f in all_files:
        if html_pattern.match(f):
            html_files.append(f)
        elif sketch_pattern.match(f):
            sketch_files.append(f)
        elif screenshot_pattern.match(f):
            screenshot_files.append(f)
        else:
            other_files.append(f)

    # 1. Basic Counts
    print(f"\n[1] File Breakdown:")
    print(f"  - Webpage HTML files (.html):         {len(html_files)}")
    print(f"  - Rendered Screenshots (.png):        {len(screenshot_files)}")
    print(f"  - Hand-drawn Sketch images (_*.png):  {len(sketch_files)}")
    print(f"  - Other auxiliary files:             {len(other_files)} (e.g., {other_files[:3]})")

    # Group by webpage ID to verify pairings
    webpage_ids = set([f.split(".")[0] for f in html_files])
    print(f"  - Unique Webpage Designs:            {len(webpage_ids)}")

    # Check how many sketches exist per webpage
    sketches_per_page = Counter()
    for s in sketch_files:
        page_id = s.split("_")[0]
        sketches_per_page[page_id] += 1
    
    sketch_counts = Counter(sketches_per_page.values())
    print("\n[2] Sketch Distribution per Webpage:")
    for count, pages in sorted(sketch_counts.items()):
        print(f"  - Pages with {count} sketch(es): {pages}")

    # 2. Check Image Dimensions (Sampling first 20 images)
    print("\n[3] Checking Image Sizes & Resolutions (Sample of 20):")
    screenshot_sizes = set()
    sketch_sizes = set()

    for f in screenshot_files[:20]:
        img_path = os.path.join(DATASET_DIR, f)
        with Image.open(img_path) as img:
            screenshot_sizes.add(img.size)

    for f in sketch_files[:20]:
        img_path = os.path.join(DATASET_DIR, f)
        with Image.open(img_path) as img:
            sketch_sizes.add(img.size)

    print(f"  - Rendered Screenshot Sizes (W x H): {list(screenshot_sizes)}")
    print(f"  - Hand-drawn Sketch Sizes (W x H):   {list(sketch_sizes)}")

    # 3. HTML Tag Frequency Analysis
    print("\n[4] Scanning HTML UI Elements across all HTML files...")
    tag_counter = Counter()
    interactive_counter = Counter()

    for html_file in html_files:
        filepath = os.path.join(DATASET_DIR, html_file)
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            parser = SimpleHTMLTagExtractor()
            parser.feed(content)
            tag_counter.update(parser.tags)
            interactive_counter.update(parser.interactive_elements)
        except Exception as e:
            continue

    print("\n  Top 15 Most Common HTML Tags:")
    for tag, count in tag_counter.most_common(15):
        print(f"    - <{tag:<8}> : {count:,} times")

    print("\n  Interactive / Form Elements:")
    for elem, count in interactive_counter.most_common():
        print(f"    - {elem:<15} : {count:,} times")

    # 4. Proposed UI Component Classes
    print("\n" + "=" * 60)
    print(" RECOMMENDED COMPONENT CLASSES FOR OBJECT DETECTION:")
    print("=" * 60)
    print(" 1. 'button'     -> <button>, <a class='btn'>, <input type='submit'>")
    print(" 2. 'heading'    -> <h1>, <h2>, <h3>, <h4>")
    print(" 3. 'text'       -> <p>, <span>, <label>")
    print(" 4. 'image'      -> <img>, image placeholders")
    print(" 5. 'input'      -> <input type='text/email/password'>, <textarea>")
    print(" 6. 'container'  -> <div>, card containers, navigation bars")
    print("=" * 60)

if __name__ == "__main__":
    inspect_dataset()

