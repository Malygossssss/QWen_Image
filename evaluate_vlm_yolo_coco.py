"""
Evaluate VLM detection results on YOLO-format Pascal VOC dataset.
Outputs YOLO-style metrics: mAP@0.5, mAP@0.5:0.95, Precision, Recall
"""

import os
import sys
import json
import yaml
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==============================
# User config
# ==============================
DATA_ROOT = "/mnt/pub/wyf/workspace/image_identification/Pascal-VOC-2012/valid"
LABEL_DIR = os.path.join(DATA_ROOT, "labels")
IMG_DIR = os.path.join(DATA_ROOT, "images")
PRED_ROOT = "/mnt/pub/wyf/workspace/image_identification/outputs"  # 你的VLM输出目录
DATA_YAML = os.path.join(os.path.dirname(DATA_ROOT), "data.yaml")

GT_JSON = "gt_coco.json"
PRED_JSON = "pred_coco.json"

# ==============================
# Step 1. Load category names
# ==============================
if os.path.exists(DATA_YAML):
    with open(DATA_YAML, "r") as f:
        data_yaml = yaml.safe_load(f)
    classes = data_yaml.get("names", [])
else:
    classes = [str(i) for i in range(20)]
name2id = {name: i + 1 for i, name in enumerate(classes)}
print(f"✅ Loaded {len(classes)} classes: {classes}")

# ==============================
# Step 2. Convert GT (YOLO txt → COCO JSON)
# ==============================
print("🔄 Converting YOLO labels to COCO ground truth format...")

coco_gt = {"images": [], "annotations": [], "categories": []}
for name, cid in name2id.items():
    coco_gt["categories"].append({"id": cid, "name": name})

ann_id = 1
img_id = 1
for label_file in tqdm(sorted(os.listdir(LABEL_DIR))):
    if not label_file.endswith(".txt"):
        continue
    img_name = os.path.splitext(label_file)[0] + ".jpg"
    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(img_path):
        img_name = os.path.splitext(label_file)[0] + ".png"
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue

    w, h = Image.open(img_path).size
    coco_gt["images"].append({
        "id": img_id,
        "file_name": img_name,
        "width": w,
        "height": h
    })

    with open(os.path.join(LABEL_DIR, label_file), "r") as f:
        for line in f.readlines():
            cls, xc, yc, bw, bh = map(float, line.strip().split())
            x = (xc - bw / 2) * w
            y = (yc - bh / 2) * h
            bw *= w
            bh *= h
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls) + 1,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            ann_id += 1
    img_id += 1

with open(GT_JSON, "w") as f:
    json.dump(coco_gt, f, indent=2)
print(f"✅ Saved ground truth to {GT_JSON}")

# ==============================
# Step 3. Convert Predictions (result.json → COCO JSON)
# ==============================
print("🔄 Converting VLM predictions to COCO format...")

preds = []
img_to_id = {}
for im in coco_gt["images"]:
    file_name = im["file_name"]
    stem = os.path.splitext(file_name)[0]
    img_to_id[stem] = im["id"]
    img_to_id[file_name] = im["id"]

missing_prediction = []
missing_json = []

for img_folder in tqdm(sorted(os.listdir(PRED_ROOT))):
    json_path = os.path.join(PRED_ROOT, img_folder, "result_absolute.json")
    if not os.path.exists(json_path):
        if os.path.isdir(os.path.join(PRED_ROOT, img_folder)):
            missing_json.append(img_folder)
        continue
    
    key = img_folder
    if key not in img_to_id:
        stem = os.path.splitext(img_folder)[0]
        key = stem if stem in img_to_id else key

    if key not in img_to_id:
        missing_prediction.append(img_folder)
        continue
    
    image_id = img_to_id[key]
    with open(json_path, "r") as f:
        dets = json.load(f)

    for det in dets:
        x1, y1, x2, y2 = det["bbox_2d"]
        w, h = x2 - x1, y2 - y1
        label = det["label"]
        cat_id = name2id.get(label, None)
        if cat_id is None:
            continue
        preds.append({
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [x1, y1, w, h],
            "score": 1.0  # 如果没有置信度，默认1.0
        })

with open(PRED_JSON, "w") as f:
    json.dump(preds, f, indent=2)
print(f"✅ Saved predictions to {PRED_JSON}")

if missing_json:
    print("⚠️ The following prediction folders did not contain result.json:")
    for folder in missing_json:
        print(f"   - {folder}")

if missing_prediction:
    print("⚠️ Could not match the following prediction folders to any ground-truth image."
          " Please ensure folder names align with image file names:")
    for folder in missing_prediction:
        print(f"   - {folder}")

if len(preds) == 0:
    print("⚠️ No predictions were matched to ground truth images. Skipping evaluation.")
    sys.exit(0)

# ==============================
# Step 4. Evaluate using COCO API
# ==============================
print("📊 Evaluating with pycocotools...")

cocoGt = COCO(GT_JSON)
cocoDt = cocoGt.loadRes(PRED_JSON)
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

print("\n📈 Summary of YOLO-style metrics:")
print("  mAP@0.5:0.95  = {:.3f}".format(cocoEval.stats[0]))
print("  mAP@0.5       = {:.3f}".format(cocoEval.stats[1]))
print("  Precision     = {:.3f}".format(cocoEval.stats[8]))
print("  Recall        = {:.3f}".format(cocoEval.stats[9]))
