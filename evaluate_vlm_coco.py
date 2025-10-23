"""
Evaluate VLM detection results on COCO 2017 validation set.
Outputs COCO-style metrics: mAP@0.5, mAP@0.5:0.95, Precision, Recall
"""

import os
import sys
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==============================
# 🔧 User config
# ==============================
DATA_ROOT = "/mnt/pub/wyf/workspace/image_identification/coco2017/val2017"   # 🔧 改为你的COCO val2017路径
GT_JSON_PATH = "/mnt/pub/wyf/workspace/image_identification/coco2017/annotations/instances_val2017.json"  # 🔧 官方COCO GT
PRED_ROOT = "/mnt/pub/wyf/workspace/image_identification/coco_outputs"       # 🔧 你的Qwen输出目录

PRED_JSON = "pred_coco.json"

# ==============================
# Step 1. Load COCO classes
# ==============================
print("🔧 Loading COCO categories...")
coco_gt = COCO(GT_JSON_PATH)
cats = coco_gt.loadCats(coco_gt.getCatIds())
name2id = {c["name"]: c["id"] for c in cats}
print(f"✅ Loaded {len(cats)} classes: {[c['name'] for c in cats]}")

# ==============================
# Step 2. Convert Predictions (result_absolute.json → COCO JSON)
# ==============================
print("🔄 Converting VLM predictions to COCO format...")

preds = []
img_to_id = {}
for im in coco_gt.dataset["images"]:
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
        # 🔧 支持 Qwen 输出格式：{"label": str, "bbox_2d": [x1, y1, x2, y2]}
        if "bbox_2d" not in det or "label" not in det:
            continue
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
            "score": det.get("score", 1.0)  # 🔧 若无置信度则默认为1.0
        })

with open(PRED_JSON, "w") as f:
    json.dump(preds, f, indent=2)
print(f"✅ Saved predictions to {PRED_JSON}")

if missing_json:
    print("⚠️ The following prediction folders did not contain result_absolute.json:")
    for folder in missing_json:
        print(f"   - {folder}")

if missing_prediction:
    print("⚠️ Could not match the following prediction folders to any ground-truth image:")
    for folder in missing_prediction:
        print(f"   - {folder}")

if len(preds) == 0:
    print("⚠️ No predictions matched ground truth images. Skipping evaluation.")
    sys.exit(0)

# ==============================
# Step 3. Evaluate using COCO API
# ==============================
print("📊 Evaluating with pycocotools...")

cocoDt = coco_gt.loadRes(PRED_JSON)
cocoEval = COCOeval(coco_gt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

print("\n📈 Summary of COCO-style metrics:")
print("  mAP@0.5:0.95  = {:.3f}".format(cocoEval.stats[0]))
print("  mAP@0.5       = {:.3f}".format(cocoEval.stats[1]))
print("  Precision     = {:.3f}".format(cocoEval.stats[8]))
print("  Recall        = {:.3f}".format(cocoEval.stats[9]))
