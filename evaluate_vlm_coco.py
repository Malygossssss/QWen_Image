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

def _resolve_pred_root(path):
    """Return an existing prediction root, falling back to the repo default."""
    if os.path.isdir(path):
        return path

    repo_default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coco_outputs")
    if os.path.isdir(repo_default):
        print(
            f"⚠️ Prediction root '{path}' not found. Falling back to '{repo_default}'."
        )
        return repo_default

    raise FileNotFoundError(
        f"Prediction root '{path}' does not exist and no fallback directory was found."
    )


PRED_ROOT = _resolve_pred_root(PRED_ROOT)

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

def _find_prediction_file(folder):
    """Locate the prediction JSON file inside ``folder``.

    Older runs exported ``result_absolute.json`` while newer ones use
    ``result.json``.  We support both to avoid silent evaluation skips.
    """

    preferred_names = ["result_absolute.json", "result.json", "result.txt"]
    for name in preferred_names:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None

for img_folder in tqdm(sorted(os.listdir(PRED_ROOT))):
    folder_path = os.path.join(PRED_ROOT, img_folder)
    json_path = _find_prediction_file(folder_path)
    if json_path is None:
        if os.path.isdir(folder_path):
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
        try:
            dets = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ JSONDecodeError in file: {json_path}")
            f.seek(0)
            text = f.read().strip()
            print(f"File content preview: {text[:200]}")  # 看看文件里到底是什么
            continue
            # dets = json.loads(f.read())

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
    expected_names = ["result_absolute.json", "result.json", "result.txt"]
    print(
        "⚠️ The following prediction folders did not contain any supported result "
        f"file (expected one of {expected_names}):"
    )
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
