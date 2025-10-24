import os
import json
import torch
from PIL import Image
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForImageTextToText
import re
from typing import List
import argparse


_DEF_CATEGORIES = []


def _load_coco_categories(annotation_path: str) -> List[str]:
    """Load category names from a COCO annotation file."""
    if not os.path.exists(annotation_path):
        return _DEF_CATEGORIES

    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _DEF_CATEGORIES

    categories = sorted({cat.get("name", "").strip() for cat in data.get("categories", []) if cat.get("name")})
    return categories or _DEF_CATEGORIES



def _build_recover_pattern(class_names: List[str]) -> re.Pattern:
    class_pattern = "|".join(re.escape(cls) for cls in class_names)
    return re.compile(
        r"\{\s*\"bbox_2d\"\s*:\s*\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]\s*,\s*"
        r"\"label\"\s*:\s*\"(?:" + class_pattern + r")\"\s*\}"
    )


_RECOVER_PATTERN = _build_recover_pattern(_DEF_CATEGORIES)


def recover_partial_detections(raw_text: str):
    """Recover detection results from partial JSON fragments."""
    recovered = []
    for match in _RECOVER_PATTERN.finditer(raw_text):
        try:
            recovered.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return recovered if recovered else None


def convert_relative_to_absolute(detections, image_width, image_height):
    """Convert relative bbox coordinates (0-1000) to pixel coordinates."""
    converted = []
    for det in detections:
        if not isinstance(det, dict):
            print(f"⚠️ 跳过异常检测项：{det}")
            continue
        bbox = det.get("bbox_2d")

        # [ADDED] 结构防护：确保 bbox 是一个包含4个数的列表
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            print(f"⚠️ 跳过异常检测框：{bbox}")
            continue

        # 有些模型输出可能是字符串 -> 尝试解析
        if isinstance(bbox[0], str):
            try:
                bbox = [float(x) for x in bbox[:4]]
            except Exception:
                print(f"⚠️ 无法解析bbox字符串: {bbox}")
                continue

        rel_x1, rel_y1, rel_x2, rel_y2 = bbox[:4]
        abs_x1 = int(rel_x1 / 1000.0 * image_width)
        abs_y1 = int(rel_y1 / 1000.0 * image_height)
        abs_x2 = int(rel_x2 / 1000.0 * image_width)
        abs_y2 = int(rel_y2 / 1000.0 * image_height)

        # [MODIFIED] 支持多种字段名（label / category / class）
        label_name = det.get("label") or det.get("category") or det.get("class") or "unknown"

        converted.append({
            "bbox_2d": [abs_x1, abs_y1, abs_x2, abs_y2],
            "label": label_name
        })
    return converted


def draw_bboxes(image, detections):
    """Draw detection bounding boxes on a PIL.Image."""
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox_2d"])
        label = det["label"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value for --skip-existing (true/false).")


parser = argparse.ArgumentParser(description="Run Qwen VL detection on COCO2017 validation images.")
parser.add_argument(
    "--skip-existing",
    type=_str_to_bool,
    default=True,
    help="When true, skip images that already have outputs under coco_outputs (default: true).",
)
args = parser.parse_args()

MODEL_PATH = "/nas_pub_data/models/base/qwen3-vl-4b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = "coco_outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print(f"Loading model from {MODEL_PATH} on {DEVICE}...")

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True
).eval()


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(REPO_ROOT, "coco2017")
IMAGE_DIR = os.path.join(DATA_ROOT, "val2017")
ANNOTATION_FILE = os.path.join(DATA_ROOT, "annotations", "instances_val2017.json")

if not os.path.isdir(IMAGE_DIR):
    raise FileNotFoundError(
        f"Expected COCO2017 validation images at {IMAGE_DIR}. Please ensure the dataset is placed correctly."
    )

coco_classes = _load_coco_categories(ANNOTATION_FILE)
_RECOVER_PATTERN = _build_recover_pattern(coco_classes)
_CLASS_LIST_TEXT = ", ".join(f"'{name}'" for name in coco_classes)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

### [MODIFIED]
# 改为自动检测所有图片：已存在结果文件则自动跳过，无需手动指定断点
for img_file in sorted(image_files):
    img_name = os.path.splitext(img_file)[0]
    save_dir = os.path.join(OUTPUT_ROOT, img_name)
    result_json = os.path.join(save_dir, "result.json")

     # 如果结果文件已存在且开启跳过，则自动跳过，防止重复处理
    if args.skip_existing and os.path.exists(result_json):
        print(f"⏩ 已检测到 {result_json}，跳过 {img_file}")
        continue
### [END MODIFIED]

    image_path = os.path.join(IMAGE_DIR, img_file)
    image = Image.open(image_path).convert("RGB")

    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (
                "Locate every instance that belongs to the following COCO2017 categories: "
                + _CLASS_LIST_TEXT 
                + ". Report each detection as {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"class_name\"} in JSON format. "
                + "Use the key name 'label' (not 'category')."
            )}
        ]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=4096)

    result_text = processor.batch_decode(output, skip_special_tokens=True)[0]
    print(f"\n📷 {img_file} Raw model output:\n", result_text)

    assistant_marker = "\nassistant\n"
    marker_idx = result_text.rfind(assistant_marker)
    if marker_idx != -1:
        assistant_text = result_text[marker_idx + len(assistant_marker):].strip()
    else:
        alt_marker = "<|im_start|>assistant"
        alt_idx = result_text.rfind(alt_marker)
        if alt_idx != -1:
            assistant_text = result_text[alt_idx + len(alt_marker):].strip()
        else:
            assistant_text = result_text

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(result_text)

    detections = None
    parse_failed = False
    try:
        json_candidates = []

        def _add_candidate(text: str):
            if not text:
                return
            candidate = text.strip()
            if not candidate:
                return
            if candidate not in json_candidates:
                json_candidates.append(candidate)

        match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", assistant_text)
        if match:
            _add_candidate(match.group(1))

        _add_candidate(assistant_text)

        array_match = re.search(r"\[\s*\{", assistant_text)
        if array_match:
            array_start = array_match.start()
            array_tail_match = re.search(r"\}\s*\]", assistant_text[array_start:])
            if array_tail_match:
                array_end = array_start + array_tail_match.end()
                _add_candidate(assistant_text[array_start:array_end])

        obj_start = assistant_text.find("{")
        obj_end = assistant_text.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            _add_candidate(assistant_text[obj_start:obj_end + 1])

        parsed_successfully = False
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    pass

            if isinstance(parsed, dict):
                detections = [parsed]
                parsed_successfully = True
                break
            if isinstance(parsed, list):
                detections = parsed
                parsed_successfully = True
                break

        if not parsed_successfully:
            if json_candidates:
                parse_failed = True
            detections = None

        if detections:
            normalized = []
            dropped_any = False
            for d in detections:
                if not isinstance(d, dict):
                    print(f"⚠️ 忽略非字典检测项：{d}")
                    dropped_any = True
                    continue
                if "category" in d and "label" not in d:
                    d["label"] = d.pop("category")
                if "class" in d and "label" not in d:
                    d["label"] = d.pop("class")
                normalized.append(d)
            detections = normalized
            if dropped_any and not detections:
                parse_failed = True
                detections = None

    except Exception as e:
        parse_failed = True
        print("⚠️ JSON 解析失败:", e)
        print("🔍 原始输出片段:", assistant_text[:300])

    if detections is None:
        recovered = recover_partial_detections(assistant_text)
        if recovered:
            detections = recovered
            if parse_failed:
                print(f"⚠️ 已从部分 JSON 中恢复 {len(detections)} 个检测结果")
            else:
                print(f"⚠️ 已从文本中提取 {len(detections)} 个检测结果")
        elif parse_failed:
            print("⚠️ 无法从模型输出中恢复有效检测结果")

    if detections is None:
        print(f"❌ {img_file} 未检测到边界框，请检查模型输出")
        continue

    abs_detections = convert_relative_to_absolute(
        detections,
        image.width,
        image.height,
    )

    with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(abs_detections, f, ensure_ascii=False, indent=2)

    vis_img = draw_bboxes(image, abs_detections)
    vis_img.save(os.path.join(save_dir, "detection.jpg"))

    if abs_detections:
        print(f"✅ 检测结果已保存到 {save_dir}")
    else:
        print(f"⚠️ {img_file} 未解析到检测结果，已保存空结果文件")