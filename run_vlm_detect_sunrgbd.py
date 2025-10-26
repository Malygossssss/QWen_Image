import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import re
from typing import List, Optional
import argparse


_DEF_CATEGORIES = [
    "bed",
    "chair",
    "sofa",
    "table",
    "desk",
    "dresser",
    "night_stand",
    "bookshelf",
    "bathtub",
    "toilet",
    "lamp",
    "cabinet",
    "refrigerator",
    "sink",
    "door",
    "picture",
    "tv",
    "window",
]


_DEF_PROMPT_GUIDANCE = (
    "Return each detection using JSON objects formatted as {\"label\": \"class_name\", "
    "\"bbox_3d\": [center_x, center_y, center_z, width, height, depth, heading]}. "
    "Normalize center_x, center_y, width, and height to the 0-1000 range relative to the image size. "
    "Express center_z and depth in meters and heading in radians."
)


_NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"


def _load_sunrgbd_categories(metadata_root: str) -> List[str]:
    """Attempt to load SUN RGB-D category names from metadata files."""
    candidates = [
        os.path.join(metadata_root, "category_list.txt"),
        os.path.join(metadata_root, "category_list.tsv"),
        os.path.join(metadata_root, "object_list.txt"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                categories = [line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")]
        except OSError:
            continue
        categories = sorted({c for c in categories if c})
        if categories:
            return categories
    return _DEF_CATEGORIES


def _build_recover_pattern(class_names: List[str]) -> re.Pattern:
    class_pattern = "|".join(re.escape(cls) for cls in class_names)
    bbox3d = r"\[\s*{0}(?:\s*,\s*{0}){{5,6}}\s*\]".format(_NUMBER_PATTERN)
    pattern = (
        r"\{[^{}]*?\"(?:label|category|class)\"\s*:\s*\"(?:" + class_pattern + r")\""
        r"[^{}]*?\"bbox_3d\"\s*:\s*" + bbox3d + r"[^{}]*?\}"
    )
    return re.compile(pattern)


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


def _parse_numeric_list(raw_value) -> Optional[List[float]]:
    if isinstance(raw_value, (list, tuple)):
        try:
            return [float(v) for v in raw_value]
        except (TypeError, ValueError):
            return None
    if isinstance(raw_value, str):
        try:
            candidate = json.loads(raw_value)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, (list, tuple)):
            try:
                return [float(v) for v in candidate]
            except (TypeError, ValueError):
                return None
    return None


def convert_relative_to_absolute_3d(detections, image_width, image_height):
    """Convert relative 3D bbox coordinates (0-1000) to pixel coordinates and keep depth info."""
    converted = []
    for det in detections:
        if not isinstance(det, dict):
            print(f"⚠️ 跳过异常检测项：{det}")
            continue

        bbox3d = det.get("bbox_3d")
        label_name = det.get("label") or det.get("category") or det.get("class") or "unknown"

        result = {"label": label_name}

        parsed_bbox3d = _parse_numeric_list(bbox3d) if bbox3d is not None else None
        if parsed_bbox3d and len(parsed_bbox3d) >= 6:
            converted_bbox3d = parsed_bbox3d[:7]
            if len(converted_bbox3d) < 7:
                converted_bbox3d.extend([0.0] * (7 - len(converted_bbox3d)))

            converted_bbox3d[0] = converted_bbox3d[0] / 1000.0 * image_width
            converted_bbox3d[1] = converted_bbox3d[1] / 1000.0 * image_height
            converted_bbox3d[3] = converted_bbox3d[3] / 1000.0 * image_width
            converted_bbox3d[4] = converted_bbox3d[4] / 1000.0 * image_height

            result["bbox_3d"] = converted_bbox3d

        if "score" in det:
            try:
                result["score"] = float(det["score"])
            except (TypeError, ValueError):
                pass

        if "bbox_3d" not in result:
            print(f"⚠️ 跳过缺少有效3D边界框的检测项：{det}")
            continue

        converted.append(result)
    return converted

def _str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value for --skip-existing (true/false).")


parser = argparse.ArgumentParser(description="Run Qwen VL detection on SUNRGBD images.")
parser.add_argument(
    "--skip-existing",
    type=_str_to_bool,
    default=True,
    help="When true, skip images that already have outputs under sunrgbd_outputs (default: true).",
)
args = parser.parse_args()

MODEL_PATH = "/nas_pub_data/models/base/qwen3-vl-4b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = "sunrgbd_outputs"
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
DATA_ROOT = os.path.join(REPO_ROOT, "SUNRGBD_DATA")
SUNRGBD_ROOT = os.path.join(DATA_ROOT, "SUNRGBD")
METADATA_ROOT = os.path.join(DATA_ROOT, "SUNRGBDMetaData")

if not os.path.isdir(SUNRGBD_ROOT):
    raise FileNotFoundError(
        f"Expected SUNRGBD images under {SUNRGBD_ROOT}. Please ensure the dataset is placed correctly."
    )

sunrgbd_classes = _load_sunrgbd_categories(METADATA_ROOT)
_RECOVER_PATTERN = _build_recover_pattern(sunrgbd_classes)
_CLASS_LIST_TEXT = ", ".join(f"'{name}'" for name in sunrgbd_classes)

image_files = []
image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
for root_dir, _, files in os.walk(SUNRGBD_ROOT):
    for file_name in files:
        if os.path.splitext(file_name)[1].lower() in image_exts:
            image_files.append(os.path.join(root_dir, file_name))

if not image_files:
    raise RuntimeError(f"No image files found under {SUNRGBD_ROOT}.")

for image_path in sorted(image_files):
    relative_path = os.path.relpath(image_path, SUNRGBD_ROOT)
    relative_stem = os.path.splitext(relative_path)[0]
    save_dir = os.path.join(OUTPUT_ROOT, relative_stem)
    result_json = os.path.join(save_dir, "result.json")

    if args.skip_existing and os.path.exists(result_json):
        print(f"⏩ 已检测到 {result_json}，跳过 {relative_path}")
        continue

    image = Image.open(image_path).convert("RGB")

    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (
                "You are performing SUN RGB-D 3D object detection. "
                "Locate every instance that belongs to the following categories: "
                + _CLASS_LIST_TEXT
                + ". "
                + _DEF_PROMPT_GUIDANCE
            )}
        ]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=4096)

    result_text = processor.batch_decode(output, skip_special_tokens=True)[0]
    print(f"\n📷 {relative_path} Raw model output:\n", result_text)

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
        print(f"❌ {relative_path} 未检测到边界框，请检查模型输出")
        continue

    abs_detections = convert_relative_to_absolute_3d(
        detections,
        image.width,
        image.height,
    )

    with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(abs_detections, f, ensure_ascii=False, indent=2)
        print(f"✅ 检测结果已保存到 {save_dir}")
    else:
        print(f"⚠️ {relative_path} 未解析到检测结果，已保存空结果文件")