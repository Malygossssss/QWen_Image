import os
import json
import tarfile
import math
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
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
    "Return each detection as a JSON object with SUN RGB-D style fields: "
    "{\"classname\": \"class_name\", \"label\": class_index, \"centroid\": [x, y, z], "
    "\"coeffs\": [sx, sy, sz], \"basis\": [[bx1, bx2, bx3], [by1, by2, by3], [bz1, bz2, bz3]], "
    "\"orientation\": [ox, oy, oz]}. "
    "Centroid coordinates and coeffs (half sizes) should be in meters. "
    "Basis is a 3x3 rotation matrix (row-major) and orientation is the facing unit vector."
)


_NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"


def _load_sunrgbd_categories(metadata_root: str) -> List[str]:
    """Attempt to load SUN RGB-D category names from metadata files or archives."""

    def _parse_category_stream(stream) -> List[str]:
        try:
            raw = stream.read()
        except OSError:
            return []
        if not raw:
            return []
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")
        categories = [
            line.strip().split()[0]
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return sorted({c for c in categories if c})

    def _load_from_file(path: str) -> List[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                categories = [
                    line.strip().split()[0]
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
        except OSError:
            return []
        categories = sorted({c for c in categories if c})
        return categories
        
    candidates = [
        os.path.join(metadata_root, "category_list.txt"),
        os.path.join(metadata_root, "category_list.tsv"),
        os.path.join(metadata_root, "object_list.txt"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        categories = _load_from_file(path)
        if categories:
            return categories
        
    archive_candidates = [
        os.path.join(metadata_root, "sunrgbd_train_test_labels.tar.gz"),
        os.path.join(metadata_root, "train13labels.tgz"),
        os.path.join(metadata_root, "test13labels.tgz"),
    ]
    archive_filenames = {"category_list.txt", "category_list.tsv", "object_list.txt"}

    for archive_path in archive_candidates:
        if not os.path.isfile(archive_path):
            continue
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    base_name = os.path.basename(member.name)
                    if base_name not in archive_filenames:
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    categories = _parse_category_stream(extracted)
                    if categories:
                        return categories
        except (tarfile.TarError, OSError):
            continue

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

def _coerce_float_sequence(value: Union[str, Sequence, None]) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        result: List[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                result.append(float(item))
            elif isinstance(item, str):
                try:
                    result.append(float(item))
                except ValueError:
                    continue
        return result if result else None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            numbers = re.findall(_NUMBER_PATTERN, value)
            return [float(num) for num in numbers] if numbers else None
        else:
            if isinstance(parsed, (list, tuple)):
                return _coerce_float_sequence(parsed)
            if isinstance(parsed, (int, float)):
                return [float(parsed)]
    return None


def _normalize_classname(raw: Union[str, int, float, None], class_names: Sequence[str]) -> Tuple[str, Optional[int]]:
    classname = None
    label_index: Optional[int] = None
    if isinstance(raw, (int, float)):
        candidate = int(raw)
        if 1 <= candidate <= len(class_names):
            classname = class_names[candidate - 1]
            label_index = candidate
        else:
            classname = str(candidate)
            label_index = candidate
    elif isinstance(raw, str):
        stripped = raw.strip()
        if stripped:
            classname = stripped
            for idx, name in enumerate(class_names, start=1):
                if stripped == name or stripped.lower() == name.lower():
                    label_index = idx
                    classname = name
                    break
    if classname is None:
        classname = "unknown"
    if label_index is None:
        try:
            idx = class_names.index(classname) + 1
        except ValueError:
            label_index = None
        else:
            label_index = idx
    return classname, label_index


def _yaw_to_basis_and_orientation(yaw: float) -> Tuple[List[List[float]], List[float]]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    basis = [
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ]
    orientation = [-s, -c, 0.0]
    return basis, orientation


def _orientation_to_yaw(orientation: Sequence[float]) -> Optional[float]:
    if len(orientation) < 2:
        return None
    ox, oy = orientation[0], orientation[1]
    if abs(ox) < 1e-8 and abs(oy) < 1e-8:
        return None
    return math.atan2(-ox, -oy)


def _convert_detection_to_sunrgbd(
    detection: Dict[str, object], class_names: Sequence[str]
) -> Dict[str, object]:
    raw_class = (
        detection.get("classname")
        or detection.get("class")
        or detection.get("category")
        or detection.get("label")
    )
    classname, label_index = _normalize_classname(raw_class, class_names)

    bbox = detection.get("bbox_3d") or detection.get("bbox3d") or detection.get("box3d")
    bbox_values = _coerce_float_sequence(bbox)

    centroid = _coerce_float_sequence(
        detection.get("centroid")
        or detection.get("center")
        or detection.get("center_3d")
        or (bbox_values[:3] if bbox_values and len(bbox_values) >= 3 else None)
    )
    if centroid is None:
        centroid = [0.0, 0.0, 0.0]
    else:
        centroid = centroid[:3] + [0.0] * max(0, 3 - len(centroid))

    coeffs = _coerce_float_sequence(detection.get("coeffs") or detection.get("sizes") or detection.get("dimensions") or detection.get("size"))
    heading: Optional[float] = None
    orientation = _coerce_float_sequence(detection.get("orientation"))
    basis_values = detection.get("basis")

    if coeffs is None and bbox_values and len(bbox_values) >= 6:
        dims = bbox_values[3:6]
        coeffs = [d / 2.0 for d in dims]
        if len(bbox_values) >= 7:
            heading = float(bbox_values[6])
        if len(bbox_values) >= 9 and orientation is None:
            orientation = bbox_values[6:9]

    if coeffs is None:
        coeffs = [0.5, 0.5, 0.5]
    else:
        coeffs = coeffs[:3] + [0.5] * max(0, 3 - len(coeffs))

    if orientation is not None and len(orientation) >= 3:
        orientation = orientation[:3] + [0.0] * max(0, 3 - len(orientation))
        if heading is None:
            inferred_yaw = _orientation_to_yaw(orientation)
            if inferred_yaw is not None:
                heading = inferred_yaw

    basis: Optional[List[List[float]]] = None
    if isinstance(basis_values, (list, tuple)) and len(basis_values) == 3:
        rows: List[List[float]] = []
        for row in basis_values:  # type: ignore[arg-type]
            seq = _coerce_float_sequence(row)
            if seq is None:
                rows = []
                break
            rows.append((seq + [0.0, 0.0, 0.0])[:3])
        if len(rows) == 3:
            basis = rows

    if basis is None:
        if heading is None:
            heading = 0.0
        basis, default_orientation = _yaw_to_basis_and_orientation(heading)
        if orientation is None:
            orientation = default_orientation

    if orientation is None:
        orientation = [0.0, -1.0, 0.0]

    result: Dict[str, object] = {
        "classname": classname,
        "label": label_index if label_index is not None else classname,
        "centroid": [float(x) for x in centroid[:3]],
        "coeffs": [float(x) for x in coeffs[:3]],
        "basis": basis,
        "orientation": [float(x) for x in orientation[:3]],
    }

    if "score" in detection:
        result["score"] = detection["score"]
    elif "confidence" in detection:
        result["score"] = detection["confidence"]

    for key in ("sequenceName", "sample_id", "id"):
        if key in detection and key not in result:
            result[key] = detection[key]

    return result

def _normalize_split_entry(entry: str) -> Optional[str]:
    if not entry:
        return None
    normalized = entry.strip().replace("\\", "/")
    if not normalized:
        return None
    normalized = normalized.lstrip("./")
    if normalized.startswith("SUNRGBD/"):
        normalized = normalized[len("SUNRGBD/") :]
    return normalized or None


def _parse_split_stream(stream) -> Set[str]:
    try:
        raw = stream.read()
    except OSError:
        return set()
    if not raw:
        return set()
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")
    else:
        text = str(raw)

    entries: Set[str] = set()
    for line in text.splitlines():
        candidate = _normalize_split_entry(line)
        if not candidate:
            continue
        if not (
            candidate.lower().startswith(("kv1/", "kv2/", "realsense/", "xtion/"))
            or "/image" in candidate.lower()
        ):
            continue
        entries.add(candidate)
        entries.add(candidate.rstrip("/"))
        if candidate.endswith(".jpg"):
            entries.add(candidate[:-4])
        if not candidate.startswith("SUNRGBD/"):
            entries.add("SUNRGBD/" + candidate)
    return entries


def _load_sunrgbd_split(metadata_root: str, split: str = "test") -> Optional[Set[str]]:
    split = split.lower()

    def _load_from_path(path: str) -> Set[str]:
        try:
            with open(path, "rb") as f:
                entries = _parse_split_stream(f)
        except OSError:
            return set()
        if entries:
            print(f"✅ Loaded {len(entries)} entries from split file: {path}")
        return entries

    candidate_files = [
        os.path.join(metadata_root, f"sunrgbd_{split}.txt"),
        os.path.join(metadata_root, f"sunrgbd_{split}_list.txt"),
        os.path.join(metadata_root, f"{split}_data_list.txt"),
        os.path.join(metadata_root, f"{split}.txt"),
    ]

    for path in candidate_files:
        if os.path.isfile(path):
            entries = _load_from_path(path)
            if entries:
                return entries

    for root_dir, _, files in os.walk(metadata_root):
        for file_name in files:
            lower = file_name.lower()
            if split not in lower:
                continue
            if not lower.endswith((".txt", ".lst", ".csv")):
                continue
            path = os.path.join(root_dir, file_name)
            entries = _load_from_path(path)
            if entries:
                return entries

    archive_candidates = [
        os.path.join(metadata_root, "sunrgbd_train_test_labels.tar.gz"),
        os.path.join(metadata_root, "train13labels.tgz"),
        os.path.join(metadata_root, "test13labels.tgz"),
    ]

    for archive_path in archive_candidates:
        if not os.path.isfile(archive_path):
            continue
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    base_name = os.path.basename(member.name).lower()
                    if split not in base_name:
                        continue
                    if not base_name.endswith((".txt", ".lst", ".csv")):
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    entries = _parse_split_stream(extracted)
                    if entries:
                        print(
                            "✅ Loaded {} entries from archive split file: {}".format(
                                len(entries), member.name
                            )
                        )
                        return entries
        except (tarfile.TarError, OSError):
            continue

    print(
        f"⚠️ Unable to locate SUNRGBD '{split}' split definition under {metadata_root}."
    )
    return None


def _relative_image_keys(relative_path: str) -> Sequence[str]:
    normalized = relative_path.replace("\\", "/").lstrip("./")
    keys = {normalized}
    if normalized.startswith("SUNRGBD/"):
        keys.add(normalized[len("SUNRGBD/") :])
    else:
        keys.add("SUNRGBD/" + normalized)
    stem = os.path.splitext(normalized)[0]
    keys.add(stem)
    if stem.startswith("SUNRGBD/"):
        keys.add(stem[len("SUNRGBD/") :])
    dir_path = os.path.dirname(normalized)
    keys.add(dir_path)
    if dir_path.startswith("SUNRGBD/"):
        keys.add(dir_path[len("SUNRGBD/") :])
    parent = os.path.dirname(dir_path)
    if parent:
        keys.add(parent)
        if parent.startswith("SUNRGBD/"):
            keys.add(parent[len("SUNRGBD/") :])
    grand_parent = os.path.dirname(parent)
    if grand_parent:
        keys.add(grand_parent)
        if grand_parent.startswith("SUNRGBD/"):
            keys.add(grand_parent[len("SUNRGBD/") :])
    return list(keys)

def recover_partial_detections(raw_text: str):
    """Recover detection results from partial JSON fragments."""
    recovered = []
    for match in _RECOVER_PATTERN.finditer(raw_text):
        try:
            recovered.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return recovered if recovered else None

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
    if os.path.basename(root_dir) != "image":
        continue
    for file_name in files:
        if os.path.splitext(file_name)[1].lower() in image_exts:
            image_files.append(os.path.join(root_dir, file_name))

if not image_files:
    raise RuntimeError(f"No image files found under {SUNRGBD_ROOT}.")

test_split = _load_sunrgbd_split(METADATA_ROOT, split="test")
if test_split:
    before_count = len(image_files)
    filtered_files = []
    for path in image_files:
        relative_path = os.path.relpath(path, SUNRGBD_ROOT)
        keys = _relative_image_keys(relative_path)
        if any(key in test_split for key in keys):
            filtered_files.append(path)
    image_files = filtered_files
    if not image_files:
        raise RuntimeError(
            "Located SUNRGBD test split but none of the dataset images matched it. "
            "Please verify the dataset structure."
        )
    removed = before_count - len(image_files)
    print(
        f"🧪 Filtering to SUNRGBD test split: {len(image_files)} images (removed {removed})."
    )
else:
    raise RuntimeError(
        "Unable to locate the SUNRGBD test split definition. "
        "Please ensure the official split files are available under the metadata directory."
    )

sorted_image_files = sorted(image_files)
total_images = len(sorted_image_files)

for index, image_path in enumerate(sorted_image_files, start=1):
    relative_path = os.path.relpath(image_path, SUNRGBD_ROOT)
    relative_stem = os.path.splitext(relative_path)[0]
    save_dir = os.path.join(OUTPUT_ROOT, relative_stem)
    result_json = os.path.join(save_dir, "result.json")

    if args.skip_existing and os.path.exists(result_json):
        remaining = total_images - index
        print(
            f"⏩ 已检测到 {result_json}，跳过 {relative_path}。还剩 {remaining} 张待处理"
        )
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
    output = model.generate(**inputs, max_new_tokens=1024)

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
            filtered = []
            dropped_any = False
            for d in detections:
                if not isinstance(d, dict):
                    print(f"⚠️ 忽略非字典检测项：{d}")
                    dropped_any = True
                    continue
                filtered.append(d)
            detections = filtered
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

    result_path = os.path.join(save_dir, "result.json")

    if detections is None:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        image.close()
        remaining = total_images - index
        print(
            f"⚠️ {relative_path} 未解析到检测结果，已保存空结果文件。还剩 {remaining} 张待处理"
        )
        continue

    formatted_detections = []
    for det in detections:
        if not isinstance(det, dict):
            print(f"⚠️ 忽略非字典检测项：{det}")
            continue
        try:
            converted = _convert_detection_to_sunrgbd(det, sunrgbd_classes)
        except Exception as exc:
            print(f"⚠️ 转换检测结果到 SUNRGBD 格式时出错: {exc}")
            continue
        formatted_detections.append(converted)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(formatted_detections, f, ensure_ascii=False, indent=2)
    print(f"✅ 检测结果已保存到 {save_dir}")
    image.close()
    remaining = total_images - index
    print(f"📊 剩余待处理图像数量：{remaining}")