"""Validate SUNRGBD VLM detection outputs for structural correctness.

The script locates the SUNRGBD test split, enumerates the corresponding
images, and verifies that each entry in ``sunrgbd_outputs`` contains a
parseable ``result.json`` file with detections that follow the expected
schema.
"""

import argparse
import json
import math
import os
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import scipy.io as sio
except ImportError:  # pragma: no cover - optional dependency
    sio = None


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


def _load_sunrgbd_categories(metadata_root: str) -> List[str]:
    """Attempt to load SUNRGBD category names from metadata files."""

    def _parse_category_stream(stream) -> List[str]:
        try:
            raw = stream.read()
        except OSError:
            return []
        if not raw:
            return []
        if isinstance(raw, bytes):
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="ignore")
        else:
            text = str(raw)
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
        return sorted({c for c in categories if c})

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


def _gather_sunrgbd_images(dataset_root: str) -> List[str]:
    image_files: List[str] = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for root_dir, _, files in os.walk(dataset_root):
        if os.path.basename(root_dir) != "image":
            continue
        for file_name in files:
            if os.path.splitext(file_name)[1].lower() in image_exts:
                image_files.append(os.path.join(root_dir, file_name))
    return image_files


def _validate_detection(
    detection: Dict[str, object],
    known_classes: Set[str],
) -> Tuple[bool, Optional[str]]:
    if not isinstance(detection, dict):
        return False, "Detection entry is not a JSON object."

    label = detection.get("label")
    if not isinstance(label, str) or not label:
        return False, "Missing or invalid 'label' field."
    if label not in known_classes:
        return False, f"Unknown label '{label}'."

    bbox = detection.get("bbox_3d")
    if bbox is None:
        bbox = detection.get("bbox3d")
    if bbox is None:
        return False, "Missing 'bbox_3d' field."
    if not isinstance(bbox, (list, tuple)):
        return False, "'bbox_3d' must be a list or tuple."
    if len(bbox) < 7:
        return False, "'bbox_3d' must contain at least 7 numeric values."

    for value in bbox:
        try:
            float(value)
        except (TypeError, ValueError):
            return False, "'bbox_3d' contains a non-numeric value."

    return True, None

def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prediction_to_aabb(detection: Dict[str, object]) -> Optional[Tuple[float, float, float, float, float, float]]:
    bbox = detection.get("bbox_3d")
    if bbox is None:
        bbox = detection.get("bbox3d")
    if bbox is None:
        return None

    if not isinstance(bbox, (list, tuple)) or len(bbox) < 6:
        return None

    values: List[Optional[float]] = [_safe_float(v) for v in bbox[:6]]
    if any(v is None for v in values):
        return None

    center_x, center_y, center_z, width, height, depth = values  # type: ignore[misc]

    if width is None or height is None or depth is None:
        return None

    if width <= 0 or height <= 0 or depth <= 0:
        return None

    # The prompt asks the model to report (x, y, z) centers in meters, where
    # x points to the right, y upward, and z forward from the camera. We
    # convert the detection into an axis-aligned 3D box following the SUNRGBD
    # gravity-aligned coordinate system: (x -> right, y -> forward, z -> up).
    center_right = center_x
    center_up = center_y
    center_forward = center_z

    size_right = width
    size_up = height
    size_forward = depth

    x_min = center_right - size_right / 2.0
    x_max = center_right + size_right / 2.0
    z_min = center_up - size_up / 2.0
    z_max = center_up + size_up / 2.0
    y_min = center_forward - size_forward / 2.0
    y_max = center_forward + size_forward / 2.0

    return (x_min, y_min, z_min, x_max, y_max, z_max)


@dataclass
class GroundTruthBox:
    label: str
    box: Tuple[float, float, float, float, float, float]


@dataclass
class GroundTruthSet:
    by_image: Dict[str, List[GroundTruthBox]]
    class_counts: Dict[str, int]


def _load_ground_truth_boxes(
    metadata_root: str,
    test_image_rel_paths: Iterable[str],
    known_classes: Set[str],
) -> Optional[GroundTruthSet]:
    if sio is None:
        print(
            "ℹ️ SciPy is not installed; skipping ground-truth loading and mAP computation."
        )
        return None

    metadata_path = os.path.join(metadata_root, "SUNRGBDMeta3DBB_v2.mat")
    if not os.path.isfile(metadata_path):
        print(
            "ℹ️ SUNRGBDMeta3DBB_v2.mat was not found. Unable to compute mAP without ground-truth boxes."
        )
        return None

    try:
        mat_data = sio.loadmat(metadata_path)
    except (OSError, NotImplementedError) as exc:
        print(f"ℹ️ Failed to load {metadata_path}: {exc}. Skipping mAP computation.")
        return None

    meta_entries = mat_data.get("SUNRGBDMeta")
    if meta_entries is None or not isinstance(meta_entries, np.ndarray):
        print(
            "ℹ️ Unexpected SUNRGBD metadata structure. Skipping mAP computation."
        )
        return None

    meta_entries = meta_entries[0]
    desired: Set[str] = {path.replace("\\", "/") for path in test_image_rel_paths}

    boxes_by_image: Dict[str, List[GroundTruthBox]] = defaultdict(list)
    class_counts: Dict[str, int] = defaultdict(int)

    for entry in meta_entries:
        try:
            rgb_path = entry["rgbpath"][0]
        except Exception:
            continue

        if not isinstance(rgb_path, str):
            continue

        normalized = rgb_path.replace("\\", "/")
        if "SUNRGBD/" in normalized:
            normalized = normalized.split("SUNRGBD/", 1)[1]

        if normalized not in desired:
            continue

        try:
            gt_boxes = entry["groundtruth3DBB"][0]
        except Exception:
            continue

        relative_path = normalized

        for gt_box in gt_boxes:
            try:
                label = gt_box["classname"][0]
            except Exception:
                continue
            if not isinstance(label, str) or label not in known_classes:
                continue

            try:
                centroid = np.asarray(gt_box["centroid"], dtype=np.float64).reshape(3)
                coeffs = np.asarray(gt_box["coeffs"], dtype=np.float64).reshape(3)
                basis = np.asarray(gt_box["basis"], dtype=np.float64).reshape(3, 3)
            except Exception:
                continue

            corner_offsets = np.array(list(product([-1.0, 1.0], repeat=3)), dtype=np.float64)
            corner_offsets *= coeffs[np.newaxis, :]
            corners = (basis @ corner_offsets.T).T + centroid

            mins = corners.min(axis=0)
            maxs = corners.max(axis=0)
            aabb = (
                float(mins[0]),
                float(mins[1]),
                float(mins[2]),
                float(maxs[0]),
                float(maxs[1]),
                float(maxs[2]),
            )

            boxes_by_image[relative_path].append(GroundTruthBox(label=label, box=aabb))
            class_counts[label] += 1

    if not boxes_by_image:
        print(
            "ℹ️ No overlapping ground-truth annotations were found for the evaluated split. "
            "Skipping mAP computation."
        )
        return None

    return GroundTruthSet(by_image=dict(boxes_by_image), class_counts=dict(class_counts))


def _axis_aligned_iou_3d(
    box_a: Tuple[float, float, float, float, float, float],
    box_b: Tuple[float, float, float, float, float, float],
) -> float:
    ax_min, ay_min, az_min, ax_max, ay_max, az_max = box_a
    bx_min, by_min, bz_min, bx_max, by_max, bz_max = box_b

    inter_x = max(0.0, min(ax_max, bx_max) - max(ax_min, bx_min))
    inter_y = max(0.0, min(ay_max, by_max) - max(ay_min, by_min))
    inter_z = max(0.0, min(az_max, bz_max) - max(az_min, bz_min))
    inter_vol = inter_x * inter_y * inter_z

    if inter_vol <= 0.0:
        return 0.0

    vol_a = max(ax_max - ax_min, 0.0) * max(ay_max - ay_min, 0.0) * max(az_max - az_min, 0.0)
    vol_b = max(bx_max - bx_min, 0.0) * max(by_max - by_min, 0.0) * max(bz_max - bz_min, 0.0)

    if vol_a <= 0.0 or vol_b <= 0.0:
        return 0.0

    return inter_vol / (vol_a + vol_b - inter_vol)


def _compute_average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0 or precisions.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])
    return float(ap)


def _evaluate_map(
    predictions_by_class: Dict[str, List[Dict[str, object]]],
    ground_truth: GroundTruthSet,
    iou_threshold: float,
) -> Tuple[Dict[str, float], float]:
    aps: Dict[str, float] = {}

    all_labels = set(predictions_by_class.keys()) | set(ground_truth.class_counts.keys())

    for label in all_labels:
        preds = predictions_by_class.get(label, [])
        gt_per_image = {
            image_id: [dict(box=box.box, used=False) for box in boxes if box.label == label]
            for image_id, boxes in ground_truth.by_image.items()
        }

        gt_total = sum(len(items) for items in gt_per_image.values())
        if gt_total == 0:
            aps[label] = float("nan")
            continue

        sorted_preds = sorted(
            preds,
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )

        tp = np.zeros(len(sorted_preds), dtype=np.float64)
        fp = np.zeros(len(sorted_preds), dtype=np.float64)

        for idx, pred in enumerate(sorted_preds):
            image_id = pred.get("image_id")
            pred_box = pred.get("box")
            if not isinstance(image_id, str) or pred_box is None:
                fp[idx] = 1.0
                continue

            candidates = gt_per_image.get(image_id)
            if not candidates:
                fp[idx] = 1.0
                continue

            ious = [
                (_axis_aligned_iou_3d(pred_box, candidate["box"]), candidate_index)
                for candidate_index, candidate in enumerate(candidates)
                if not candidate["used"]
            ]

            if not ious:
                fp[idx] = 1.0
                continue

            best_iou, best_index = max(ious, key=lambda pair: pair[0])

            if best_iou >= iou_threshold:
                tp[idx] = 1.0
                candidates[best_index]["used"] = True
            else:
                fp[idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = cum_tp / gt_total
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

        if sorted_preds:
            aps[label] = _compute_average_precision(recalls, precisions)
        else:
            aps[label] = 0.0

    valid_aps = [ap for ap in aps.values() if not math.isnan(ap)]
    mean_ap = float(np.mean(valid_aps)) if valid_aps else float("nan")
    return aps, mean_ap

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Qwen-VL SUNRGBD detection outputs for structural correctness."
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to the SUNRGBD_DATA directory (defaults to <repo>/SUNRGBD_DATA).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Path to the directory containing sunrgbd_outputs (defaults to <repo>/sunrgbd_outputs).",
    )
    parser.add_argument(
        "--metadata-root",
        default=None,
        help="Override the SUNRGBD metadata directory (defaults to <data-root>/SUNRGBDMetaData).",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root or os.path.join(repo_root, "SUNRGBD_DATA")
    output_root = args.output_root or os.path.join(repo_root, "sunrgbd_outputs")
    metadata_root = args.metadata_root or os.path.join(data_root, "SUNRGBDMetaData")
    dataset_root = os.path.join(data_root, "SUNRGBD")

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"SUNRGBD dataset not found at {dataset_root}. Please verify the --data-root argument."
        )
    if not os.path.isdir(output_root):
        raise FileNotFoundError(
            f"SUNRGBD detection outputs not found at {output_root}."
        )

    categories = _load_sunrgbd_categories(metadata_root)
    known_classes = set(categories)
    print(f"✅ Loaded {len(categories)} SUNRGBD categories.")

    split_entries = _load_sunrgbd_split(metadata_root, split="test")
    if not split_entries:
        raise RuntimeError(
            "Unable to locate the SUNRGBD test split metadata. "
            "Ensure the official split files are available in the metadata directory."
        )

    image_files = _gather_sunrgbd_images(dataset_root)
    if not image_files:
        raise RuntimeError(f"No image files found under {dataset_root}.")

    test_images: List[str] = []
    for image_path in image_files:
        relative_path = os.path.relpath(image_path, dataset_root)
        if any(key in split_entries for key in _relative_image_keys(relative_path)):
            test_images.append(image_path)

    if not test_images:
        raise RuntimeError(
            "No dataset images matched the SUNRGBD test split definition. "
            "Please confirm that the dataset layout aligns with the official release."
        )

    print(
        f"🧪 Validating {len(test_images)} SUNRGBD test images against outputs in {output_root}..."
    )

    relative_paths_for_gt = [
        os.path.relpath(path, dataset_root).replace("\\", "/") for path in test_images
    ]
    ground_truth = _load_ground_truth_boxes(metadata_root, relative_paths_for_gt, known_classes)

    missing_outputs: List[str] = []
    invalid_json: List[str] = []
    invalid_detections: List[Tuple[str, int, str]] = []
    total_detections = 0
    total_valid_detections = 0

    filtered_unknown_labels = 0
    unknown_label_examples: List[Tuple[str, int, str]] = []
    unknown_label_counts: Dict[str, int] = {}

    predictions_by_class: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for image_path in sorted(test_images):
        relative_path = os.path.relpath(image_path, dataset_root)
        relative_stem = os.path.splitext(relative_path)[0]
        prediction_dir = os.path.join(output_root, relative_stem)
        result_path = os.path.join(prediction_dir, "result.json")

        if not os.path.exists(result_path):
            missing_outputs.append(relative_path)
            continue

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            invalid_json.append(relative_path)
            continue
        except OSError:
            invalid_json.append(relative_path)
            continue

        if payload is None:
            payload = []
        if not isinstance(payload, list):
            invalid_json.append(relative_path)
            continue

        for idx, det in enumerate(payload):
            total_detections += 1
            label = det.get("label")
            if isinstance(label, str) and label and label not in known_classes:
                filtered_unknown_labels += 1
                if len(unknown_label_examples) < 20:
                    unknown_label_examples.append((relative_path, idx, label))
                unknown_label_counts[label] = unknown_label_counts.get(label, 0) + 1
                continue

            is_valid, reason = _validate_detection(det, known_classes)
            if not is_valid:
                invalid_detections.append((relative_path, idx, reason or "Invalid detection."))
                continue
            total_valid_detections += 1

            if ground_truth is not None:
                aabb = _prediction_to_aabb(det)
                if aabb is None:
                    continue

                score = det.get("score")
                if score is None:
                    score = det.get("confidence")
                if score is None:
                    score = det.get("probability")

                score_value = _safe_float(score)
                if score_value is None:
                    score_value = 1.0

                predictions_by_class[label].append(
                    {
                        "image_id": relative_path.replace("\\", "/"),
                        "score": score_value,
                        "box": aabb,
                    }
                )

    valid_files = len(test_images) - len(missing_outputs) - len(invalid_json)
    print(
        f"📊 Summary: {valid_files} valid prediction files, "
        f"{len(missing_outputs)} missing, {len(invalid_json)} unreadable."
    )
    print(
        f"📦 Detections: {total_valid_detections} valid entries out of {total_detections} candidates."
    )

    if filtered_unknown_labels:
        print(
            "ℹ️ Ignored {} detections whose labels are not in the known SUNRGBD "
            "category list.".format(filtered_unknown_labels)
        )
        summary_items = sorted(
            unknown_label_counts.items(), key=lambda item: item[1], reverse=True
        )
        summary_str = ", ".join(f"{label}: {count}" for label, count in summary_items[:5])
        if summary_str:
            print(f"   • Top unknown labels: {summary_str}")
        if unknown_label_examples:
            print("   • Example ignored detections:")
            for rel_path, idx, label in unknown_label_examples:
                print(f"      - {rel_path} [entry {idx}]: label='{label}'")

    if missing_outputs:
        print("⚠️ Missing result.json for the following test images:")
        for path in missing_outputs[:20]:
            print(f"   - {path}")
        if len(missing_outputs) > 20:
            print(f"   ... and {len(missing_outputs) - 20} more.")

    if invalid_json:
        print("⚠️ Unable to parse result.json for the following test images:")
        for path in invalid_json[:20]:
            print(f"   - {path}")
        if len(invalid_json) > 20:
            print(f"   ... and {len(invalid_json) - 20} more.")

    if invalid_detections:
        print("⚠️ Detected malformed detections:")
        for rel_path, idx, reason in invalid_detections[:20]:
            print(f"   - {rel_path} [entry {idx}]: {reason}")
        if len(invalid_detections) > 20:
            print(f"   ... and {len(invalid_detections) - 20} more.")

    if ground_truth is not None and predictions_by_class:
        thresholds = [0.25, 0.5]
        for threshold in thresholds:
            aps, mean_ap = _evaluate_map(predictions_by_class, ground_truth, threshold)
            print(f"📈 Axis-aligned 3D mAP@{threshold:.2f} (IoU ≥ {threshold:.2f}):")
            for class_name in sorted(known_classes):
                if class_name not in ground_truth.class_counts:
                    continue
                ap_value = aps.get(class_name)
                gt_count = ground_truth.class_counts.get(class_name, 0)
                if ap_value is None or math.isnan(ap_value):
                    ap_str = "--"
                else:
                    ap_str = f"{ap_value:.3f}"
                print(f"   - {class_name:12s}: AP = {ap_str} (gt: {gt_count})")

            if not math.isnan(mean_ap):
                print(f"   ➤ mAP@{threshold:.2f}: {mean_ap:.3f}")
            else:
                print(f"   ➤ mAP@{threshold:.2f}: --")
if __name__ == "__main__":
    main()