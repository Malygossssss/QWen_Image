"""Validate SUNRGBD VLM detection outputs for structural correctness.

The script locates the SUNRGBD test split, enumerates the corresponding
images, and verifies that each entry in ``sunrgbd_outputs`` contains a
parseable ``result.json`` file with detections that follow the expected
schema.
"""

import argparse
import json
import os
import tarfile
from typing import Dict, List, Optional, Sequence, Set, Tuple


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

    missing_outputs: List[str] = []
    invalid_json: List[str] = []
    invalid_detections: List[Tuple[str, int, str]] = []
    total_detections = 0
    total_valid_detections = 0

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

        total_detections += len(payload)
        for idx, det in enumerate(payload):
            is_valid, reason = _validate_detection(det, known_classes)
            if not is_valid:
                invalid_detections.append((relative_path, idx, reason or "Invalid detection."))
                continue
            total_valid_detections += 1

    valid_files = len(test_images) - len(missing_outputs) - len(invalid_json)
    print(
        f"📊 Summary: {valid_files} valid prediction files, "
        f"{len(missing_outputs)} missing, {len(invalid_json)} unreadable."
    )
    print(
        f"📦 Detections: {total_valid_detections} valid entries out of {total_detections} candidates."
    )

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
if __name__ == "__main__":
    main()