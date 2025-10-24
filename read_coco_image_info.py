"""Utility to extract COCO annotation details for a specific image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load COCO annotations for a single image identified by its file name"
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the COCO 2017 dataset root (contains 'annotations' directory).",
    )
    parser.add_argument(
        "image_name",
        type=str,
        help="Image file name (e.g. '000000000139.jpg') to query in the annotations.",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help=(
            "Optional path to the COCO annotation JSON. Defaults to "
            "<dataset_root>/annotations/instances_val2017.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional file to write the resulting JSON data. When omitted the data "
            "is printed to stdout."
        ),
    )
    return parser.parse_args()


def load_annotations(annotation_path: Path) -> dict:
    with annotation_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_image_info(data: dict, image_name: str) -> dict:
    for image in data.get("images", []):
        if image.get("file_name") == image_name:
            return image
    raise ValueError(f"Image '{image_name}' not found in annotation file")


def collect_annotations(
    data: dict,
    image_id: int,
    category_lookup: Dict[int, str],
) -> List[dict]:
    """Return bbox-centric annotation details for a specific image."""

    result: List[dict] = []
    for ann in data.get("annotations", []):
        if ann.get("image_id") != image_id:
            continue

        bbox = ann.get("bbox")
        if bbox is None:
            continue

        category_id = ann.get("category_id")
        simplified = {"bbox": bbox}

        if category_id is not None:
            simplified["category_id"] = category_id
            simplified["category_name"] = category_lookup.get(
                category_id, str(category_id)
            )

        annotation_id = ann.get("id")
        if annotation_id is not None:
            simplified["annotation_id"] = annotation_id

        result.append(simplified)
    return result


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    annotation_file = (
        args.annotation_file
        if args.annotation_file is not None
        else dataset_root / "annotations" / "instances_val2017.json"
    )
    annotation_file = annotation_file.expanduser().resolve()
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    data = load_annotations(annotation_file)

    category_lookup = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    image_info = find_image_info(data, args.image_name)
    annotations = collect_annotations(data, image_info["id"], category_lookup)

    result = {
        "image": image_info,
        "annotations": annotations,
    }

    output_path = args.output
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()