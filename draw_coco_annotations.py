"""Utility to overlay COCO 2017 annotations on the validation images.

This script expects the COCO 2017 directory layout::

    coco2017
    ├── annotations
    │   └── instances_val2017.json
    └── val2017

It loads the bounding boxes and category labels from the annotation JSON
file and draws them onto each corresponding image in ``val2017``. The
annotated images are written to a new folder under the provided dataset
root (``annotated_val2017`` by default).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay bounding boxes and labels from COCO 2017 annotations "
            "onto the validation images."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help=(
            "Path to the COCO 2017 dataset root containing the 'annotations' "
            "and 'val2017' folders."
        ),
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
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to store annotated images. Defaults to a folder named "
            "'annotated_val2017' inside the dataset root."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotated images if they already exist.",
    )
    return parser.parse_args()


def load_coco_annotations(
    annotation_path: Path,
) -> tuple[Dict[int, str], Dict[int, List[dict]], Dict[int, str]]:
    """Load COCO annotations.

    Returns:
        A tuple of ``(category_lookup, image_annotations, image_id_to_file)``.
    """

    with annotation_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    category_lookup = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    image_annotations: Dict[int, List[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        image_annotations[ann["image_id"]].append(ann)

    image_id_to_file = {
        image_info["id"]: image_info["file_name"] for image_info in data.get("images", [])
    }

    return category_lookup, image_annotations, image_id_to_file


def compute_color(category_id: int) -> tuple[int, int, int]:
    """Compute a deterministic BGR color for a given category id."""

    rng = np.random.default_rng(category_id)
    color = rng.integers(0, 256, size=3, dtype=np.uint8)
    # OpenCV expects BGR order
    return int(color[2]), int(color[1]), int(color[0])


def annotate_image(
    image_path: Path,
    annotations: Iterable[dict],
    category_lookup: Dict[int, str],
) -> np.ndarray:
    """Return an image with annotations drawn on top."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x, y, width, height = bbox
        x1 = int(max(0, np.floor(x)))
        y1 = int(max(0, np.floor(y)))
        x2 = int(min(image.shape[1], np.ceil(x + width)))
        y2 = int(min(image.shape[0], np.ceil(y + height)))

        if x2 <= x1 or y2 <= y1:
            continue

        top_left = (x1, y1)
        bottom_right = (x2, y2)
        category_id = ann.get("category_id")
        label = category_lookup.get(category_id, str(category_id))

        color = compute_color(category_id)
        cv2.rectangle(image, top_left, bottom_right, color, thickness=2)

        caption = f"{label}"
        score = ann.get("score")
        if score is not None:
            caption += f" {score:.2f}"

        text_size, _baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_width, text_height = text_size

        text_bg_left = int(max(0, min(top_left[0], image.shape[1] - text_width - 6)))
        text_bg_right = int(min(image.shape[1], text_bg_left + text_width + 6))

        text_bg_top = int(max(0, top_left[1] - text_height - 8))
        text_bg_bottom = int(min(image.shape[0], top_left[1] - 2))

        if text_bg_bottom <= text_bg_top:
            text_bg_top = int(
                max(0, min(image.shape[0] - text_height - 6, top_left[1] + 2))
            )
            text_bg_bottom = int(min(image.shape[0], text_bg_top + text_height + 6))

        cv2.rectangle(
            image,
            (text_bg_left, text_bg_top),
            (text_bg_right, text_bg_bottom),
            color,
            thickness=-1,
        )
        text_position = (
            text_bg_left + 3,
            text_bg_bottom - 3,
        )
        cv2.putText(
            image,
            caption,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return image


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

    val_dir = dataset_root / "val2017"
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else dataset_root / "annotated_val2017"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    category_lookup, image_annotations, image_id_to_file = load_coco_annotations(
        annotation_file
    )

    total_images = len(image_id_to_file)
    print(
        f"Found {total_images} images with annotations. Writing results to {output_dir}."
    )

    sorted_images = sorted(image_id_to_file.items(), key=lambda item: item[1])
    for idx, (image_id, file_name) in enumerate(sorted_images, start=1):
        image_path = val_dir / file_name
        if not image_path.exists():
            print(f"[WARN] Missing image file for id {image_id}: {image_path}")
            continue

        output_path = output_dir / file_name
        if output_path.exists() and not args.overwrite:
            continue

        annotations = image_annotations.get(image_id, [])
        if not annotations:
            # No annotations; copy the original image for completeness.
            try:
                shutil.copy2(image_path, output_path)
            except OSError as exc:
                print(f"[WARN] Failed to copy image {image_path}: {exc}")
            continue

        annotated_image = annotate_image(image_path, annotations, category_lookup)
        cv2.imwrite(str(output_path), annotated_image)

        if idx % 100 == 0:
            print(f"Processed {idx}/{total_images} images...")

    print("Annotation visualization complete.")


if __name__ == "__main__":
    main()