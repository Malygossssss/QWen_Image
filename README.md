# QWen Image Detection Utilities

This repository collects a handful of helper scripts we use to run the
Qwen-VL vision-language model on COCO 2017 and Pascal VOC style datasets,
convert the results into COCO-style JSON, and visualize or inspect the
outputs. The scripts assume you already have access to the
`qwen3-vl-4b-instruct` checkpoint (or a compatible local copy) and the relevant
datasets on disk.

## Environment prerequisites

* Python 3.9+
* PyTorch with CUDA (optional, but strongly recommended)
* `transformers`, `torchvision`, `opencv-python`, `numpy`, `Pillow`
* `pycocotools`, `tqdm`, `pyyaml`

Install the dependencies with pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers opencv-python numpy pillow pycocotools tqdm pyyaml
```

Several scripts expect the datasets to exist at the paths hard-coded near the
top of each file (see the usage sections below). Adjust the constants to match
your environment or override them via command-line switches when supported.

---

## `run_vlm_detect_coco2017.py`

Run the Qwen-VL detector on the COCO 2017 validation split. The script walks
through every image under `<repo>/coco2017/val2017` and writes the structured
results and a visualization per image to `coco_outputs/<image_id>/`.

```bash
python run_vlm_detect_coco2017.py \
  --skip-existing true   # default; skip images that already have result.json
```

* `--skip-existing` (bool, default `true`): when `true` the script skips images
  that already have `result.json` in `coco_outputs`. Set it to `false` to
  reprocess every image and overwrite prior outputs.
* Update `MODEL_PATH` if your checkpoint is stored elsewhere.
* Place the COCO dataset in `<repo>/coco2017` or edit `DATA_ROOT` accordingly.

Each image folder will contain:

* `result.txt`: raw model response
* `result.json`: absolute pixel coordinates for every detection
* `detection.jpg`: visualization with bounding boxes drawn

## `run_vlm_detect_sunrgbd.py`

Run volumetric detections on the SUN RGB-D dataset with Qwen3-VL. The script
discovers the dataset split, constructs a text-and-image prompt for each frame,
parses structured responses, and rewrites the 3D box coordinates into absolute
pixel units before saving them per image under `sunrgbd_outputs/`.

```bash
python run_vlm_detect_sunrgbd.py \
  --skip-existing true   # default; reuse existing results in sunrgbd_outputs
```

Usage notes:

 **Dataset layout** – Place the official SUN RGB-D release inside
  `<repo>/SUNRGBD_DATA/`. The script scans the `SUNRGBD` folder for image files
  (common extensions only) and uses the `SUNRGBDMetaData` folder to load the
  class list when present; otherwise it falls back to a built-in set of indoor
  categories.
* **Model & device** – Update `MODEL_PATH` if your Qwen checkpoint lives
  elsewhere. Inference runs in BF16 with `device_map="auto"`, so it will prefer
  GPUs when available and fall back to CPU otherwise.
* **Prompt & recovery** – For each image the script issues a chat-style prompt
  that instructs the model to reply with JSON detections. It stores the raw
  response as `result.txt`, attempts to parse the JSON directly, and applies a
  regex-based fallback to recover partial objects when the response is noisy.
* **Coordinate handling** – Successful detections are converted from the
  normalized 0–1000 space used in the prompt into pixel space for center
  coordinates and box dimensions while preserving the depth, center_z, heading,
  and optional score fields. Invalid or incomplete entries are skipped with a
  warning.
* **Outputs** – Each image-specific folder contains the filtered detections in
  `result.json` (absolute pixel units) alongside the raw text response
  (`result.txt`). Re-run with `--skip-existing false` to overwrite prior
  results.

## `run_vlm_detect.py`

Equivalent detection pipeline for Pascal VOC style datasets. By default it
scans `/mnt/pub/wyf/workspace/image_identification/Pascal-VOC-2012/valid/images`
for images and writes outputs to `outputs/<image_stem>/`.

```bash
python run_vlm_detect.py
```

To adapt it to another dataset:

1. Change `IMAGE_DIR` to point to your validation images.
2. Update `VOC_CLASSES` if you have a different label set.
3. Optionally edit `MODEL_PATH` or `OUTPUT_ROOT` to match your paths.

As with the COCO script, each image directory stores `result.txt`, `result.json`
(relative coordinates), and a visualization of the detections.

## `evaluate_vlm_coco.py`

Compute COCO metrics (mAP, precision, recall) for detections produced by
`run_vlm_detect_coco2017.py`.

1. Edit the module-level constants at the top of the script to point to your
   COCO data and detection output directories (`DATA_ROOT`, `GT_JSON_PATH`,
   `PRED_ROOT`).
2. Run the evaluation:

   ```bash
   python evaluate_vlm_coco.py
   ```

The evaluator automatically looks for `result.json`, `result_absolute.json`, or
`result.txt` within each image folder. It writes the merged detections to
`pred_coco.json`, prints the COCO summary table, and reports any missing files.

## `evaluate_vlm_Pascal.py`

Converts YOLO-format Pascal VOC annotations into COCO JSON and evaluates the
model outputs stored under `outputs/`.

1. Update `DATA_ROOT` so that `DATA_ROOT/images` and `DATA_ROOT/labels` point to
   your YOLO-style validation split.
2. Ensure `PRED_ROOT` points to the directory produced by `run_vlm_detect.py`.
3. Provide a `data.yaml` (sibling of `DATA_ROOT`) listing the class names if you
   need human-readable labels.
4. Execute the script:

   ```bash
   python evaluate_vlm_Pascal.py
   ```

The script exports `gt_coco.json` (converted ground truth) and `pred_coco.json`
(consolidated predictions) and prints the COCO evaluation summary.

## `draw_coco_annotations.py`

Overlay ground-truth COCO boxes on the validation images to inspect label
quality.

```bash
python draw_coco_annotations.py \
  /path/to/coco2017 \
  --output-dir /tmp/annotated_val2017 \
  --overwrite
```

Arguments:

* `dataset_root` (positional): directory containing `annotations/` and
  `val2017/`.
* `--annotation-file`: optional explicit path to the annotation JSON.
* `--output-dir`: where to store the annotated images. Defaults to
  `<dataset_root>/annotated_val2017`.
* `--overwrite`: replace existing annotated images instead of skipping them.

## `read_coco_image_info.py`

Quick utility to inspect the metadata and ground-truth boxes for a single COCO
image.

```bash
python read_coco_image_info.py \
  /path/to/coco2017 \
  000000000139.jpg \
  --output /tmp/000000000139.json
```

The script prints a JSON blob with the image metadata and a simplified list of
annotations. When `--output` is provided it also writes the JSON to disk.

---

## Troubleshooting tips

* **Model loading errors**: verify the `MODEL_PATH` is reachable and matches the
  checkpoint you intend to use.
* **CUDA out-of-memory**: lower `max_new_tokens` or run the scripts on CPU by
  forcing `DEVICE = "cpu"`.
* **Missing dataset files**: double-check the hard-coded paths near the top of
  each script and ensure the directory structure matches the expectation.