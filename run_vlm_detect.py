import os
import json
import torch
from PIL import Image
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForImageTextToText
import re


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

_CLASS_PATTERN = "|".join(re.escape(cls) for cls in VOC_CLASSES)
_RECOVER_PATTERN = re.compile(
    r"\{\s*\"bbox_2d\"\s*:\s*\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]\s*,\s*"
    r"\"label\"\s*:\s*\"(?:" + _CLASS_PATTERN + r")\"\s*\}"
)


def recover_partial_detections(raw_text: str):
    """从可能被截断的 JSON 片段中恢复检测结果。"""
    recovered = []
    for match in _RECOVER_PATTERN.finditer(raw_text):
        try:
            recovered.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return recovered if recovered else None


def draw_bboxes(image, detections):
    """
    在输入 PIL.Image 上画出目标检测框和标签
    detections: [{"bbox_2d": [x1, y1, x2, y2], "label": "xxx"}, ...]
    """
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox_2d"])  # 坐标取整
        label = det["label"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ==============================
# Step 1: 加载模型
# ==============================
MODEL_PATH = "/nas_pub_data/models/base/qwen2.5-vl-3b-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = "outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print(f"Loading model from {MODEL_PATH} on {DEVICE}...")

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True
).eval()


# ==============================
# Step 2: 遍历目录中的所有图片
# ==============================
IMAGE_DIR = "/mnt/pub/wyf/workspace/image_identification/Pascal-VOC-2012/valid/images"
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_file in image_files:
    IMAGE_PATH = os.path.join(IMAGE_DIR, img_file)
    image = Image.open(IMAGE_PATH).convert("RGB")

    # ==============================
    # Step 3: 构造 Prompt
    # ==============================
    category_list = ", ".join(VOC_CLASSES)

    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (
                f"Please detect all objects in this image, but only from the following 20 Pascal VOC categories:\n"
                f"[{category_list}].\n"
                f"For each detected object, return a JSON list where each item is:\n"
                f'{{\"bbox_2d\":[x1,y1,x2,y2], \"label\":\"<one of the above names>\"}}.\n'
                f"Make sure category names strictly match the list above (e.g., use 'sofa' instead of 'couch', 'tvmonitor' instead of 'tv').\n"
                f"Use integer pixel coordinates for bounding boxes.\n"
                f"Return only the JSON array without Markdown formatting or code block syntax."
            )}
        ]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # ==============================
    # Step 4: 模型推理
    # ==============================
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs, max_new_tokens=512)

    result_text = processor.batch_decode(output, skip_special_tokens=True)[0]
    print(f"\n📷 {img_file} Raw model output:\n", result_text)

    # 仅保留最后一次 assistant 的回答内容，避免 prompt 干扰 JSON 解析
    assistant_marker = "\nassistant\n"
    marker_idx = result_text.rfind(assistant_marker)
    if marker_idx != -1:
        assistant_text = result_text[marker_idx + len(assistant_marker):].strip()
    else:
        # 有些模型使用 <|im_start|>assistant 作为分隔符
        alt_marker = "<|im_start|>assistant"
        alt_idx = result_text.rfind(alt_marker)
        if alt_idx != -1:
            assistant_text = result_text[alt_idx + len(alt_marker):].strip()
        else:
            assistant_text = result_text

    # ==============================
    # Step 5: 创建输出目录
    # ==============================
    img_name = os.path.splitext(img_file)[0]
    save_dir = os.path.join(OUTPUT_ROOT, img_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(result_text)

    # ==============================
    # Step 6: 尝试解析 JSON 并保存
    # ==============================
    detections = None
    parse_failed = False
    try:
        # ① 提取 ```json ... ``` 中的 JSON 内容
        match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", assistant_text)
        if match:
            json_str = match.group(1).strip()
            detections = json.loads(json_str)
        else:
            # 如果没有代码块标记，则退而求其次查找第一个 [ ] 段
            start = assistant_text.find("[")
            end = assistant_text.rfind("]")
            if start != -1 and end != -1:
                json_str = assistant_text[start:end+1]
                detections = json.loads(json_str)
            else:
                parse_failed = True
                print("⚠️ 未找到有效 JSON 方括号")

        # 将字符串或字典形式的结果转换为列表，便于后续处理
        if isinstance(detections, str):
            detections = json.loads(detections)
        if isinstance(detections, dict):
            detections = [detections]

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

    # ==============================
    # Step 7: 保存结果与可视化
    # ==============================
    if detections is None:
        print(f"❌ {img_file} 未检测到边界框，请检查模型输出")
        continue

    with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)

    out_img = os.path.join(save_dir, "detection.jpg")
    vis_img = draw_bboxes(image, detections)
    vis_img.save(out_img)

    if detections:
        print(f"✅ 检测结果已保存到 {save_dir}")
    else:
        print(f"⚠️ {img_file} 未解析到检测结果，已保存空结果文件")
