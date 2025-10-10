import os
import json
import torch
from PIL import Image
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForImageTextToText
import re


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
    voc_classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    category_list = ", ".join(voc_classes)

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
    try:
        # ① 提取 ```json ... ``` 中的 JSON 内容
        match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", result_text)
        if match:
            json_str = match.group(1).strip()
            detections = json.loads(json_str)
        else:
            # 如果没有代码块标记，则退而求其次查找第一个 [ ] 段
            start = result_text.find("[")
            end = result_text.rfind("]")
            if start != -1 and end != -1:
                json_str = result_text[start:end+1]
                detections = json.loads(json_str)
            else:
                print("⚠️ 未找到有效 JSON 方括号")

    except Exception as e:
        print("⚠️ JSON 解析失败:", e)
        print("🔍 原始输出片段:", result_text[:300])

    # ==============================
    # Step 7: 保存结果与可视化
    # ==============================
    if detections:
        with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(detections, f, ensure_ascii=False, indent=2)

        out_img = os.path.join(save_dir, "detection.jpg")
        vis_img = draw_bboxes(image, detections)
        vis_img.save(out_img)
        print(f"✅ 检测结果已保存到 {save_dir}")
    else:
        print(f"❌ {img_file} 未检测到边界框，请检查模型输出")
