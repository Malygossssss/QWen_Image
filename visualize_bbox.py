import cv2
import json
import numpy as np
from pathlib import Path

def visualize_bboxes(image_path, bbox_json_path, output_path=None):
    """
    在图片上可视化bounding boxes
    
    参数:
        image_path: 图片路径
        bbox_json_path: bbox的JSON文件路径
        output_path: 输出图片路径（可选，默认为原图名_annotated.jpg）
    
    返回:
        annotated_image: 标注后的图片
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 获取图片真实尺寸
    img_height, img_width = image.shape[:2]
    
    # 读取bbox JSON文件
    with open(bbox_json_path, 'r', encoding='utf-8') as f:
        bboxes = json.load(f)
    
    # 为不同类别定义颜色（BGR格式）
    colors = {
        'person': (0, 255, 0),      # 绿色
        'bird': (255, 0, 0),        # 蓝色
        'boat': (0, 0, 255),        # 红色
        'car': (255, 255, 0),       # 青色
        'default': (255, 165, 0)    # 橙色
    }
    
    # 绘制每个bbox
    for bbox_info in bboxes:
        bbox = bbox_info['bbox_2d']
        label = bbox_info['label']
        
        # 将相对坐标转换为绝对坐标
        # 假设相对坐标的范围是0-1000
        x1_rel, y1_rel, x2_rel, y2_rel = bbox
        x1 = int(x1_rel / 1000 * img_width)
        y1 = int(y1_rel / 1000 * img_height)
        x2 = int(x2_rel / 1000 * img_width)
        y2 = int(y2_rel / 1000 * img_height)
        
        # 选择颜色
        color = colors.get(label, colors['default'])
        
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label_text = label
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            image, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width + 10, y1), 
            color, 
            -1
        )
        
        # 绘制标签文字
        cv2.putText(
            image, 
            label_text, 
            (x1 + 5, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
    
    # 保存结果
    if output_path is None:
        input_path = Path(image_path)
        output_dir = Path("visual_bbox_output")
        output_dir.mkdir(exist_ok=True)  # 创建输出文件夹
        output_path = str(output_dir / f"{input_path.name}")
    
    cv2.imwrite(output_path, image)
    print(f"标注后的图片已保存至: {output_path}")
    
    return image


def visualize_bboxes_from_dict(image_path, bbox_list, output_path=None):
    """
    直接从bbox列表（字典）可视化，无需JSON文件
    
    参数:
        image_path: 图片路径
        bbox_list: bbox列表，格式如：
                   [{"bbox_2d": [x1, y1, x2, y2], "label": "person"}, ...]
        output_path: 输出图片路径（可选）
    
    返回:
        annotated_image: 标注后的图片
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 获取图片真实尺寸
    img_height, img_width = image.shape[:2]
    
    # 为不同类别定义颜色
    colors = {
        'person': (0, 255, 0),
        'bird': (255, 0, 0),
        'boat': (0, 0, 255),
        'car': (255, 255, 0),
        'default': (255, 165, 0)
    }
    
    # 绘制每个bbox
    for bbox_info in bbox_list:
        bbox = bbox_info['bbox_2d']
        label = bbox_info['label']
        
        # 将相对坐标转换为绝对坐标
        # 假设相对坐标的范围是0-1000
        x1_rel, y1_rel, x2_rel, y2_rel = bbox
        x1 = int(x1_rel / 1000 * img_width)
        y1 = int(y1_rel / 1000 * img_height)
        x2 = int(x2_rel / 1000 * img_width)
        y2 = int(y2_rel / 1000 * img_height)
        
        color = colors.get(label, colors['default'])
        
        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label_text = label
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            image, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width + 10, y1), 
            color, 
            -1
        )
        cv2.putText(
            image, 
            label_text, 
            (x1 + 5, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
    
    # 保存结果
    if output_path is None:
        input_path = Path(image_path)
        output_dir = Path("visual_bbox_output")
        output_dir.mkdir(exist_ok=True)  # 创建输出文件夹
        output_path = str(output_dir / f"{input_path.name}")
    
    cv2.imwrite(output_path, image)
    print(f"标注后的图片已保存至: {output_path}")
    
    return image


# 使用示例
if __name__ == "__main__":
    # 方法1: 从JSON文件读取
    image_path = "/mnt/pub/wyf/workspace/image_identification/Pascal-VOC-2012/valid/images/2012_003569_jpg.rf.783887c5e82cdecf095af83a49c4eb77.jpg"
    # bbox_json_path = "bboxes.json"
    # visualize_bboxes(image_path, bbox_json_path)
    
    # 方法2: 直接传入bbox列表
    bbox_list = [
        {"bbox_2d": [368, 274, 671, 985], "label": "person"},
        {"bbox_2d": [360, 214, 547, 545], "label": "person"},
        {"bbox_2d": [396, 714, 544, 990], "label": "chair"}
    ]
    visualize_bboxes_from_dict(image_path, bbox_list)