import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import json
import os
from itertools import product

# === 1. 读取mat文件 ===
meta = sio.loadmat('SUNRGBD_DATA/SUNRGBDMetaData/SUNRGBDMeta3DBB_v2.mat')
sample = meta['SUNRGBDMeta'][0][0] # 示例

K = sample['K']  # 相机内参矩阵
rgb_path = sample['rgbpath'][0]     # RGB图片路径
Rtilt = np.asarray(sample['Rtilt'], dtype=np.float64)

# === 替换路径前缀 ===
# 旧的前缀（原始数据作者电脑）
old_prefix = "/n/fs/sun3d/data/SUNRGBD/"
# 你的本地前缀
new_prefix = "SUNRGBD_DATA/SUNRGBD/"

# 替换路径
rgb_path = rgb_path.replace(old_prefix, new_prefix)

# 检查路径是否存在
if not os.path.exists(rgb_path):
    print("❌ 图像路径不存在:", rgb_path)
else:
    print("✅ 找到图像:", rgb_path)
    img = cv2.imread(rgb_path)

bboxes = sample['groundtruth3DBB'][0]  # 3D框数组
print(bboxes.dtype.names)

corner_signs = np.array(list(product([-1, 1], repeat=3)), dtype=np.float64)

def sunrgbd_to_camera(points: np.ndarray) -> np.ndarray:
    """Convert SUNRGBD gravity-aligned coordinates to the camera frame.

    In the SUNRGBD annotations the axes are defined as:
        - x points to the right of the camera
        - y points forward from the camera (depth)
        - z points upward (aligned with gravity)

    The OpenCV camera frame used for projection expects:
        - x pointing right
        - y pointing down
        - z pointing forward

    Therefore we keep the x component, map the forward (y) axis to the
    camera z axis, and flip the sign of the vertical axis so that positive
    values point downwards in image space.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be of shape (N, 3)")

    x = points[:, 0]
    y_forward = points[:, 1]
    z_up = points[:, 2]

    x_cam = x
    y_cam = -z_up
    z_cam = y_forward

    return np.stack([x_cam, y_cam, z_cam], axis=1)

# === 2. 打开RGB图像 ===
img = cv2.imread(rgb_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === 3. 定义投影函数 ===
def project_points(points_3d, K):
    """将3D点投影到图像平面"""
    pts_2d = (K @ points_3d.T).T
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, :2]

# === 4. 遍历每个3D框 ===
for bb in bboxes:
    centroid = np.asarray(bb['centroid'], dtype=np.float64).reshape(3)
    coeffs = np.asarray(bb['coeffs'], dtype=np.float64).reshape(3)
    basis = np.asarray(bb['basis'], dtype=np.float64)  # 3x3，每列是一个方向的单位向量

    # 构造 8 个顶点：coeffs 是每个方向的半轴长度，basis 的列向量是对应的单位方向
    # (3,8) => 每列是一个 corner 的局部坐标偏移
    corner_offsets = (corner_signs.T * coeffs[:, None])
    corners_world = (basis @ corner_offsets).T + centroid

    # 将 SUNRGBD 的坐标轴映射到 OpenCV 相机坐标系
    # 先使用 Rtilt 将重力对齐坐标系旋转到相机坐标系
    corners_camera = (Rtilt.T @ corners_world.T).T

    # 再将 SUNRGBD 的相机坐标映射到 OpenCV 坐标轴约定
    corners_camera = sunrgbd_to_camera(corners_camera)

    # 如果框被裁剪在摄像机后面，则跳过，避免投影出现极端值
    # if np.any(corners_camera[:, 2] <= 1e-3):
    #     continue

    # 投影到2D
    corners_img = project_points(corners_camera, K)

    # 使用符号组合自动找出需要连接的棱
    corners_int = np.round(corners_img).astype(int)
    for i in range(len(corner_signs)):
        for j in range(i + 1, len(corner_signs)):
            # 两个顶点符号只有一个维度不同 => 一条棱
            if np.sum(np.abs(corner_signs[i] - corner_signs[j])) == 2:
                cv2.line(img, tuple(corners_int[i]), tuple(corners_int[j]), (255, 0, 0), 2)

# === 5. 叠加预测结果（如有） ===
relative_rgb = os.path.relpath(rgb_path, new_prefix)
pred_json_path = os.path.join(
    'sunrgbd_outputs', os.path.splitext(relative_rgb)[0], 'result.json'
)

if os.path.exists(pred_json_path):
    print("✅ 找到预测结果:", pred_json_path)
    try:
        with open(pred_json_path, 'r', encoding='utf-8') as f:
            preds = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print("⚠️ 无法读取预测结果:", exc)
        preds = []

    if isinstance(preds, list):
        for pred in preds:
            box = pred.get('bbox_3d') or pred.get('bbox3d')
            if not box or len(box) < 6:
                continue

            try:
                center = np.asarray(box[:3], dtype=np.float64).reshape(3)
                lengths = np.asarray(box[3:6], dtype=np.float64).reshape(3)
            except (ValueError, TypeError):
                continue

            if np.any(lengths <= 0):
                continue

            # 预测框按照与真值相同的方式投影
            corner_offsets = (corner_signs * (lengths / 2.0))
            corners_world = corner_offsets + center

            corners_camera = (Rtilt.T @ corners_world.T).T
            corners_camera = sunrgbd_to_camera(corners_camera)
            corners_img = project_points(corners_camera, K)
            corners_int = np.round(corners_img).astype(int)

            for i in range(len(corner_signs)):
                for j in range(i + 1, len(corner_signs)):
                    if np.sum(np.abs(corner_signs[i] - corner_signs[j])) == 2:
                        cv2.line(img, tuple(corners_int[i]), tuple(corners_int[j]), (0, 255, 0), 2)
else:
    print("ℹ️ 未找到对应的预测 result.json 文件:", pred_json_path)

# === 6. 显示结果 ===
plt.imshow(img)
plt.axis('off')

output_dir = os.path.join(os.path.dirname(__file__), "visual_3bbox_output")
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "sunrgbd_vis_3dbbox.jpg")
plt.imsave(save_path, img)
print(f"✅ 可视化结果已保存到: {save_path}")
