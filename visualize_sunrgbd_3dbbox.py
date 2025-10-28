import scipy.io as sio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from itertools import product

# === 1. 读取mat文件 ===
meta = sio.loadmat('SUNRGBD_DATA/SUNRGBDMetaData/SUNRGBDMeta3DBB_v2.mat')
sample = meta['SUNRGBDMeta'][0][0]  # 示例取第一张

K = sample['K']  # 相机内参矩阵
rgb_path = sample['rgbpath'][0]     # RGB图片路径

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

R_tilt = np.asarray(sample['Rtilt'])  # 将对齐重力的世界坐标旋转回相机坐标系

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
    corner_signs = np.array(list(product([-1, 1], repeat=3)), dtype=np.float64)
    # (3,8) => 每列是一个 corner 的局部坐标偏移
    corner_offsets = (corner_signs.T * coeffs[:, None])
    corners_world = (basis @ corner_offsets).T + centroid

    # 将对齐重力的世界坐标转换到相机坐标系
    corners_camera = (R_tilt.T @ corners_world.T).T

    # 如果框被裁剪在摄像机后面，则跳过，避免投影出现极端值
    if np.any(corners_camera[:, 2] <= 1e-3):
        continue

    # 投影到2D
    corners_img = project_points(corners_camera, K)

    # 使用符号组合自动找出需要连接的棱
    corners_int = np.round(corners_img).astype(int)
    for i in range(len(corner_signs)):
        for j in range(i + 1, len(corner_signs)):
            # 两个顶点符号只有一个维度不同 => 一条棱
            if np.sum(np.abs(corner_signs[i] - corner_signs[j])) == 2:
                cv2.line(img, tuple(corners_int[i]), tuple(corners_int[j]), (255, 0, 0), 2)

# === 5. 显示结果 ===
plt.imshow(img)
plt.axis('off')
save_path = "/mnt/pub/wyf/workspace/image_identification/visual_3bbox_output/sunrgbd_vis_3dbbox.jpg"
plt.imsave(save_path, img)
print(f"✅ 可视化结果已保存到: {save_path}")
