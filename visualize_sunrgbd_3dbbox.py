import scipy.io as sio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# === 1. 读取mat文件 ===
meta = sio.loadmat('/mnt/pub/wyf/workspace/image_identification/SUNRGBD_DATA/SUNRGBDMetaData/SUNRGBDMeta3DBB_v2.mat')
sample = meta['SUNRGBDMeta'][0][0]  # 示例取第一张

K = sample['K']  # 相机内参矩阵
rgb_path = sample['rgbpath'][0]     # RGB图片路径

# === 替换路径前缀 ===
# 旧的前缀（原始数据作者电脑）
old_prefix = "/n/fs/sun3d/data/SUNRGBD/"
# 你的本地前缀
new_prefix = "/mnt/pub/wyf/workspace/image_identification/SUNRGBD_DATA/SUNRGBD/"

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
    centroid = bb['centroid'][0]
    coeffs = bb['coeffs'][0]
    
    # ✅【修改1】将原本的 orient 改为 orientation（3x3旋转矩阵）
    if 'orientation' in bb.dtype.names:
        R = bb['orientation']
    else:
        R = np.eye(3)  # 若缺失则用单位矩阵

    # 构造立方体八个角点
    l, h, w = coeffs
    corners = np.array([
        [ l/2,  h/2,  w/2],
        [ l/2,  h/2, -w/2],
        [-l/2,  h/2, -w/2],
        [-l/2,  h/2,  w/2],
        [ l/2, -h/2,  w/2],
        [ l/2, -h/2, -w/2],
        [-l/2, -h/2, -w/2],
        [-l/2, -h/2,  w/2],
    ])
    
    # ✅【修改2】使用 orientation 矩阵直接旋转 + 平移
    corners_world = (R @ corners.T).T + centroid

    # 投影到2D
    corners_img = project_points(corners_world, K)

    # 绘制立方体边
    corners_img = corners_img.astype(int)
    for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        cv2.line(img, tuple(corners_img[i]), tuple(corners_img[j]), (255,0,0), 2)

# === 5. 显示结果 ===
plt.imshow(img)
plt.axis('off')
save_path = "/mnt/pub/wyf/workspace/image_identification/visual_3bbox_output/sunrgbd_vis_3dbbox.jpg"
plt.imsave(save_path, img)
print(f"✅ 可视化结果已保存到: {save_path}")
