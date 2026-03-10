import numpy as np
import cv2, os
import matplotlib.pyplot as plt

# ==========================================
# 核心函数：计算并投影 3D 框的 8 个顶点
# ==========================================
def get_projected_corners(label_3d, P2):
    """
    输入:
        label_3d: [x, y, z, h, w, l, ry] (相机坐标系)
        P2: 3x4 投影矩阵
    输出:
        pts_2d: (8, 2) 8个顶点在图像上的像素坐标 [u, v]
    """
    x, y, z, h, w, l, ry = label_3d
    # ry = -ry
    
    # 1. 定义局部坐标系下的 8 个顶点 (原点在底面中心)
    # KITTI 轴向: x=左右(w), y=上下(h), z=前后(l)
    # 注意：y向上为负
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h] 
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    
    # 2. 绕 Y 轴旋转矩阵
    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    
    # 3. 旋转并平移到相机坐标系
    corners_3d = np.dot(R, corners_3d)
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    # 4. 投影到图像平面
    corners_3d_hom = np.vstack([corners_3d, np.ones(8)])
    pts_2d_hom = np.dot(P2, corners_3d_hom)
    
    # 归一化
    pts_2d = pts_2d_hom[:2, :] / pts_2d_hom[2, :]
    return pts_2d.T.astype(int)

# ==========================================
# 绘图函数：将 3D 和 2D 框画在图上
# ==========================================
def draw_projected_box(image, corners, bbox_2d, color_3d=(0, 255, 0), color_2d=(0, 0, 255)):
    """
    image: OpenCV 图像 (BGR)
    corners: (8, 2) 3D顶点的投影
    bbox_2d: [xmin, ymin, xmax, ymax] 手动算出的2D框
    """
    # 1. 绘制 3D 丝体框 (绿色)
    # 定义连接顺序 (底面4点，顶面4点，垂直4点)
    # 0-3: 底面, 4-7: 顶面
    for i in range(4):
        # 画底面边
        cv2.line(image, tuple(corners[i]), tuple(corners[(i+1)%4]), color_3d, 2)
        # 画顶面边
        cv2.line(image, tuple(corners[i+4]), tuple(corners[((i+1)%4)+4]), color_3d, 2)
        # 画连接底面和顶面的垂直边
        cv2.line(image, tuple(corners[i]), tuple(corners[i+4]), color_3d, 2)

    # 2. 绘制计算出的最小外接 2D 矩形 (红色)
    xmin, ymin, xmax, ymax = [int(c) for c in bbox_2d]
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_2d, 2)
    
    # 添加图例文本
    cv2.putText(image, 'Green: Projected 3D Corners', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_3d, 2)
    cv2.putText(image, 'Red: Calculated 2D BBox', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_2d, 2)

    return image

# ==========================================
# 主程序示例 (使用 KITTI 样例数据)
# ==========================================

# 1. 模拟 KITTI 标定矩阵 P2 (左彩色相机)
P2_sample = np.array([[616.730000, 0.000000, 972.505000, 0.000000], [0.000000, 900.091000, 540.903000, 0.000000], [0.000000, 0.000000, 1.000000, 0.000000]])

# 2. 模拟一条 3D 标注数据 (一辆Car)
# 格式: [x, y, z, h, w, l, ry] (相机坐标系)
#label_3d_sample =[-4.4434, -1.9531, 4.3730, 1.0000, 0.3700, 0.8100, 1.4359] 
label_3d_sample =[-4.4434, -1.9531, 4.3730, 0.8100, 1.0000, 0.3700, 1.4359] 

# 3. 创建一张空白图片用于演示 (或者你可以读取一张实际的 KITTI 图片)
# 1. 读取原图
img = cv2.imread("data/3DBox_Annotation_20260305154810/training/image_2/1772689948923010048.png") # 替换为你的照片路径
img_H, img_W = img.shape[:2] 

# --- 步骤 A: 计算 3D 顶点的投影 ---
projected_corners = get_projected_corners(label_3d_sample, P2_sample)

# --- 步骤 B: 手动计算 2D 最小外接框 (xmin, ymin, xmax, ymax) ---
# 这正是你上一个问题想验证的核心逻辑
xmin = np.min(projected_corners[:, 0])
ymin = np.min(projected_corners[:, 1])
xmax = np.max(projected_corners[:, 0])
ymax = np.max(projected_corners[:, 1])

# 边界裁剪 (KITTI standard)
xmin = max(0, min(img_W - 1, xmin))
ymin = max(0, min(img_H - 1, ymin))
xmax = max(0, min(img_W - 1, xmax))
ymax = max(0, min(img_H - 1, ymax))

calculated_bbox_2d = [xmin, ymin, xmax, ymax]

# --- 步骤 C: 可视化 ---
vis_image = draw_projected_box(img.copy(), projected_corners, calculated_bbox_2d)

# 显示结果
save_path = 'debug_result2.jpg' 
cv2.imwrite(save_path, vis_image)
print(f"✅ 图片已保存至容器内路径: {os.path.abspath(save_path)}")
#plt.figure(figsize=(15, 5))
#plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
#plt.axis('off')
#plt.title("KITTI 3D-to-2D Projection Visualization")
#plt.show()
#if __name__ == '__main__':
#    #label_3d = [1.0000, 0.3700, 0.8100, 4.3730, -4.4434, -1.9531, 1.4359]
#    label_3d = [1.0000, 0.3700, 0.8100, 4.3730, -4.4434, -1.9531, 1.4359]
#    label_3d = [-4.4434, -1.9531, 4.3730, 1.0000, 0.3700, 0.8100, 1.4359]
#    label_3d = [-4.4434, -1.9531, 4.3730, 0.3700, 0.8100, 1.0000, 1.4359]
#    label_3d = [-4.4434, -1.9531, 4.3730, 0.8100, 1.0000, 0.3700, 1.4359]
#    P2 = [[616.730000, 0.000000, 972.505000, 0.000000], [0.000000, 900.091000, 540.903000, 0.000000], [0.000000, 0.000000, 1.000000, 0.000000]] 
#    img_size = [1920, 1080]
#    label_2d = project_3d_to_2d(label_3d, P2, img_size)
#    print(f'3D box:{label_3d} -> 2D box:{label_2d}')
