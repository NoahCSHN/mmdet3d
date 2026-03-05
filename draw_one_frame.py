import cv2
import torch
import mmcv
import numpy as np
import copy
from mmengine.config import Config
from mmdet3d.registry import DATASETS
from mmdet3d.apis import init_model
from mmengine.dataset import pseudo_collate
from mmdet3d.visualization import Det3DLocalVisualizer

# ================= 配置路径 =================
cfg_path = 'projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_kitti-3d.py'
ckpt_path = 'data/work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_kitti-3d_nopretrained/epoch_15.pth'

print("🚀 1. 正在加载配置和模型...")
cfg = Config.fromfile(cfg_path)
model = init_model(cfg, ckpt_path, device='cuda:0')
model.eval()

print("📦 2. 正在构建数据集 (为了获取完美的相机内参矩阵)...")
dataset = DATASETS.build(cfg.val_dataloader.dataset)
data_info = dataset[0] 
data_batch = pseudo_collate([data_info])

print("🧠 3. 正在进行单帧神经网络推理...")
with torch.no_grad():
    results = model.test_step(data_batch)

print("🎨 4. 初始化底层画笔，绕过所有高级流水线...")
data_sample = results[0]

# 解析图像
img_path = data_sample.img_path
if isinstance(img_path, list):
    img_path = img_path[0]
img = mmcv.imread(img_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')

# 建立极简画板并铺上底图
visualizer = Det3DLocalVisualizer()
visualizer.set_image(img)

# 强行提取和对齐投影矩阵
input_meta = copy.deepcopy(data_sample.metainfo)
if 'lidar2img' in input_meta:
    l2i = input_meta['lidar2img']
    if isinstance(l2i, list):
        l2i = np.array(l2i[0])
    if isinstance(l2i, np.ndarray) and l2i.ndim == 3:
        l2i = l2i[0]
    input_meta['lidar2img'] = l2i

print("🖌️ 5. 正在执行纯数学 3D 投影绘制...")
# 画出真实的绿色框
if 'gt_instances_3d' in data_sample:
    visualizer.draw_proj_bboxes_3d(
        data_sample.gt_instances_3d.bboxes_3d, 
        input_meta, 
        edge_colors='green'
    )

# 画出预测的红色框 (仅保留置信度 > 0.3 的框)
if 'pred_instances_3d' in data_sample:
    preds = data_sample.pred_instances_3d
    scores = preds.scores_3d.cpu().numpy()
    mask = scores > 0.3
    valid_bboxes = preds.bboxes_3d[mask]
    visualizer.draw_proj_bboxes_3d(
        valid_bboxes, 
        input_meta, 
        edge_colors='red'
    )

print("💾 6. 使用 OpenCV 强行写入硬盘...")
drawn_img = visualizer.get_image()
bgr_img = cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR)
save_path = './data/work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_kitti-3d_nopretrained/final_result_0000_bulletproof.png'
cv2.imwrite(save_path, bgr_img)

print(f"✅ 绝杀成功！请查看当前目录下的 {save_path}")
