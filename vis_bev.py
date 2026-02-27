import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS
from mmengine.dataset import pseudo_collate
from mmdet3d.apis import init_model

# ================= 配置区 =================
#config_file = 'projects/BEVFusion/configs/bevfusion_lidar_only_lightly.py'
#checkpoint_file = 'data/work_dirs/bevfusion_lidar_only_lightly/epoch_20.pth'
#config_file = 'projects/BEVFusion/configs/bevfusion_lidar_only_lightly.py'
#checkpoint_file = 'data/work_dirs/bevfusion_lidar_only_lightly_pretrained/epoch_5.pth'
#config_file = 'projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
#checkpoint_file = 'data/work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_pretrained/epoch_5.pth'
root_dir = 'data/work_dirs/bevfusion_lidar_only_lightly_pointpillars'
vis_dir = os.path.join(root_dir, 'vis_results')
config_file = 'projects/BEVFusion/configs/bevfusion_lidar_only_lightly.py'
checkpoint_file = './data/centerpoint_02pillar_renamed_for_bevfusion.pth' #os.path.join(root_dir, 'epoch_15.pth')
score_thr = 0.001   # 置信度阈值：只画出大于 0.3 的框
vis_range = 51.2  # 可视化范围：完美匹配你刚刚修改的 ±15m
# ==========================================

print("1. 正在加载模型与配置...")
model = init_model(config_file, checkpoint_file, device='cuda:0')
cfg = Config.fromfile(config_file)

print("2. 正在准备验证集数据...")
dataset = DATASETS.build(cfg.val_dataloader.dataset)
os.makedirs(vis_dir, exist_ok=True)

print("3. 开始推理并生成 BEV 俯视图...")
# 我们直接提取前 10 个验证场景进行画图
for i in range(10):  
    print(f" -> 正在渲染第 {i+1} 个场景...")
    
    # --- 1. 读取点云原始数据 (用于画出球场环境背景) ---
    data_info = dataset.get_data_info(i)
    pts_path = data_info['lidar_points']['lidar_path']
    # NuScenes 数据集的点云包含 5 个维度 (x,y,z,intensity,ring)
    pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 5)

    # --- 2. 运行模型极速推理 ---
    data = dataset[i]
    batch_data = pseudo_collate([data])
    with torch.no_grad():
        result = model.test_step(batch_data)[0]

    # --- 3. 暴力提取 3D 边界框 ---
    pred = result.pred_instances_3d
    bboxes = pred.bboxes_3d.tensor.cpu().numpy()
    scores = pred.scores_3d.cpu().numpy()
    # --- 调试：打印预测结果的真相 ---
    print(f"DEBUG: 场景 {i+1} 检测到的目标总数 (未过滤): {len(bboxes)}")
    if len(scores) > 0:
        print(f"DEBUG: 最高置信度: {scores.max():.4f}")
    else:
        print("DEBUG: 警告！模型在该场景输出为 0 个框")

    # --- 4. 使用 Matplotlib 纯手动渲染 2D 俯视图 ---
    plt.figure(figsize=(10, 10), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # 画出点云（灰色的小点）
    plt.scatter(pts[:, 0], pts[:, 1], s=0.5, c='gray', alpha=0.5)

    # 遍历画出每一个模型预测的目标框
    for bbox, score in zip(bboxes, scores):
        if score < score_thr:
            continue  # 过滤掉低于阈值的盲猜框
            
        x, y, z, l, w, h, yaw = bbox[:7]
        
        # 计算边界框在 2D BEV 视角的 4 个角点
        corners = np.array([
            [-l/2, -w/2], [l/2, -w/2], 
            [l/2, w/2], [-l/2, w/2], 
            [-l/2, -w/2] # 闭合框
        ])
        
        # 根据目标旋转角 (yaw) 旋转框
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw)], 
            [np.sin(yaw), np.cos(yaw)]
        ])
        corners = np.dot(corners, rot_mat.T) + np.array([x, y])
        
        # 将红色边界框画在图上
        plt.plot(corners[:, 0], corners[:, 1], c='red', linewidth=2)
        
        # (可选)在框旁边标出置信度得分
        plt.text(x, y, f'{score:.2f}', color='yellow', fontsize=10)

    # 限制视野在你的 ±15m 感知范围内
    plt.xlim(-vis_range, vis_range)
    plt.ylim(-vis_range, vis_range)
    plt.axis('off') # 关掉多余的坐标轴，让画面更干净

    # 保存这件艺术品！
    plt.savefig(os.path.join(vis_dir, f'bev_sample_{i+1}.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    
    plt.close()

print("可视化大功告成！快去查看 ./vis_results 文件夹吧！")

