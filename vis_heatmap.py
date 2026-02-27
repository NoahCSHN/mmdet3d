import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import DATASETS
from mmdet3d.apis import init_model

# ================= 【神级修复：运行时拦截】 =================
# 在导入模型之前，我们强行修改底层 Scatter 模块的张量逻辑！
from mmdet3d.models.middle_encoders import PointPillarsScatter
original_forward = PointPillarsScatter.forward

def patched_forward(self, voxel_features, coors, *args, **kwargs):
    # 致命 Bug 修复：将 BEVFusion 输出的 [b, x, y, z] 纠正为 [b, z, y, x]
    coors_fixed = coors.clone()
    coors_fixed[:, 1] = coors[:, 3]  # 把 Z (0) 放到索引 1
    coors_fixed[:, 3] = coors[:, 1]  # 把 X 放到索引 3
    
    return original_forward(self, voxel_features, coors_fixed, *args, **kwargs)

# 狸猫换太子，替换原始的 forward 函数
PointPillarsScatter.forward = patched_forward
# ============================================================

from mmdet3d.apis import init_model

# ================= 配置区 =================
config_file = 'projects/BEVFusion/configs/bevfusion_lidar_only_lightly.py'
#checkpoint_file = 'data/work_dirs/bevfusion_lidar_only_lightly_pointpillars_nopretrained/epoch_20.pth'
checkpoint_file = './data/centerpoint_02pillar_renamed_for_bevfusion.pth' #os.path.join(root_dir, 'epoch_15.pth')
save_dir = './data/work_dirs/bevfusion_lidar_only_lightly_pointpillars/vis_heatmaps'
device = 'cuda:0'
# ==========================================

os.makedirs(save_dir, exist_ok=True)
model = init_model(config_file, checkpoint_file, device=device)
cfg = Config.fromfile(config_file)
dataset = DATASETS.build(cfg.val_dataloader.dataset)
model.eval()

for i in range(5):  
    print(f" -> 正在分析第 {i+1} 帧数据...")
    data = dataset[i]
    inputs = pseudo_collate([data])
    batch_inputs = inputs['inputs']
    
    # 回退到你之前能成功跑通的数据准备逻辑
    if isinstance(batch_inputs['points'], list):
        batch_inputs['points'] = [p.to(device) for p in batch_inputs['points']]
    
    with torch.no_grad():
        # 1. 提取点云特征 (此时 Scatter 模块会自动触发我们的修复代码！)
        x = model.extract_pts_feat(batch_inputs)
        
        # 2. 驱动 Backbone 和 Neck
        x = model.pts_backbone(x)
        x = model.pts_neck(x)
        
        # 3. 传入检测头并提取热力图
        head_input = [x] if not isinstance(x, (list, tuple)) else x
        outs = model.bbox_head(head_input)
        
        heatmaps = []
        for task_output in outs:
            hm_tensor = task_output[0]['heatmap'].sigmoid()
            heatmaps.append(hm_tensor)

    task_idx = 0 
    hm = heatmaps[task_idx][0].max(dim=0)[0].cpu().numpy()
    
    plt.figure(figsize=(8, 8), facecolor='black')
    plt.imshow(hm, extent=[-51.2, 51.2, -51.2, 51.2], cmap='inferno', origin='lower')
    plt.colorbar(label='Confidence Score')
    plt.title(f'TRUE Task {task_idx} Heatmap - Frame {i+1}', color='white')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f'true_heatmap_task{task_idx}_frame{i+1}.png'), bbox_inches='tight', pad_inches=0.1)
    plt.close()

print(f"可视化完成！快去查看 {save_dir}，左边那条亮线消失了吗？")
