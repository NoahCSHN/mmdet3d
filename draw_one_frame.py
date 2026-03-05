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

# 【核心黑科技】：动态修改验证集的流水线，强行加载真实标签（GT）！
test_pipeline = cfg.val_dataloader.dataset.pipeline
# 寻找合适的位置插入标注加载器
has_ann = any(t['type'] == 'LoadAnnotations3D' for t in test_pipeline)
if not has_ann:
    test_pipeline.insert(2, dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True))

# 强行修改打包器，不准它丢弃真实框
for t in test_pipeline:
    if t['type'] == 'Pack3DDetInputs':
        if 'gt_bboxes_3d' not in t['keys']:
            t['keys'].append('gt_bboxes_3d')
        if 'gt_labels_3d' not in t['keys']:
            t['keys'].append('gt_labels_3d')

model = init_model(cfg, ckpt_path, device='cuda:0')
model.eval()

print("📦 2. 正在构建携带 GT 的完整数据集...")
dataset = DATASETS.build(cfg.val_dataloader.dataset)
data_info = dataset[0] 
data_batch = pseudo_collate([data_info])

print("🧠 3. 正在进行单帧神经网络推理...")
with torch.no_grad():
    results = model.test_step(data_batch)

print("🎨 4. 初始化底层画笔...")
# 注意：results 里的 data_sample 包含了推断结果，但为了绝对安全，我们直接从输入里抓 GT
data_sample = results[0]
raw_data_sample = data_info['data_samples'] 

# 解析图像
img_path = data_sample.img_path
if isinstance(img_path, list):
    img_path = img_path[0]
img = mmcv.imread(img_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')

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

# 【步骤 A】：画出真实的绿色框 (Ground Truth)
if hasattr(raw_data_sample, 'gt_instances_3d'):
    gt_bboxes = raw_data_sample.gt_instances_3d.bboxes_3d
    if len(gt_bboxes) > 0:
        visualizer.draw_proj_bboxes_3d(
            gt_bboxes, 
            input_meta, 
            edge_colors='green'
        )
        print(f"🟩 成功绘制 {len(gt_bboxes)} 个真实的 Ground Truth 绿框！")

# 【步骤 B】：画出预测的红色框 (Predictions)
if 'pred_instances_3d' in data_sample:
    preds = data_sample.pred_instances_3d
    if hasattr(preds, 'bboxes_3d'):
        scores = preds.scores_3d.cpu().numpy()
        mask = scores > 0.3 # 只保留置信度大于 30% 的预测框
        valid_bboxes = preds.bboxes_3d[mask]
        
        if len(valid_bboxes) > 0:
            visualizer.draw_proj_bboxes_3d(
                valid_bboxes, 
                input_meta, 
                edge_colors='red'
            )
            print(f"🟥 成功绘制 {len(valid_bboxes)} 个高置信度预测红框！")
        else:
            print("⚠️ 这一帧模型没有预测出置信度大于 0.3 的物体。")

print("💾 6. 使用 OpenCV 强行写入硬盘...")
drawn_img = visualizer.get_image()
bgr_img = cv2.cvtColor(drawn_img, cv2.COLOR_RGB2BGR)
save_path = './final_result_0000_with_GT.png'
cv2.imwrite(save_path, bgr_img)

print(f"✅ 完美提取！请查看当前目录下的 {save_path}")
