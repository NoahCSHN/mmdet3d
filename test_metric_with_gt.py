import mmengine
import numpy as np
import torch
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes #

# ================= 1. 配置路径与类别 =================
#
data_root = 'data/3DBox_Annotation_20260305154810/'
ann_file = data_root + 'kitti_infos_val.pkl'
class_names = ['Distance_Marker', 'Structure']
metainfo = {'classes': class_names}

def mock_gt_as_pred():
    metric_cfg = dict(
        type='CustomKittiMetric',
        ann_file=ann_file,
        metric='bbox'
    )
    evaluator = METRICS.build(metric_cfg)
    evaluator.dataset_meta = metainfo
    
    print(f"加载真值文件: {ann_file}")
    raw_data = mmengine.load(ann_file)
    data_list = raw_data['data_list']
    evaluator.data_infos = data_list 
    
    fake_results = []

    # ================= 2. 【核心修复】：使用 i 作为索引 =================
    # 这里使用 enumerate 确保 sample_idx 永远在 0 到 len(data_list)-1 之间
    for i, instance_info in enumerate(data_list):
        gt_instances = instance_info.get('instances', [])
        
        gt_bboxes_3d = []
        gt_labels_3d = []
        for inst in gt_instances:
            gt_bboxes_3d.append(inst['bbox_3d'])
            gt_labels_3d.append(inst['bbox_label_3d'])

        # 构造 3D 框
        bboxes_tensor = torch.tensor(gt_bboxes_3d, dtype=torch.float32) if gt_bboxes_3d else torch.zeros((0, 7))
        
        # 【核心修正】：模拟完整的 Data Sample 结构，包含校准信息
        fake_results.append({
            'pred_instances_3d': {
                'bboxes_3d': LiDARInstance3DBoxes(bboxes_tensor, box_dim=7),
                'labels_3d': torch.tensor(gt_labels_3d, dtype=torch.long),
                'scores_3d': torch.ones(len(gt_labels_3d))
            },
            # 必须透传校准矩阵，否则转换后的坐标全是错的
            'lidar2cam': instance_info.get('lidar2cam', np.eye(4)),
            'cam2img': instance_info.get('cam2img', np.eye(4)),
            'sample_idx': i 
        })

    print(f"成功构造 {len(fake_results)} 帧预测数据。正在评估...")

    # ================= 2. 调试：检查转换逻辑 =================
    print("--- 正在检查第一帧的转换结果 ---")
    
    # 【修复点】：显式传入 class_names 参数
    # 在 MMDet3D 1.x 中，该函数的参数顺序通常为 (results, class_names, outfile_prefix)
    try:
        kitti_results = evaluator.bbox2result_kitti(
            fake_results, 
            class_names,  # 传入你定义的 ['Distance_Marker', 'Structure']
        )
        
        # 查看转换后的第一条记录
        if len(kitti_results) > 0 and len(kitti_results[0]) > 0:
            # kitti_results 是一个列表的列表，外层是帧，内层是该帧的检测框字典

            first_box = kitti_results[0][0] 
            print(f"转换后的类名: {first_box['name']}")
            print(f"转换后的置信度: {first_box['score']}")
            print(f"转换后的 3D 中心 (Location): {first_box['location']}")
            # 【关键线索探测】：检查转换后的数据里是否有 'MarkBand'
            all_names = set([box['name'] for frame in kitti_results for box in frame])
            print(f"转换结果中包含的所有类名: {all_names}")
    except Exception as e:
        print(f"转换检查失败: {e}")

    # ================= 3. 执行完整评估 =================
    print("\n正在通过真值模拟计算评估指标...")
    metrics = evaluator.compute_metrics(fake_results)

    # ================= 3. 执行评估 =================
    metrics = evaluator.compute_metrics(fake_results)
    
    print("\n" + "="*45)
    print("评估结果 (GT 完美模拟测试):")
    for k, v in metrics.items():
        print(f"{k:45s}: {v:.4f}")
    print("="*45)

if __name__ == '__main__':
    from custom_kitti_metric import CustomKittiMetric
    mock_gt_as_pred()
