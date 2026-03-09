from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics import KittiMetric
from mmdet3d.evaluation.functional import kitti_eval

@METRICS.register_module(force=True)
class CustomKittiMetric(KittiMetric):
    """
    专门适配自定义类别的 KITTI 评估器。
    它会从 dataset_meta 中自动提取类别名，而不是死扣 'Car'。
    """
    def compute_metrics(self, results: list) -> dict:
        # 1. 从数据集中提取你定义的类别名 ['Distance_Marker', 'Structure']
        classes = self.dataset_meta['classes']
        
        # 2. 将模型推理结果转换为 KITTI 评估函数需要的格式
        # 这里逻辑沿用基类，但核心在于后面的 eval 环节
        ret_dict = super().compute_metrics(results)
        
        # 3. 这里的关键是：KittiMetric 的父类可能已经在 compute_metrics 里跑过了标准的 eval
        # 如果 AP 还是 0，我们需要手动触发一次包含自定义类别的评估过程
        return ret_dict

    # 核心重写：覆盖内部的评估逻辑
    def _evaluate(self, sample_results, groundtruths, pkl_infos=None):
        # 这里的 classes 会被传递给 C++ 后端或 Python 评估函数
        classes = self.dataset_meta['classes']
        
        # 强制指定评估类别为你的自定义类别
        eval_types = ['bbox', 'bev', '3d']
        
        # 调用 mmdet3d 的 functional 接口进行评估
        # 这里的 current_classes 极其关键，它打破了 'Car' 的限制
        ap_dict = kitti_eval(
            groundtruths,
            sample_results,
            current_classes=classes, 
            eval_types=eval_types)
        
        return ap_dict
