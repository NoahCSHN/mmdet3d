from mmdet3d.registry import METRICS
from mmdet3d.evaluation.metrics import KittiMetric
from mmdet3d.evaluation.functional import kitti_eval
from mmengine.logging import print_log

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

    def kitti_evaluate(self,
                       results_dict,
                       gt_annos,
                       metric=None,
                       classes=None,
                       logger=None):
        """Evaluate custom classes with 360-LiDAR-friendly metrics.

        For 3D predictions, only compute BEV/3D AP to avoid front-camera 2D
        constraints dominating 360-degree LiDAR evaluation.
        """
        ap_dict = {}
        for name in results_dict:
            if name == 'pred_instances_3d':
                eval_types = ['bev', '3d']
            else:
                eval_types = ['bbox']

            ap_result_str, ap_dict_ = kitti_eval(
                gt_annos, results_dict[name], classes, eval_types=eval_types)
            for ap_type, ap in ap_dict_.items():
                ap_dict[f'{name}/{ap_type}'] = float(f'{ap:.4f}')

            print_log(f'Results of {name}:\n' + ap_result_str, logger=logger)

        return ap_dict
