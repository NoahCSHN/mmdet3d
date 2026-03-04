# 文件路径: mmdet3d/datasets/custom_kitti_dataset.py
from mmdet3d.registry import DATASETS
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.kitti_dataset import KittiDataset

@DATASETS.register_module(force=True)
class CustomKittiDataset(KittiDataset):
    """自定义的数据集类，使用自定义的类别"""
    
    METAINFO = {
        'classes': ('MarkBand', ), # 你的目标类别
        'palette': [(255, 0, 0)]   
    }

    def parse_data_info(self, info: dict) -> dict:
        # 【核心魔法】：在底层框架进行"安检"过滤之前，直接在内存里篡改数据！
        if 'instances' in info:
            for inst in info['instances']:
                # 无论官方底层把它标成了 8 (DontCare) 还是其他数字，强行洗脑为 0
                inst['bbox_label'] = 0
                if 'bbox_label_3d' in inst:
                    inst['bbox_label_3d'] = 0
                # 强行修正名字，防止因为名字不匹配被抛弃
                if 'bbox_label_name' in inst:
                    inst['bbox_label_name'] = 'MarkBand'
                    
        # 洗白完成后，再交给官方底层去走标准流程
        return super().parse_data_info(info)

@TRANSFORMS.register_module()
class PrepareKITTIMultiViewImage:
    """将 KITTI 的单视角图像和内参伪装成 BEVFusion 期望的多视角列表"""
    def __call__(self, results):
        if 'img' in results and not isinstance(results['img'], list):
            results['img'] = [results['img']]
        if 'cam2img' in results and not isinstance(results['cam2img'], list):
            results['cam2img'] = [results['cam2img']]
        if 'lidar2cam' in results and not isinstance(results['lidar2cam'], list):
            results['lidar2cam'] = [results['lidar2cam']]

        # 2. 动态计算 lidar2img
        # 矩阵乘法：内参 (cam2img) 乘以 外参 (lidar2cam)
        if 'lidar2img' not in results:
            lidar2img = np.matmul(results['cam2img'][0], results['lidar2cam'][0])
            results['lidar2img'] = [lidar2img]
        return results
