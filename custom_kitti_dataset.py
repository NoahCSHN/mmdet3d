# 文件路径: mmdet3d/datasets/custom_kitti_dataset.py
import numpy as np
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

@TRANSFORMS.register_module(force=True)
class PrepareKITTIMultiViewImage:
    """将 KITTI 伪装成多视角，并使用严格的 (N_views, 4, 4) 矩阵维度约束"""
    def __call__(self, results):
        # 1. 强制图像为多视角列表格式 [img]
        if 'img' in results and not isinstance(results['img'], list):
            results['img'] = [results['img']]

        # 2. 安全提取内参和外参 (剥离可能存在的 list 包装)
        cam2img = results.get('cam2img', np.eye(4))
        lidar2cam = results.get('lidar2cam', np.eye(4))

        if isinstance(cam2img, list): cam2img = cam2img[0]
        if isinstance(lidar2cam, list): lidar2cam = lidar2cam[0]

        # 3. 严格对齐 4x4 矩阵
        if cam2img.shape == (3, 4):
            cam2img = np.vstack((cam2img, [0, 0, 0, 1]))
        elif cam2img.shape == (3, 3):
            tmp = np.eye(4)
            tmp[:3, :3] = cam2img
            cam2img = tmp

        if lidar2cam.shape == (3, 4):
            lidar2cam = np.vstack((lidar2cam, [0, 0, 0, 1]))

        # 4. 计算 LSS 所需的逆矩阵和投影矩阵
        cam2lidar = np.linalg.inv(lidar2cam)
        lidar2img = cam2img @ lidar2cam

        # 5. 【极其关键】强制增加 N_views 维度，形塑为 (1, 4, 4) 的 3D 张量！
        # 彻底杜绝 MMEngine 打包时的降维陷阱，完美适配 LSS 模块的 [:, :3, :3] 切片
        results['cam2img'] = np.stack([cam2img], axis=0)
        results['lidar2cam'] = np.stack([lidar2cam], axis=0)
        results['cam2lidar'] = np.stack([cam2lidar], axis=0)
        results['lidar2img'] = np.stack([lidar2img], axis=0)

        return results

@TRANSFORMS.register_module(force=True)
class EnsureMultiViewMetas:
    """终极防线：无论前面的算子怎么降维，在送入网络前强行包裹为多视角格式"""
    def __call__(self, results):
        import numpy as np
        # 【核心修改】：把 img_aug_matrix 也加入强行升维的保护名单！
        target_keys = ['cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar', 'img_aug_matrix']
        for key in target_keys:
            if key in results:
                val = results[key]
                if isinstance(val, np.ndarray) and val.shape == (4, 4):
                    results[key] = [val]
        return results
