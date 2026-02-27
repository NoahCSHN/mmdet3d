# 文件名: custom_patch.py
import torch
from mmdet3d.models.middle_encoders import PointPillarsScatter
from mmdet3d.models.task_modules.coders import CenterPointBBoxCoder
from mmdet3d.registry import MODELS, TASK_UTILS

@MODELS.register_module(force=True)
class BEVFusionPointPillarsScatter(PointPillarsScatter):
    def forward(self, voxel_features, coors, batch_size=None, **kwargs):
        coors_fixed = coors.clone()
        coors_fixed[:, 1] = coors[:, 3]
        coors_fixed[:, 3] = coors[:, 1]
        out = super().forward(voxel_features, coors_fixed, batch_size, **kwargs)
        return out.transpose(2, 3)

@TASK_UTILS.register_module(force=True)
class BEVFusionAbsoluteBBoxCoder(CenterPointBBoxCoder):
    def decode(self, *args, **kwargs):
        def swap_yx(obj):
            if isinstance(obj, torch.Tensor):
                return obj.transpose(-2, -1).contiguous()
            elif isinstance(obj, list):
                return [swap_yx(item) for item in obj]
            return obj

        new_kwargs = {k: swap_yx(v) for k, v in kwargs.items()}
        new_args = [swap_yx(a) for a in args]

        results = super().decode(*new_args, **new_kwargs)
        for res in results:
            if res['bboxes'].shape[0] > 0:
                res['bboxes'][:, 3:6] = torch.log(res['bboxes'][:, 3:6])
        return results
