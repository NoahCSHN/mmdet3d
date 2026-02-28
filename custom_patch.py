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
        # 空间画布翻转 [B, C, Y, X] -> [B, C, X, Y]
        #return out.transpose(2, 3)
        return out

@TASK_UTILS.register_module(force=True)
class BEVFusionAbsoluteBBoxCoder(CenterPointBBoxCoder):
    def decode(self, *args, **kwargs):
        # 1. 翻转空间画布 [X, Y] -> [Y, X]
        def swap_spatial_yx(obj):
            if isinstance(obj, torch.Tensor):
                return obj.transpose(-2, -1).contiguous()
            elif isinstance(obj, list):
                return [swap_spatial_yx(item) for item in obj]
            return obj

        new_kwargs = {k: swap_spatial_yx(v) for k, v in kwargs.items()}
        new_args = [swap_spatial_yx(a) for a in args]

        # 2. 翻转物理通道 (智能兼容 3D 和 4D 张量)
        def swap_channels(t_list):
            for t in t_list:
                if t.dim() == 3:  # 被剥离了 Batch 维: [C, Y, X]
                    t[[0, 1], ...] = t[[1, 0], ...]
                elif t.dim() == 4:  # 保留了 Batch 维: [B, C, Y, X]
                    t[:, [0, 1], ...] = t[:, [1, 0], ...]

        if 'reg' in new_kwargs: swap_channels(new_kwargs['reg'])
        if 'vel' in new_kwargs: swap_channels(new_kwargs['vel'])
        if 'rot' in new_kwargs: swap_channels(new_kwargs['rot'])
        if 'dim' in new_kwargs: swap_channels(new_kwargs['dim'])

        # 3. 交给原生解码器
        results = super().decode(*new_args, **new_kwargs)
        
        # 4. 尺寸还原 (如果是自己训练出的对数尺寸，请删除这两行)
        #for res in results:
        #    if res['bboxes'].shape[0] > 0:
        #        res['bboxes'][:, 3:6] = torch.log(res['bboxes'][:, 3:6])
                
        return results
