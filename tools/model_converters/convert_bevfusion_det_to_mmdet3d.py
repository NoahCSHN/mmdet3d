import argparse
from collections import OrderedDict

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet3d.registry import MODELS


PREFIX_RULES = [
    ('encoders.camera.backbone.', 'img_backbone.'),
    ('encoders.camera.neck.', 'img_neck.'),
    ('encoders.camera.vtransform.', 'view_transform.'),
    ('encoders.lidar.backbone.', 'pts_middle_encoder.'),
    ('decoder.backbone.', 'pts_backbone.'),
    ('decoder.neck.', 'pts_neck.'),
    ('heads.object.', 'bbox_head.'),
    ('fuser.', 'fusion_layer.'),
]


def remap_decoder_subkeys(key: str) -> str:
    """Remap old transformer decoder naming to current mmdet3d naming."""
    if not key.startswith('bbox_head.decoder.'):
        return key

    key = key.replace('.self_attn.in_proj_', '.self_attn.attn.in_proj_')
    key = key.replace('.self_attn.out_proj.', '.self_attn.attn.out_proj.')
    key = key.replace('.multihead_attn.in_proj_', '.cross_attn.attn.in_proj_')
    key = key.replace('.multihead_attn.out_proj.', '.cross_attn.attn.out_proj.')
    key = key.replace('.linear1.', '.ffn.layers.0.0.')
    key = key.replace('.linear2.', '.ffn.layers.1.')
    key = key.replace('.norm1.', '.norms.0.')
    key = key.replace('.norm2.', '.norms.1.')
    key = key.replace('.norm3.', '.norms.2.')
    return key


def remap_key(old_key: str) -> str:
    for old_prefix, new_prefix in PREFIX_RULES:
        if old_key.startswith(old_prefix):
            mapped = new_prefix + old_key[len(old_prefix):]
            return remap_decoder_subkeys(mapped)
    return remap_decoder_subkeys(old_key)


def maybe_permute_spconv_weight(key: str, tensor: torch.Tensor,
                                target_shape) -> torch.Tensor:
    """Handle old/new sparse conv kernel layout mismatch when needed."""
    if not key.startswith('pts_middle_encoder.'):
        return tensor
    if tensor.ndim != 5:
        return tensor
    if tuple(tensor.shape) == tuple(target_shape):
        return tensor

    # Old layout: [out, kx, ky, kz, in] -> New layout: [kx, ky, kz, in, out]
    permuted = tensor.permute(1, 2, 3, 4, 0).contiguous()
    if tuple(permuted.shape) == tuple(target_shape):
        return permuted
    return tensor


def build_target_state_dict(config_path: str):
    cfg = Config.fromfile(config_path)
    # Prevent downloading/initializing ImageNet checkpoint during conversion.
    if 'img_backbone' in cfg.model and isinstance(cfg.model.img_backbone, dict):
        cfg.model.img_backbone['init_cfg'] = None

    init_default_scope('mmdet3d')
    model = MODELS.build(cfg.model)
    return model.state_dict()


def convert_checkpoint(src_path: str, dst_path: str, config_path: str):
    ckpt = torch.load(src_path, map_location='cpu')
    src_sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
    if not isinstance(src_sd, dict):
        raise TypeError('Unsupported checkpoint format: state dict not found')

    target_sd = build_target_state_dict(config_path)

    converted = OrderedDict()
    stats = {
        'total_src': 0,
        'mapped': 0,
        'kept': 0,
        'missing_in_target': 0,
        'shape_mismatch': 0,
        'permuted_spconv': 0,
    }

    for old_k, old_v in src_sd.items():
        stats['total_src'] += 1
        new_k = remap_key(old_k)
        if new_k != old_k:
            stats['mapped'] += 1

        if new_k not in target_sd:
            stats['missing_in_target'] += 1
            continue

        v = old_v
        if v.ndim == 5:
            maybe_v = maybe_permute_spconv_weight(new_k, v, target_sd[new_k].shape)
            if maybe_v is not v:
                stats['permuted_spconv'] += 1
            v = maybe_v

        if tuple(v.shape) != tuple(target_sd[new_k].shape):
            stats['shape_mismatch'] += 1
            continue

        converted[new_k] = v
        stats['kept'] += 1

    out = {
        'meta': {
            'converted_from': src_path,
            'converter': 'convert_bevfusion_det_to_mmdet3d.py',
            'stats': stats,
        },
        'state_dict': converted,
    }
    torch.save(out, dst_path)

    print('Conversion done.')
    for k, v in stats.items():
        print(f'{k}: {v}')
    print(f'output: {dst_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert old BEVFusion det checkpoint keys to mmdet3d BEVFusion keys.'
    )
    parser.add_argument('--src', default='data/bevfusion-det.pth')
    parser.add_argument('--dst', default='data/bevfusion_det_compat.pth')
    parser.add_argument(
        '--config',
        default='projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_r50_imagenet_8xb4-cyclic-20e_nus-3d.py')
    args = parser.parse_args()

    convert_checkpoint(args.src, args.dst, args.config)
