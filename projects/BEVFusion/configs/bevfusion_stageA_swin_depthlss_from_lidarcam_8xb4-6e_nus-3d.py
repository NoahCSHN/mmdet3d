_base_ = ['./bevfusion_cam_voxel0075_second_secfpn_r50_depth_8xb4-60e_nus-3d.py']

# Stage A:
# Use camera-only BEV pipeline (same logic as Stage B) but with Swin backbone.
# This stage verifies explicit depth-supervision training is correct before switching to R50.

model = dict(
    img_backbone=dict(
        _delete_=True,
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')
    ),
    img_neck=dict(
        _delete_=True,
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        _delete_=True,
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2,
        loss_depth_weight=0.1,
        use_sparse_depth=False),
    fusion_layer=None)

# Keep camera-only inference mode while preserving lidar points in train pipeline for depth GT projection.
train_input_modality = dict(use_lidar=True, use_camera=True)
test_input_modality = dict(use_lidar=False, use_camera=True)

# Start from official lidar-cam checkpoint and rely on non-strict matching.
load_from = 'data/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'

train_dataloader = dict(batch_size=3, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True)
test_dataloader = val_dataloader

lr = 2.5e-5
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=6,
        end=6,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))
