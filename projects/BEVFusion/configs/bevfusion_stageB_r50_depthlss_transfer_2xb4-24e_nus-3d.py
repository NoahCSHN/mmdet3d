_base_ = ['./bevfusion_cam_voxel0075_second_secfpn_r50_depth_8xb4-60e_nus-3d.py']

# Stage B:
# Switch to ResNet50 backbone, but transfer all matching weights from Stage A.
# Keep img_backbone ImageNet initialization and let load_from fill other modules.

# Replace this with your actual Stage A checkpoint path when resuming.
# Example:
# load_from = 'data/work_dirs/bevfusion_stageA_swin_depthlss_from_lidarcam_8xb4-6e_nus-3d/epoch_6.pth'
load_from = 'data/work_dirs/bevfusion_stageA_swin_depthlss_from_lidarcam_8xb4-6e_nus-3d/epoch_4.pth'

# Single-GPU friendly defaults.
train_dataloader = dict(batch_size=3, num_workers=4, persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True)
test_dataloader = val_dataloader

# Slightly conservative LR for architecture transition.
lr = 5e-5
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# 24 epochs for adaptation to R50.
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
        T_max=24,
        end=24,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=1)

# Keep global hooks explicit in this stage config.
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))

