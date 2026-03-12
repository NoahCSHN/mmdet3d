_base_ = ['./bevfusion_stageB_r50_depthlss_transfer_2xb4-24e_nus-3d.py']

# Stage C:
# Move to your custom dataset and custom task head.
# Start from Stage B checkpoint and finetune.

# Replace with your Stage B checkpoint path.
load_from = 'data/work_dirs/bevfusion_stageB_r50_depthlss_transfer_2xb4-24e_nus-3d/epoch_24.pth'

# -------------------------
# 1) Dataset customization
# -------------------------
# Replace these with your custom dataset settings.
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        metainfo=metainfo))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator

# -------------------------
# 2) Head customization
# -------------------------
# Replace with your own head type and params.
model = dict(
    bbox_head=dict(
        # Example: your custom head class name.
        # type='YourCustomHead',
        num_classes=len(class_names)))

# -------------------------
# 3) Finetune schedule
# -------------------------
# Small LR is safer when both dataset and head change.
lr = 2e-5
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=1)
