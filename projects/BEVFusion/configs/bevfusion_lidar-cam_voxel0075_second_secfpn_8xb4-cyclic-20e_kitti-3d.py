_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion', 'custom_kitti_dataset', 'custom_kitti_metric'], 
    allow_failed_imports=False)

# ================= 1. 基础物理范围与类别 =================
class_names = ['Distance_Marker', 'Structure']
metainfo = dict(classes=class_names)
dataset_type = 'CustomKittiDataset'
data_root = 'data/3DBox_Annotation_20260309105213/'
#dataset_type = 'KittiDataset'
#data_root = 'data/kitti/'

# 【核心修复】：明确指定点云和图像的子目录前缀！
data_prefix = dict(
    pts='training/velodyne', 
    img='training/image_2')

# 【精算后的 KITTI 前视范围】
# X: 0 到 74.4 (74.4 / 0.075 = 992)
# Y: -40.8 到 40.8 (81.6 / 0.075 = 1088)
# Z: -3.0 到 1.0 (4.0 / 0.2 = 20)
#point_cloud_range = [0.0, -40.8, -3.0, 74.4, 40.8, 1.0]
#voxel_size = [0.075, 0.075, 0.2]
#grid_size = [992, 1088, 20] 
# 恢复 360 度全景物理范围，包容你的所有点云
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]

# 精确重算 Grid Size: (54 - (-54)) / 0.075 = 1440
grid_size = [1440, 1440, 40]

# 仅使用 camera 分支做检测；点云仍在 pipeline 中读取，用于 DepthNet 深度监督
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# ================= 2. BEVFusion 模型架构 =================
model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[120000, 160000],
            voxelize_reduce=True)),
            
    # ------ LiDAR 检测分支关闭（仅保留点云用于深度监督） ------
    pts_voxel_encoder=None,
    pts_middle_encoder=None,
    pts_backbone=dict(
        type='SECOND',
        in_channels=80,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    # ------ Camera 分支 ------
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=[1, 2, 3],
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
        
    # 【核心：LSS 视锥转换对齐】
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        # X 轴 248 个 bin，经过 downsample=2 后特征图 X 宽为 124 (与激光雷达 992/8=124 完美对齐)
        xbound=[-54.0, 54.0, 0.3], 
        # Y 轴 272 个 bin，经过 downsample=2 后特征图 Y 宽为 136 (与激光雷达 1088/8=136 完美对齐)
        ybound=[-54.0, 54.0, 0.3], 
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2,
        loss_depth_weight=0.2),
        
    fusion_layer=None,

    # ------ TransFusion 检测头 ------
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512, # 保持预训练模型的输入通道数
        hidden_channel=128,
        num_classes=2,   # 【只预测 MarkBand 1 类】
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128, feedforward_channels=256, num_fcs=2,
                ffn_drop=0.1, act_cfg=dict(type='ReLU', inplace=True)),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        train_cfg=dict(
            dataset='kitti',
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            # 去掉速度 (vel) 的权重，一共 8 位 (x,y,z,w,l,h,sin,cos)
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='mmdet.FocalLossCost', gamma=2.0, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        test_cfg=dict(
            dataset='kitti',
            grid_size=grid_size,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            pc_range=[-54.0, -54.0], # 仅前视
            nms_type=None),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]), # 去掉了 vel=[2, 2]
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            code_size=8), # 尺寸由 10 变为 8
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)))

# ================= 3. 数据流水线 =================
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'), 
    dict(type='PrepareKITTIMultiViewImage'), # 调用我们刚写的包裹函数！
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # 【注意】多模态融合强制关闭 ObjectSample (Copy-Paste)，因为贴了点云没有对应的图像像素，会导致网络学习错乱！
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.55], # 缩放 KITTI 图像
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=False, # 关闭图像翻转，防止和点云出现空间错位
        is_train=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='EnsureMultiViewMetas'), # <--- 插入在这里！(打包前的最后一道安检)
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['cam2img', 'lidar2cam', 'img_shape', 'box_type_3d', 'sample_idx', 'lidar_path', 'img_path', 'lidar2img', 'cam2lidar', 'img_aug_matrix'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadImageFromFile'),
    dict(type='PrepareKITTIMultiViewImage'),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='EnsureMultiViewMetas'), # <--- 插入在这里！(打包前的最后一道安检)
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points'],
        meta_keys=['cam2img', 'lidar2cam', 'img_shape', 'box_type_3d', 'sample_idx', 'lidar_path', 'img_path', 'lidar2img', 'cam2lidar', 'img_aug_matrix', 'num_pts_feats'])
]

# ================= 4. Dataloader 与 训练策略 =================
train_dataloader = dict(
    batch_size=2, # 多模态显存占用极大，建议调低
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='kitti_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file='kitti_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline, metainfo=metainfo, modality=input_modality,
        test_mode=True, box_type_3d='LiDAR'))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CustomKittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    pcd_limit_range=point_cloud_range,
    filter_by_camera_fov=False)
test_evaluator = val_evaluator
# ================= 5. 加载预训练权重进行微调 =================
# 【核心】请替换为你之前在 nuScenes 上训练好的 BEVFusion (带 TransFusionHead) 权重路径
# load_from = 'data/bevfusion_fixed.pth'
# load_from = 'data/bevfusion_merged_init_v4.pth'

lr = 2.5e-4 # 微调阶段保持柔和的学习率
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=15, end=15, by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=60, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# 删掉我们刚才加的 visualization=dict(...)，只保留下面这两行
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# 确保这段依然保留在文件底部
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer')

#load_from = 'data/work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_kitti-3d_nopretrained/epoch_15.pth'

