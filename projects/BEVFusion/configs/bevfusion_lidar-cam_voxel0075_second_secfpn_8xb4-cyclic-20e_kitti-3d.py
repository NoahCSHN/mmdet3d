_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion', 'custom_kitti_dataset'], 
    allow_failed_imports=False)

# ================= 1. 基础物理范围与类别 =================
class_names = ['MarkBand']
metainfo = dict(classes=class_names)
dataset_type = 'CustomKittiDataset'
data_root = 'data/dataset_v3/'

# 【核心修复】：明确指定点云和图像的子目录前缀！
data_prefix = dict(
    pts='training/velodyne', 
    img='training/image_2')

# 【精算后的 KITTI 前视范围】
# X: 0 到 74.4 (74.4 / 0.075 = 992)
# Y: -40.8 到 40.8 (81.6 / 0.075 = 1088)
# Z: -3.0 到 1.0 (4.0 / 0.2 = 20)
point_cloud_range = [0.0, -40.8, -3.0, 74.4, 40.8, 1.0]
voxel_size = [0.075, 0.075, 0.2]
grid_size = [992, 1088, 20] 

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
            
    # ------ LiDAR 分支 ------
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4), # KITTI 点云通常只有 4 维 (x,y,z,intensity)
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=4,
        sparse_shape=[1088, 992, 20], # [Y, X, Z] 对应 grid_size
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
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
        convert_weights=True),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
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
        xbound=[0.0, 74.4, 0.3], 
        # Y 轴 272 个 bin，经过 downsample=2 后特征图 Y 宽为 136 (与激光雷达 1088/8=136 完美对齐)
        ybound=[-40.8, 40.8, 0.3], 
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2),
        
    fusion_layer=dict(
        type='ConvFuser', in_channels=[80, 256], out_channels=256),

    # ------ TransFusion 检测头 ------
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512, # 保持预训练模型的输入通道数
        hidden_channel=128,
        num_classes=1,   # 【只预测 MarkBand 1 类】
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
            pc_range=[0.0, -40.8], # 仅前视
            nms_type=None),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]), # 去掉了 vel=[2, 2]
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[0.0, -40.8],
            post_center_range=[0.0, -40.8, -5.0, 74.4, 40.8, 5.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            code_size=8), # 尺寸由 10 变为 8
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
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
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['cam2img', 'lidar2cam', 'img_shape', 'box_type_3d', 'sample_idx', 'lidar_path', 'img_path'])
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
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points'],
        meta_keys=['cam2img', 'lidar2cam', 'img_shape', 'box_type_3d', 'sample_idx', 'lidar_path', 'img_path'])
]

# ================= 4. Dataloader 与 训练策略 =================
train_dataloader = dict(
    batch_size=2, # 多模态显存占用极大，建议调低
    num_workers=0,
    persistent_workers=False,
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
    batch_size=1, num_workers=0, persistent_workers=False, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root, ann_file='kitti_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline, metainfo=metainfo, modality=input_modality,
        test_mode=True, box_type_3d='LiDAR'))
test_dataloader = val_dataloader

val_evaluator = dict(type='KittiMetric', ann_file=data_root + 'kitti_infos_val.pkl', metric='bbox')
test_evaluator = val_evaluator

# ================= 5. 加载预训练权重进行微调 =================
# 【核心】请替换为你之前在 nuScenes 上训练好的 BEVFusion (带 TransFusionHead) 权重路径
# load_from = 'data/bevfusion_fixed.pth'
# load_from = 'data/bevfusion_merged_init_v4.pth'

lr = 2.5e-5 # 微调阶段保持柔和的学习率
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=15, end=15, by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=15, val_interval=1)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))
