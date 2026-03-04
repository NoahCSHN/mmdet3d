# 1. 继承你刚才上传的配置文件 (请确保文件名和路径正确)
_base_ = ['./bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_kitti-3d.py']

# ================= 1. 基础物理范围与类别 =================
# 【核心修改】：把类别严格限制为 'car' 1 个类
class_names = ['car']
metainfo = dict(classes=class_names)
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/' # 确保你的 nuScenes 数据集放在这个目录下
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')
input_modality = dict(use_lidar=True, use_camera=True)

# 恢复 360 度全景物理范围，包容你的所有点云
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]

# 精确重算 Grid Size: (54 - (-54)) / 0.075 = 1440
grid_size = [1440, 1440, 40]
# ================= 2. BEVFusion 模型架构微调 =================
model = dict(
    # nuScenes 的点云包含 5 个维度 (x, y, z, intensity, ring)
    pts_voxel_encoder=dict(num_features=5),

    # 2. 【核心修复】：稀疏主干网络也必须接收 5 维输入！覆盖掉 base 里的 4
    pts_middle_encoder=dict(in_channels=5),
    
    bbox_head=dict(
        num_classes=1, 
        train_cfg=dict(
            dataset='NuScenesDataset',
            # 恢复 nuScenes 的 10 维权重 (比 KITTI 多了最后的两个速度维度 0.2, 0.2)
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2] 
        ),
        test_cfg=dict(dataset='NuScenesDataset'),
        # 恢复 nuScenes 必须预测的速度分支 (vel)
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(code_size=10) # 尺寸由 KITTI 的 8 恢复为 10
    )
)

# ================= 3. 重写数据流水线 (切断 KITTI 补丁，使用 6 相机加载流) =================
# 注意：这里我们只改变数据“预处理”的方式，处理完后依然送进你配置好的 LSS 和 Fusion Layer
train_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color'), # 正规的 6 相机加载算子
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    
    dict(type='ObjectNameFilter', classes=class_names), # 【核心验证：只放行 car】
    
    dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.38, 0.55], bot_pct_lim=[0.0, 0.0], rot_lim=[-5.4, 5.4], rand_flip=True, is_train=True),
    # 使用 BEVFusion 官方的空间增强算子，彻底避免 4D 张量打包崩溃
    dict(type='BEVFusionGlobalRotScaleTrans', scale_ratio_range=[0.9, 1.1], rot_range=[-0.78539816, 0.78539816], translation_std=[0.5, 0.5, 0.5]),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'],
        # 不需要我们手写的补丁了，用标准的多视角 meta_keys
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

test_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color'),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=9, load_dim=5, use_dim=5, pad_empty_sweeps=True, remove_close=True),
    dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.48, 0.48], bot_pct_lim=[0.0, 0.0], rot_lim=[0.0, 0.0], rand_flip=False, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

# ================= 4. 安全替换 Dataloader 与 评估器 =================
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset', # 解决类别不平衡
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR'))
)

val_dataloader = dict(
    batch_size=1, 
    num_workers=2, 
    persistent_workers=True, 
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, 
        data_root=data_root, 
        ann_file='nuscenes_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline, 
        metainfo=metainfo, 
        modality=input_modality,
        test_mode=True, 
        box_type_3d='LiDAR')
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='NuScenesMetric', 
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl', 
    metric='bbox')
test_evaluator = val_evaluator

