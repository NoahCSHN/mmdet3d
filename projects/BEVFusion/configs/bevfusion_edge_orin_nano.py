# 1. 继承仓库中原有的 BEVFusion (LiDAR+Camera) 配置文件
_base_ = ['./bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

# 2. 调整 Voxel Size，从默认的 0.075 增大一倍到 0.15
# 这样能将点云的体素数量缩减近4倍，大幅降低 3D 卷积的内存带宽消耗
voxel_size = [0.15, 0.15, 0.2]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# 对应的 BEV 特征图尺寸会减小，提升融合后的 2D CNN 处理速度
sparse_shape = [41, 720, 720]  # 原本是 [41, 1440, 1440]

model = dict(
    # 修改点云预处理器中的 Voxel 大小
    data_preprocessor=dict(
        voxelize_cfg=dict(
            voxel_size=voxel_size,
            max_num_points=10,
            max_voxels=[60000, 90000] # 因为 voxel 变大，最大 voxel 数量也可以下调以节省显存
        )
    ),

    # 3. 图像骨干网络轻量化 (Swin-T 替换为 MobileNetV3)
    # img_backbone=dict(
    #     _delete_=True, # 删除原本的 SwinTransformer
    #     type='mmcls.MobileNetV3',
    #     arch='small',
    #     out_indices=(3, ), # 提取单层特征，避免多尺度带来的额外开销
    #     init_cfg=dict(
    #         type='Pretrained', 
    #         checkpoint='open-mmlab://mmcls/mobilenet_v3_small'
    #     )
    # ),
    # img_neck=dict(
    #     _delete_=True,
    #     type='FPN',
    #     in_channels=[96],  # MobileNetV3 small stage 3 的通道数
    #     out_channels=256,
    #     num_outs=1
    # ),
    # ResNet-18
    img_backbone=dict(
        _delete_=True,
        type='mmdet.ResNet',
        depth=18,                  # 指定 ResNet-18
        num_stages=4,
        out_indices=(1, 2, 3),     # 输出最后三层的特征
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False), # 冻结 BN，边缘部署常见操作
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
    ),
    
    img_neck=dict(
        _delete_=True,
        type='GeneralizedLSSFPN',
        in_channels=[128, 256, 512], # 对应 ResNet-18 layer 2,3,4 的输出通道
        out_channels=256,            # FPN 统一输出通道，必须与 View Transformer 匹配
        start_level=0,
        num_outs=3,
        act_cfg=dict(type='ReLU', inplace=True)
    ),
    
    # 4. 优化 LSS 视图转换模块
    img_view_transformer=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],     # 可考虑进一步降分辨率到 [128, 352]
        feature_size=[8, 22],      # 对应图像输入降采样后的特征图尺寸
        x_bound=[-54.0, 54.0, 0.6],# BEV Grid 步长增大
        y_bound=[-54.0, 54.0, 0.6],
        z_bound=[-5.0, 3.0, 8.0],
        d_bound=[1.0, 60.0, 1.0],  # 关键：将深度切片的步长从 0.5 改为 1.0，深度预测计算量减半
        downsample=2
    ),

    # 调整 LiDAR 稀疏编码器的形状以匹配新的 voxel_size
    pts_middle_encoder=dict(
        sparse_shape=sparse_shape
    ),

    # 5. 替换 Transformer 检测头为高效的 2D CNN CenterPoint Head
    pts_bbox_head=dict(
        _delete_=True, # 彻底删除耗时的 TransFusionHead
        type='CenterHead',
        in_channels=256, # 假设图像特征和LiDAR特征融合后的维度是256
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8, # 取决于你的网络下采样倍数，通常为8
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True
    )
)
