# 1. 继承仓库中原有的 BEVFusion (LiDAR+Camera) 配置文件
_base_ = ['./bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

# 2. 调整 Voxel Size，从默认的 0.075 增大一倍到 0.15
# 这样能将点云的体素数量缩减近4倍，大幅降低 3D 卷积的内存带宽消耗
voxel_size = [0.15, 0.15, 0.2]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
grid_size = [720, 720, 40]
# 对应的 BEV 特征图尺寸会减小，提升融合后的 2D CNN 处理速度
sparse_shape = [720, 720, 41]  # 原本是 [41, 1440, 1440]

model = dict(
    # 修改点云预处理器中的 Voxel 大小
    data_preprocessor=dict(
        _delete_=True,
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[90000, 120000]
        ),
        # 核心修复点：显式提供 voxelize_cfg
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[90000, 120000],
            voxelize_reduce=True
        )
    ),

    # 3. 图像骨干网络轻量化 (Swin-T 替换为 ResNet-18)
    # 在你的配置中，这三个参数构成了一套极其精密的“微调（Fine-tune）组合拳”：
    # frozen_stages=1：直接把 ResNet-18 的最底层（Stem 和 Stage 1）彻底锁死（包含卷积核和 BN 层的所有参数），因为底层提取的边缘、纹理特征是通用的。
    # norm_eval=True：对未锁死的 Stage 2, 3, 4，冻结它们的 BN 均值和方差，防止小 Batch Size 带来统计噪声。
    # requires_grad=True：在未锁死的 Stage 2, 3, 4 中，允许 BN 层的 $\gamma$ 和 $\beta$ 随着卷积核一起更新，让高层语义特征能够完美适配当前的 3D 检测任务。
    img_backbone=dict(
        _delete_=True,
        type='mmdet.ResNet',
        depth=18,                  # 指定 ResNet-18
        num_stages=4,
        out_indices=(1, 2, 3),     # 输出最后三层的特征
        frozen_stages=1,      # 1, 冻结（Freeze）ResNet 的 Stem（初始卷积层）和 Stage 1（第一个残差Block）的权重，在反向传播时不更新它们的梯度。
        norm_cfg=dict(type='BN', requires_grad=True), 
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
    # 视图转换：解决 180 的问题
    # 如果 top-level 的 voxel_size 没生效，我们在这里硬编码
    view_transform=dict(
        _delete_=True,
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88], 
        xbound=[-54.0, 54.0, 0.6], # 强制步长 0.15
        ybound=[-54.0, 54.0, 0.6],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2  # 强制下采样8倍，输出必定为 90
    ),

    pts_backbone=dict(
        in_channels=128*5,
    ),

    # 调整 LiDAR 稀疏编码器的形状以匹配新的 voxel_size
    pts_middle_encoder=dict(
        _delete_=True, # 强烈建议加上这个，防止旧参数干扰
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=sparse_shape, # 必须匹配你的新 voxel_size 定义
        order=('conv', 'norm', 'act'),
        block_type='basicblock',
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        # 解决 89 维度问题的关键：全 1 padding
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        base_channels=16
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
    ),

    # 6. 【预判修复点】为 CenterHead 提供正确的训练与推理配置，避免 Loss 计算维度崩溃
    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=8,
            point_cloud_range=point_cloud_range,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
            
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2))
)

# ---------------------------------------------------------
# 数据加载器 (DataLoader) 配置：榨干轻量化后的显存红利
# ---------------------------------------------------------
# 假设你的 Ubuntu 22.04 训练主机使用的是 24G 显存的显卡 (如 RTX 3090/4090)
# 由于我们将 Swin-T 换成了 ResNet-18，并将 Voxel Size 增大了一倍，
# 显存占用大幅下降，此时可以大胆拉高 batch_size。

train_dataloader = dict(
    batch_size=4,  # 【关键修改】从默认的 4 提升到 8，如果显存依然有余量，可以尝试 12 或 16
    num_workers=4, # 对应提升 worker 数量以保证数据读取速度跟上 GPU 训练速度
)

# 验证集通常不需要计算梯度，显存占用更小，也可以适当调高加快评估速度
val_dataloader = dict(
    batch_size=4,
    num_workers=4
)

test_dataloader = dict(
    batch_size=4,
    num_workers=4
)

