# 1. 继承仓库中原有的 BEVFusion (LiDAR+Camera) 配置文件
_base_ = ['./bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

# =======================================================
# 第一部分：运行器级别配置 (Runner-level) —— 解除封印，满血输出
# =======================================================
# 恢复标准 20 轮训练，每 5 轮（或 2 轮）验证一次
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=5)

# 【新增这里】：强制开启梯度裁剪，保护 Backbone 不被大梯度摧毁！
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2) # 遇到大于35的梯度海啸，强行按在35！
)

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=8, eta_min=2.5e-5 * 10, begin=0, end=8, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=12, eta_min=2.5e-5 * 1e-4, begin=8, end=20, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=8, eta_min=0.85 / 0.95, begin=0, end=8, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', T_max=12, eta_min=1, begin=8, end=20, by_epoch=True, convert_to_iter_based=True)
]

# =======================================================
# 第二部分：模型级别配置 (Model-level)
# =======================================================
# 2. 调整 Voxel Size，从默认的 0.075 增大一倍到 0.15
# 这样能将点云的体素数量缩减近4倍，大幅降低 3D 卷积的内存带宽消耗
voxel_size = [0.2, 0.2, 8]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
grid_size = [720, 720, 1]

model = dict(
    type='BEVFusion',
    # =========================================================
    # 【新增这段配置】：覆盖基类，强制开启 Hard Voxelization，输出 3D 张量
    # =========================================================
    pts_voxel_layer=dict(
        max_num_points=32,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(40000, 40000),
        voxelize_reduce=True
    ),

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
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[40000, 40000] 
        ),
        voxelize_cfg=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[40000, 40000], 
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

    # --- 点云分支 (PointPillars 彻底重构) ---
    # 步骤 A：将点云编码为柱体特征 (PillarFeatureNet)
    pts_voxel_encoder=dict(
        _delete_=True,
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        legacy=False
    ),
    # 步骤 B：纯 2D 坐标撒点，彻底消灭 3D 稀疏卷积！
    pts_middle_encoder=dict(
        _delete_=True,
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[720, 720]
    ),
    # 步骤 C：2D 骨干网络 (SECOND)
    # 巧妙设计 4 次下采样，提取出 90x90 和 45x45 的多尺度特征
    pts_backbone=dict(
        _delete_=True,
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256, 512],
        layer_nums=[3, 3, 5, 5],
        layer_strides=[2, 2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    # 步骤 D：特征金字塔 (SECFPN)
    # 将 45x45 放大并与 90x90 拼接，完美输出 256 通道的 90x90 特征图！
    pts_neck=dict(
        _delete_=True,
        type='mmdet3d.SECONDFPN',
        in_channels=[256, 512],
        upsample_strides=[1, 2],
        out_channels=[128, 128],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
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

    fusion_layer=dict(
        _delete_=True,
        type='ConvFuser',
        # 【修改这里】：Camera(80) + LiDAR(128)，完美接住 208 通道
        #in_channels=[80, 256], 
        in_channels=[256], 
        # 输出标准 256 通道给后面的骨干网络
        out_channels=256
    ),

    # 5. 替换 Transformer 检测头为高效的 2D CNN CenterPoint Head
    bbox_head=dict(
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
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8, # 取决于你的网络下采样倍数，通常为8
            voxel_size=voxel_size[:2],
            code_size=9
        ),

        separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,

        # 6. 【预判修复点】为 CenterHead 提供正确的训练与推理配置，避免 Loss 计算维度崩溃
        train_cfg=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=8,
            point_cloud_range=point_cloud_range,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        ),
                
        test_cfg=dict(
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
            nms_thr=0.2
        )
    ),
)

# ---------------------------------------------------------
# 数据加载器 (DataLoader) 配置：榨干轻量化后的显存红利
# ---------------------------------------------------------
# 假设你的 Ubuntu 22.04 训练主机使用的是 24G 显存的显卡 (如 RTX 3090/4090)
# 由于我们将 Swin-T 换成了 ResNet-18，并将 Voxel Size 增大了一倍，
# 显存占用大幅下降，此时可以大胆拉高 batch_size。
# ---------------------------------------------------------
# 数据加载器 (DataLoader) 配置：榨干 24GB 显存与 CPU 多核性能
# ---------------------------------------------------------
train_dataloader = dict(
    # 【起飞】：从 2 跃升到 8！如果训练时 nvidia-smi 显示显存占用不到 18GB，你甚至可以直接改为 16！
    batch_size=8,  
    # 充分利用 CPU 多线程解码 6 路环视图像和打乱点云，防止 GPU 等待数据
    num_workers=8 
)

val_dataloader = dict(batch_size=8, num_workers=8)

test_dataloader = dict(batch_size=8, num_workers=8)

# 【注意】：既然要在 4090 上重新训练，请注释掉或删除这两行！
# 我们要从头开始享受完美的 20 轮收敛，不需要加载之前在 Nano 上挣扎的半成品权重。
# resume = True
# load_from = './data/work_dirs/bevfusion_edge_orin_nano/epoch_1.pth'
load_from = './data/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
