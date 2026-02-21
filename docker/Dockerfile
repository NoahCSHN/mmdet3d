# ==============================================================================
# 基于你的 Ubuntu 22.04 / Kernel 6.8 环境优化的 Dockerfile
# ==============================================================================

# 1. 升级基础镜像：使用支持 CUDA 12.1 和 PyTorch 2.x 的官方镜像
# 这解决了显卡算力不匹配 (sm_120/sm_89) 和显存管理的问题
ARG PYTORCH="2.1.2"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# 2. 扩展显卡架构支持列表
# 6.0/6.1=Pascal, 7.0/7.5=Volta/Turing, 8.0/8.6=Ampere(30系), 8.9=Ada(40系), 9.0=Hopper(未来兼容)
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

# 3. 修复 GPG Key 和源问题 (关键修复)
# 删除旧的 NVIDIA 签名，防止 apt-get update 报错
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && rm -f /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 || true

# (可选) 如果在国内，为了下载速度，可以取消下面几行的注释来换源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装系统依赖库
# 增加 DEBIAN_FRONTEND=noninteractive 防止安装时卡在时区选择界面
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. 安装 OpenMMLab 核心组件 (适配 PyTorch 2.x)
# 注意：这里指定了 mmcv>=2.1.0 以配合 CUDA 12
RUN pip install -U openmim && \
    mim install "mmengine>=0.10.0" "mmcv>=2.1.0" "mmdet>=3.0.0"

# 6. 安装 MMDetection3D (当前目录)
WORKDIR /mmdetection3d
COPY . /mmdetection3d
RUN pip install -v -e .

# ==============================================================================
# (可选) 针对 BEVFusion 的预编译优化
# 如果你是为了跑 projects/BEVFusion，建议取消下面几行的注释，
# 这样启动容器时就不用每次都等编译了。
# ==============================================================================
WORKDIR /mmdetection3d/projects/BEVFusion
# # 修复旧代码中的路径问题和 PyTorch 版本兼容性
RUN sed -i 's|projects.BEVFusion.||g' setup.py || true
RUN find . -type f \( -name "*.cpp" -o -name "*.cu" \) -exec sed -i 's/AT_CHECK/TORCH_CHECK/g' {} + || true
RUN pip install -v -e .
WORKDIR /mmdetection3d
