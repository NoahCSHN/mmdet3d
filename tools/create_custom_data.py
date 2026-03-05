import os
import pickle
import functools
from mmdet3d.utils import register_all_modules

# 1. 唤醒所有官方算子
register_all_modules(init_default_scope=True)

from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos
from tools.dataset_converters.create_gt_database import create_groundtruth_database
from mmdet3d.datasets.kitti_dataset import KittiDataset

# ================= 核心配置区 =================
DATA_ROOT = './data/3DBox_Annotation_20260305154810'
# 你的自定义类别列表（必须与 custom_kitti_dataset.py 保持完全一致）
CUSTOM_CLASSES = ['Distance_Marker'] 
# ==============================================
# ================= 源码级黑科技 (强制霸王级 Monkey Patch) =================
original_init = KittiDataset.__init__

@functools.wraps(original_init)
def new_init(self, *args, **kwargs):
    # 【核心重写】：无视一切传参，只要有 data_prefix，强行把 pts 路径改回正常的 velodyne！
    if 'data_prefix' in kwargs:
        kwargs['data_prefix']['pts'] = 'training/velodyne'
    else:
        kwargs['data_prefix'] = dict(pts='training/velodyne', img='training/image_2')
    original_init(self, *args, **kwargs)

KittiDataset.__init__ = new_init

# 强行替换底层类别的 METAINFO
KittiDataset.METAINFO = {
    'classes': tuple(CUSTOM_CLASSES), 
    'palette': [(255, 0, 0)] * len(CUSTOM_CLASSES)
}
# ==================================================================

def main():
    print("🚀 [1/4] 开始提取基础数据 (V1 格式)...")
    kitti.create_kitti_info_file(DATA_ROOT, 'kitti', False)
    
    print("🛠️ [2/4] 转换为 V2 格式 (官方底层会自动将未知类别识别为 DontCare/8)...")
    update_pkl_infos('kitti', out_dir=DATA_ROOT, pkl_path=f'{DATA_ROOT}/kitti_infos_train.pkl')
    update_pkl_infos('kitti', out_dir=DATA_ROOT, pkl_path=f'{DATA_ROOT}/kitti_infos_val.pkl')
    update_pkl_infos('kitti', out_dir=DATA_ROOT, pkl_path=f'{DATA_ROOT}/kitti_infos_trainval.pkl')
    #update_pkl_infos('kitti', out_dir=DATA_ROOT, pkl_path=f'{DATA_ROOT}/kitti_infos_test.pkl')
    
    print("✨ [3/4] 开启‘洗白’操作：劫持 .pkl 文件，重新映射 ID...")
    for pkl_name in ['kitti_infos_train.pkl', 'kitti_infos_val.pkl', 'kitti_infos_trainval.pkl']:
        pkl_path = os.path.join(DATA_ROOT, pkl_name)
        if not os.path.exists(pkl_path): continue
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        count = 0
        for frame in data.get('data_list', []):
            for instance in frame.get('instances', []):
                # 官方默认把不认识的当做 8。我们在这里将其强行映射回正确的 ID (0~7)
                # 注意：目前你只有 1 个类，所以全映射为 0。
                # 未来如果有 8 个类，你需要在这里根据类别的英文名 (instance.get('bbox_label_name')) 进行精确的字典映射
                if instance.get('bbox_label') == 8 or instance.get('bbox_label_name') == 'DontCare':
                    instance['bbox_label'] = 0
                    instance['bbox_label_3d'] = 0
                    instance['bbox_label_name'] = 'Distance_Marker'
                    count += 1
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  -> {pkl_name} 修正完毕，共成功洗白 {count} 个目标！")

    print("🗄️ [4/4] 内存注入官方 Dataset 并生成 GT 数据库...")
    KittiDataset.METAINFO = {'classes': tuple(CUSTOM_CLASSES), 'palette': [(255, 0, 0)] * len(CUSTOM_CLASSES)}
    
    create_groundtruth_database('KittiDataset', DATA_ROOT, 'kitti', f'kitti_infos_train.pkl', relative_path=False, mask_anno_path='instances_train.json', with_mask=False)
    print("✅ 完美收工！数据集生成无懈可击！")

if __name__ == '__main__':
    main()
