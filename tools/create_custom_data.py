import os
import pickle
import functools
import numpy as np
from mmdet3d.utils import register_all_modules
from mmdet3d.datasets.kitti_dataset import KittiDataset

# 【关键】：导入你的自定义类，确保 Registry 能找到 CustomKittiDataset
import custom_kitti_dataset 

register_all_modules(init_default_scope=True)

from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos
from tools.dataset_converters.create_gt_database import create_groundtruth_database

DATA_ROOT = './data/3DBox_Annotation_20260305154810'
CUSTOM_CLASSES = ['Distance_Marker', 'Structure'] 

# ================= 源码级补丁 =================
original_init = KittiDataset.__init__
@functools.wraps(original_init)
def new_init(self, *args, **kwargs):
    if 'data_prefix' in kwargs:
        kwargs['data_prefix']['pts'] = 'training/velodyne'
    else:
        kwargs['data_prefix'] = dict(pts='training/velodyne', img='training/image_2')
    original_init(self, *args, **kwargs)

KittiDataset.__init__ = new_init

KittiDataset.METAINFO = {
    'classes': tuple(CUSTOM_CLASSES), 
    'palette': [(255, 0, 0), (0, 255, 0)] 
}
# =================================================

def main():
    print("🚀 [1/4] 开始提取基础数据 (V1 格式)...")
    kitti.create_kitti_info_file(DATA_ROOT, 'kitti', False)
    
    print("🛠️ [2/4] 转换为 V2 格式 (依赖你修改的源码，自动映射为 0 和 1)...")
    pkl_files = ['kitti_infos_train.pkl', 'kitti_infos_val.pkl', 'kitti_infos_trainval.pkl']
    for pkl_name in pkl_files:
        if os.path.exists(os.path.join(DATA_ROOT, pkl_name)):
            update_pkl_infos('kitti', out_dir=DATA_ROOT, pkl_path=os.path.join(DATA_ROOT, pkl_name))
            
    print("✨ [3/4] 终极物理洗白：主动显式注入字符串 name 和 3D 标签！")
    for pkl_name in pkl_files:
        pkl_path = os.path.join(DATA_ROOT, pkl_name)
        if not os.path.exists(pkl_path): continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        data['metainfo'] = {
            'classes': tuple(CUSTOM_CLASSES),
            'palette': [(255, 0, 0), (0, 255, 0)]
        }
        
        count_marker = 0
        count_struct = 0
        
        for frame in data.get('data_list', []):
            valid_instances = []
            for inst in frame.get('instances', []):
                # 此时的 lbl 已经是源码映射好的 0 (Distance_Marker), 1 (Structure), 或 -1 (DontCare)
                lbl = inst.get('bbox_label', -1)
                
                if lbl == 0:  
                    inst['bbox_label'] = 0
                    inst['bbox_label_3d'] = 0         # 补齐 3D 专属 ID
                    #inst['bbox_label_name'] = 'Distance_Marker' # 核心！防止底层强转出 '0'
                    valid_instances.append(inst)
                    count_marker += 1
                elif lbl == 1: 
                    inst['bbox_label'] = 1
                    inst['bbox_label_3d'] = 1         # 补齐 3D 专属 ID
                    #inst['bbox_label_name'] = 'Structure'       # 核心！防止底层强转出 '1'
                    valid_instances.append(inst)
                    count_struct += 1
                # 我们根本不 append lbl == -1 的数据，彻底消灭 DontCare 的干扰
                    
            frame['instances'] = valid_instances 
                    
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  -> {pkl_name} 补全完毕，Marker: {count_marker}, Structure: {count_struct}")

    print("🗄️ [4/4] 内存注入官方 Dataset 并生成多类别 GT 数据库...")
    create_groundtruth_database(
        'CustomKittiDataset', # 使用你注册的名字
        DATA_ROOT, 
        'kitti', 
        'kitti_infos_train.pkl', 
        relative_path=False, 
        mask_anno_path='instances_train.json', 
        with_mask=False
    )
    print("✅ 完美收工！GT Database 生成彻底打通！")

if __name__ == '__main__':
    main()
