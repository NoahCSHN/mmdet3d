import os
import pickle
import functools
import numpy as np
from mmdet3d.utils import register_all_modules
from mmdet3d.datasets.kitti_dataset import KittiDataset

register_all_modules(init_default_scope=True)

from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos
from tools.dataset_converters.create_gt_database import create_groundtruth_database

DATA_ROOT = './data/3DBox_Annotation_20260305154810'
CUSTOM_CLASSES = ['Distance_Marker', 'Structure'] 

# ================= 源码级黑科技 (强制霸王级 Monkey Patch) =================
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
# ==================================================================

def main():
    print("🚀 [1/5] 开始提取基础数据 (V1 格式)...")
    kitti.create_kitti_info_file(DATA_ROOT, 'kitti', False)
    
    print("🥷 [2/5] 内存替身术：在 V1 .pkl 阶段强行劫持 Numpy 数组...")
    pkl_files = ['kitti_infos_train.pkl', 'kitti_infos_val.pkl', 'kitti_infos_trainval.pkl']
    for pkl_name in pkl_files:
        pkl_path = os.path.join(DATA_ROOT, pkl_name)
        if not os.path.exists(pkl_path): continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # V1 格式下，data 是一个列表
        for info in data:
            if 'annos' in info and 'name' in info['annos']:
                # 【制胜关键】：将 numpy 字符串数组转为普通列表操作，避开类型截断
                names = info['annos']['name'].tolist()
                for i in range(len(names)):
                    if names[i] == 'Distance_Marker':
                        names[i] = 'Pedestrian'
                    elif names[i] == 'Structure':
                        names[i] = 'Cyclist'
                # 重新塞回 Numpy 数组
                info['annos']['name'] = np.array(names)
                
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)

    print("🛠️ [3/5] 转换为 V2 格式 (官方会乖乖把 Pedestrian 转为 0，Cyclist 转为 1)...")
    for pkl_name in pkl_files:
        if os.path.exists(os.path.join(DATA_ROOT, pkl_name)):
            update_pkl_infos('kitti', out_dir=DATA_ROOT, pkl_path=os.path.join(DATA_ROOT, pkl_name))
            
    print("✨ [4/5] 终极洗白：官方已删除了名字，我们直接收网捞取 0 和 1！")
    for pkl_name in pkl_files:
        pkl_path = os.path.join(DATA_ROOT, pkl_name)
        if not os.path.exists(pkl_path): continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        data['metainfo'] = {'classes': tuple(CUSTOM_CLASSES)}
        
        count_marker = 0
        count_struct = 0
        
        for frame in data.get('data_list', []):
            valid_instances = []
            for inst in frame.get('instances', []):
                # 官方 V2 只保留了 ID。因为我们伪装过，0 就是 Marker，1 就是 Structure！
                lbl = inst.get('bbox_label', -1)
                if lbl == 0:  
                    count_marker += 1
                    valid_instances.append(inst)
                elif lbl == 1: 
                    count_struct += 1
                    valid_instances.append(inst)
                    
            # 过滤掉所有杂质和幽灵标签
            frame['instances'] = valid_instances 
                    
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  -> {pkl_name} 修正完毕，Marker: {count_marker}, Structure: {count_struct}")

    print("🗄️ [5/5] 内存注入官方 Dataset 并生成多类别 GT 数据库...")
    create_groundtruth_database('KittiDataset', DATA_ROOT, 'kitti', 'kitti_infos_train.pkl', relative_path=False, mask_anno_path='instances_train.json', with_mask=False)
    print("✅ 完美收工！数据底层逻辑已全部打通！")

if __name__ == '__main__':
    main()
