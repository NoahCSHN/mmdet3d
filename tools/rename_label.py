import pickle
import os

# 请确保路径指向你真实的数据存储目录
DATA_ROOT = './data/3DBox_Annotation_20260305154810'

pkl_files = ['kitti_infos_train.pkl', 'kitti_infos_val.pkl', 'kitti_infos_trainval.pkl']

for pkl_name in pkl_files:
    pkl_path = os.path.join(DATA_ROOT, pkl_name)
    if not os.path.exists(pkl_path): continue
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 强制物理覆盖元数据
    data['metainfo'] = {'classes': ('Distance_Marker', 'Structure')}
    
    actual_names_seen = set()
    fixed_count = 0
    
    for frame in data.get('data_list', []):
        valid_instances = []
        for inst in frame.get('instances', []):
            current_name = inst.get('bbox_label_name', 'Unknown')
            actual_names_seen.add(current_name)
            
            # 【暴力修复逻辑】：根据 label 索引强制映射
            # 假设 0 是 Marker, 1 是 Structure (这是由 create_custom_data.py 决定的)
            lbl = inst.get('bbox_label', -1)
            
            if lbl == 0 or current_name == 'MarkBand' or current_name == '0':
                inst['bbox_label'] = 0
                inst['bbox_label_name'] = 'Distance_Marker'
                valid_instances.append(inst)
                fixed_count += 1
            elif lbl == 1 or current_name == '1':
                inst['bbox_label'] = 1
                inst['bbox_label_name'] = 'Structure'
                valid_instances.append(inst)
                fixed_count += 1
        
        frame['instances'] = valid_instances

    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"--------------------------------------------------")
    print(f"📦 文件: {pkl_name}")
    print(f"🔍 实际发现的脏标签名: {actual_names_seen}")
    print(f"✅ 强行修复数量: {fixed_count}")

print(f"--------------------------------------------------")
print("🚀 所有数据已物理对齐到 ('Distance_Marker', 'Structure')！")
