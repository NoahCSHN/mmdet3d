import torch

# 1. 指定官方预训练权重的路径
old_ckpt_path = './data/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
# 2. 指定改名后新权重的保存路径
new_ckpt_path = './data/centerpoint_02pillar_renamed_for_bevfusion.pth'

print("正在加载原始预训练权重...")
ckpt = torch.load(old_ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict']

new_state_dict = {}
count = 0

print("开始执行批量改名手术...")
for k, v in state_dict.items():
    # 如果发现以 'pts_bbox_head' 开头的参数，强行把 'pts_' 砍掉！
    if k.startswith('pts_bbox_head.'):
        new_key = k.replace('pts_bbox_head.', 'bbox_head.')
        new_state_dict[new_key] = v
        count += 1
    else:
        new_state_dict[k] = v

# 把修改后的字典塞回权重文件
ckpt['state_dict'] = new_state_dict

# 保存新的权重文件
torch.save(ckpt, new_ckpt_path)
print(f"改名大功告成！共替换了 {count} 个参数的名字。")
print(f"新的权重已保存至: {new_ckpt_path}")
