"""
检查 metadata.h5 中的字段
"""
import h5py
import numpy as np

metadata_file = r"C:\MySoftware\dfl-pytorch\DeepFaceLab-Torch-main\workspace\data_dst\aligned\metadata.h5"

print(f"读取: {metadata_file}\n")

with h5py.File(metadata_file, 'r') as f:
    for safe_name in f.keys():
        grp = f[safe_name]
        print(f"文件: {grp.attrs.get('__original_filename__', safe_name)}")
        print(f"  Keys (datasets): {list(grp.keys())}")
        print(f"  Attrs: {dict(grp.attrs)}")
        
        # 读取 landmarks
        if 'landmarks' in grp:
            lm = grp['landmarks'][:]
            print(f"\n  landmarks 形状: {lm.shape}")
            print(f"  landmarks 类型: {type(lm)}")
            print(f"  landmarks 前3个点:\n{lm[:3]}")
