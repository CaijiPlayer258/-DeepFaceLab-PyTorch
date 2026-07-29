"""
检查两个目录是否是同一个文件
"""
import os
from pathlib import Path

file1 = Path(r"C:\MySoftware\dfl-pytorch\DeepFaceLab-Torch-main\workspace\data_dst\aligned\sorted_36585.jpg")
file2 = Path(r"C:\MySoftware\deepfacelab\DFL_RTX5000series_SHENBIAN_PytorchVersion\DFL_RTX5000series_SHENBIAN_PytorchVersion\workspace\data_src\aligned\sorted_36585.jpg")

print(f"文件1: {file1}")
print(f"  存在: {file1.exists()}")
if file1.exists():
    print(f"  文件大小: {file1.stat().st_size} bytes")
    print(f"  修改时间: {file1.stat().st_mtime}")
    print(f"  绝对路径: {file1.resolve()}")

print(f"\n文件2: {file2}")
print(f"  存在: {file2.exists()}")
if file2.exists():
    print(f"  文件大小: {file2.stat().st_size} bytes")
    print(f"  修改时间: {file2.stat().st_mtime}")
    print(f"  绝对路径: {file2.resolve()}")

if file1.exists() and file2.exists():
    # 检查是否是同一个文件（相同的 inode）
    stat1 = file1.stat()
    stat2 = file2.stat()
    
    print(f"\n对比:")
    print(f"  文件大小相同: {stat1.st_size == stat2.st_size}")
    print(f"  修改时间相同: {stat1.st_mtime == stat2.st_mtime}")
    print(f"  绝对路径相同: {file1.resolve() == file2.resolve()}")
    
    if file1.resolve() == file2.resolve():
        print(f"\n✅ 两个路径指向同一个文件")
    else:
        print(f"\n❌ 两个路径指向不同的文件！")
        print(f"   这是两个独立的副本，您需要分别回写。")
