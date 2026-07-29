import os
from pathlib import Path

# 删除损坏的HDF5文件
metadata_file = Path("workspace/data_dst/aligned/metadata.h5")
backup_file = Path("workspace/data_dst/aligned/metadata_backup.h5")

if metadata_file.exists():
    print(f"删除损坏的文件: {metadata_file}")
    metadata_file.unlink()
    print("✓ 已删除")

if backup_file.exists():
    print(f"删除备份文件: {backup_file}")
    backup_file.unlink()
    print("✓ 已删除")

print("\n现在可以重新运行排序了")
