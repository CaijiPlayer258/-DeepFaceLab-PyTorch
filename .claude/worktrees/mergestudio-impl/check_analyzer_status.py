"""
快速诊断脚本：检查Analyzer的元数据和模型状态
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_metadata():
    """检查元数据文件状态"""
    aligned_dir = project_root / "workspace" / "data_dst" / "aligned"
    metadata_file = aligned_dir / "metadata.h5"
    
    print("="*80)
    print("DeepFaceLab-Torch Analyzer 诊断报告")
    print("="*80)
    print()
    
    # 1. 检查数据集目录
    print(f"1. 数据集目录: {aligned_dir}")
    if not aligned_dir.exists():
        print(f"   ✗ 目录不存在！")
        return
    print(f"   ✓ 目录存在")
    
    # 统计图片数量
    image_files = list(aligned_dir.glob("*.jpg")) + list(aligned_dir.glob("*.png"))
    print(f"   图片数量: {len(image_files)}")
    print()
    
    # 2. 检查元数据文件
    print(f"2. 元数据文件: {metadata_file}")
    if not metadata_file.exists():
        print(f"   ✗ 文件不存在（需要先运行Analyzer）")
    else:
        print(f"   ✓ 文件存在")
        file_size_mb = metadata_file.stat().st_size / (1024 * 1024)
        print(f"   文件大小: {file_size_mb:.2f} MB")
        
        # 读取HDF5文件统计
        try:
            import h5py
            with h5py.File(metadata_file, 'r') as f:
                total_entries = len(f.keys())
                print(f"   元数据条目数: {total_entries}")
                
                # 检查第一个条目的内容
                if total_entries > 0:
                    first_key = list(f.keys())[0]
                    first_grp = f[first_key]
                    
                    print(f"\n   示例条目 ({first_key}):")
                    print(f"     - Keys: {list(first_grp.keys())}")
                    print(f"     - Attributes: {list(first_grp.attrs.keys())}")
                    
                    # 统计各字段数量（检查keys和attrs）
                    has_phash = sum(1 for k in f.keys() if 'phash' in f[k].attrs or 'phash' in f[k])
                    has_embedding = sum(1 for k in f.keys() if 'embedding' in f[k].attrs or 'embedding' in f[k])
                    has_landmarks = sum(1 for k in f.keys() if 'landmarks' in f[k].attrs or 'landmarks' in f[k])
                    has_histogram = sum(1 for k in f.keys() if 'histogram_rgb' in f[k].attrs or 'histogram_rgb' in f[k])
                    has_pose = sum(1 for k in f.keys() if 'pitch' in f[k].attrs or 'pitch' in f[k])
                    
                    print(f"\n   字段统计:")
                    print(f"     - phash: {has_phash}/{total_entries}")
                    print(f"     - embedding: {has_embedding}/{total_entries}")
                    print(f"     - landmarks: {has_landmarks}/{total_entries}")
                    print(f"     - histogram: {has_histogram}/{total_entries}")
                    print(f"     - pose: {has_pose}/{total_entries}")
        except Exception as e:
            print(f"   ✗ 读取失败: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    
    # 3. 检查ArcFace模型
    print(f"3. ArcFace模型文件:")
    model_files = [
        project_root / "modelhub" / "w600k_mbf.onnx",
        project_root / "modelhub" / "w600k_r50.onnx"
    ]
    
    for model_file in model_files:
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"   ✓ {model_file.name} ({size_mb:.2f} MB)")
        else:
            print(f"   ✗ {model_file.name} (不存在)")
    
    print()
    
    # 4. 检查ONNX Runtime GPU支持
    print(f"4. ONNX Runtime配置:")
    try:
        import onnxruntime as rt
        providers = rt.get_available_providers()
        print(f"   可用提供者: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print(f"   ✓ GPU加速已启用 (CUDA)")
        elif 'DmlExecutionProvider' in providers:
            print(f"   ✓ GPU加速已启用 (DirectML)")
        else:
            print(f"   ✗ 仅CPU模式（建议安装onnxruntime-gpu）")
    except Exception as e:
        print(f"   ✗ 检查失败: {e}")
    
    print()
    print("="*80)
    print("诊断完成")
    print("="*80)
    print()
    print("建议:")
    print("1. 如果元数据为0，请运行: python FacesetProcessor/Analyzer.py --input workspace/data_dst/aligned --features all")
    print("2. 如果需要GPU加速，请安装: pip install onnxruntime-gpu")
    print("3. 查看完整帮助: python FacesetProcessor/Analyzer.py --help")

if __name__ == '__main__':
    check_metadata()
