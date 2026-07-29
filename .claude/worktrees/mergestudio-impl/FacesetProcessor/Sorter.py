"""
DeepFaceLab Torch - FacesetProcessor Sorter Module
人脸集排序器模块：基于多种特征对人脸进行排序
支持多进程、可组合、易扩展的架构
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json
import argparse
from typing import Dict, List, Optional, Tuple, Callable
import tqdm
from multiprocessing import Pool, cpu_count
import operator
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入多语言支持
from strings import S

# 减少ONNX Runtime警告
import onnxruntime
onnxruntime.set_default_logger_severity(3)
import warnings
warnings.filterwarnings('ignore', module='onnxruntime')

# 导入项目模块
from core import pathex
from core.cv2ex import cv2_imread
from facelib.LandmarksProcessor import get_image_hull_mask, estimate_pitch_yaw_roll
from DFLIMG import DFLIMG

# 导入基类
from FacesetBaseProcessor import FacesetBaseProcessor


class SortMethod:
    """排序方法枚举"""
    PHASH = "phash"              # 感知哈希排序
    HIST = "hist"                # 直方图排序
    BLUR = "blur"                # 模糊度排序
    FACE_POSE = "face_pose"      # 人脸姿态排序
    RESOLUTION = "resolution"    # 分辨率排序
    COLOR = "color"              # 颜色排序
    NAME = "name"                # 文件名排序
    ABS_DIFF = "absdiff"         # 绝对差异排序
    MOTION_BLUR = "motion_blur"  # 运动模糊排序
    OCCLUSION = "occlusion"      # 遮挡排序
    LANDMARKS_OCCLUSION = "landmarks_occlusion"  #  landmarks遮挡排序
    MIX_OCCLUSION = "mix_occlusion"  # 混合遮挡排序
    HIST_DISSIM = "hist_dissim"  # 直方图相似度排序
    
    # 所有可用的排序方法
    ALL_METHODS = [
        PHASH, HIST, BLUR, FACE_POSE, RESOLUTION, COLOR, NAME,
        ABS_DIFF, MOTION_BLUR, OCCLUSION, LANDMARKS_OCCLUSION,
        MIX_OCCLUSION, HIST_DISSIM
    ]


def _calculate_phash_worker(args):
    """
    多进程工作函数：计算单张图片的感知哈希
    
    Args:
        args: (image_path_str,)
        
    Returns:
        (filename, phash_value)
    """
    from imagehash import phash
    from PIL import Image
    
    image_path_str = args[0]
    
    try:
        image = Image.open(image_path_str)
        hash_value = phash(image)
        return (Path(image_path_str).name, int(str(hash_value), 16))
    except Exception as e:
        return (Path(image_path_str).name, None)


def _calculate_histogram_worker(args):
    """
    多进程工作函数：计算单张图片的直方图
    
    Args:
        args: (image_path_str,)
        
    Returns:
        (filename, histogram_dict)
    """
    image_path_str = args[0]
    
    try:
        image = cv2_imread(image_path_str)
        if image is None:
            return (Path(image_path_str).name, None)
        
        # RGB直方图
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
        
        # 归一化
        hist_b = (hist_b / hist_b.sum()).tolist()
        hist_g = (hist_g / hist_g.sum()).tolist()
        hist_r = (hist_r / hist_r.sum()).tolist()
        
        return (Path(image_path_str).name, {
            'b': hist_b,
            'g': hist_g,
            'r': hist_r
        })
    except Exception as e:
        return (Path(image_path_str).name, None)


def _calculate_blur_worker(args):
    """
    多进程工作函数：计算单张图片的模糊度
    
    Args:
        args: (image_path_str, landmarks, use_motion_blur)
        
    Returns:
        (filename, blur_value)
    """
    image_path_str, landmarks, use_motion_blur = args
    
    try:
        image = cv2_imread(image_path_str)
        if image is None:
            return (Path(image_path_str).name, 0.0)
        
        # 如果有landmarks，应用mask
        if landmarks:
            hull_mask = get_image_hull_mask(image.shape, np.array(landmarks))
            image = (image * hull_mask).astype(np.uint8)
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if use_motion_blur:
            # 运动模糊检测
            value = cv2.Laplacian(gray, cv2.CV_64F, ksize=11).var()
        else:
            # 普通模糊检测
            value = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return (Path(image_path_str).name, float(value))
    except Exception as e:
        return (Path(image_path_str).name, 0.0)


def _calculate_face_pose_worker(args):
    """
    多进程工作函数：计算单张图片的人脸姿态
    
    Args:
        args: (image_path_str, landmarks)
        
    Returns:
        (filename, pose_dict)
    """
    image_path_str, landmarks = args
    
    try:
        image = cv2_imread(image_path_str)
        if image is None or landmarks is None:
            return (Path(image_path_str).name, None)
        
        # 估计姿态
        pitch, yaw, roll = estimate_pitch_yaw_roll(np.array(landmarks), image.shape[1], image.shape[0])
        
        return (Path(image_path_str).name, {
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll)
        })
    except Exception as e:
        return (Path(image_path_str).name, None)


def _calculate_resolution_worker(args):
    """
    多进程工作函数：计算单张图片的分辨率
    
    Args:
        args: (image_path_str,)
        
    Returns:
        (filename, resolution)
    """
    image_path_str = args[0]
    
    try:
        image = cv2_imread(image_path_str)
        if image is None:
            return (Path(image_path_str).name, 0)
        
        h, w = image.shape[:2]
        resolution = h * w
        
        return (Path(image_path_str).name, resolution)
    except Exception as e:
        return (Path(image_path_str).name, 0)


def _sort_phash_chunk(args):
    """
    多进程工作函数：对单个块进行 phash 贪心排序
    
    Args:
        args: (chunk_indices, file_list, phash_dict)
        
    Returns:
        sorted_original_indices: 排序后的原始索引列表
    """
    chunk_indices, file_list, phash_dict = args
    
    # 块内贪心排序
    sorted_chunk = []
    remaining = set(range(len(chunk_indices)))
    
    if not remaining:
        return []
    
    current_idx = 0
    sorted_chunk.append(current_idx)
    remaining.remove(current_idx)
    
    while remaining:
        current_hash = phash_dict[file_list[chunk_indices[current_idx]]]
        min_distance = float('inf')
        min_index = -1
        
        for i in remaining:
            test_hash = phash_dict[file_list[chunk_indices[i]]]
            xor_result = bin(current_hash ^ test_hash).count('1')
            
            if xor_result < min_distance:
                min_distance = xor_result
                min_index = i
        
        if min_index >= 0:
            sorted_chunk.append(min_index)
            remaining.remove(min_index)
            current_idx = min_index
    
    # 返回原始索引
    return [chunk_indices[i] for i in sorted_chunk]


def _sort_hist_chunk(args):
    """
    多进程工作函数：对单个块进行直方图贪心排序
    
    Args:
        args: (chunk_indices, file_list, hist_arrays_dict)
        
    Returns:
        sorted_original_indices: 排序后的原始索引列表
    """
    chunk_indices, file_list, hist_arrays_dict = args
    
    chunk_files = [file_list[i] for i in chunk_indices]
    chunk_hist = {f: hist_arrays_dict[f] for f in chunk_files}
    
    # 块内贪心排序
    sorted_chunk = []
    remaining = set(range(len(chunk_files)))
    
    if not remaining:
        return []
    
    current_idx = 0
    sorted_chunk.append(current_idx)
    remaining.remove(current_idx)
    
    while remaining:
        current_file = chunk_files[current_idx]
        current_hist_b, current_hist_g, current_hist_r = chunk_hist[current_file]
        
        min_score = float("inf")
        min_index = -1
        
        for i in remaining:
            test_file = chunk_files[i]
            test_hist_b, test_hist_g, test_hist_r = chunk_hist[test_file]
            
            score = (cv2.compareHist(current_hist_b, test_hist_b, cv2.HISTCMP_BHATTACHARYYA) +
                     cv2.compareHist(current_hist_g, test_hist_g, cv2.HISTCMP_BHATTACHARYYA) +
                     cv2.compareHist(current_hist_r, test_hist_r, cv2.HISTCMP_BHATTACHARYYA))
            
            if score < min_score:
                min_score = score
                min_index = i
        
        if min_index >= 0:
            sorted_chunk.append(min_index)
            remaining.remove(min_index)
            current_idx = min_index
    
    # 返回原始索引
    return [chunk_indices[i] for i in sorted_chunk]


class FaceSorter(FacesetBaseProcessor):
    """人脸排序器 - 基于多种特征对人脸进行排序
    
    继承自 FacesetBaseProcessor，自动获得：
    - HDF5 流式访问能力
    - 图像文件扫描
    - 元数据缓存管理
    - 完整性校验
    """
    
    def __init__(self, faceset_path: Path, metadata_file: Optional[Path] = None):
        """
        初始化排序器
        
        Args:
            faceset_path: 人脸数据集路径
            metadata_file: 元数据文件路径（可选，默认为 faceset_path/metadata.h5）
        """
        # 调用父类初始化（会自动初始化 HDF5 访问器和扫描文件）
        super().__init__(faceset_path, metadata_file)
    
    def _save_to_temp_h5(self, new_features: Dict[str, Dict], index: int) -> Optional[Path]:
        """
        将特征保存到临时HDF5文件
        
        Args:
            new_features: {filename: {feature_key: feature_value}}
            index: 临时文件索引
            
        Returns:
            临时文件路径，失败返回None
        """
        if not new_features:
            return None
        
        try:
            import h5py
            temp_file = self.faceset_path / f"_temp_sort_{index}.h5"
            
            with h5py.File(temp_file, 'w') as f:
                for filename, features in new_features.items():
                    safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                    grp = f.create_group(safe_name)
                    grp.attrs['__original_filename__'] = filename
                    
                    for key, value in features.items():
                        if isinstance(value, (list, np.ndarray)):
                            grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                        elif isinstance(value, (int, float, str)):
                            grp.attrs[key] = value
                        else:
                            grp.attrs[key] = str(value)
                
                f.flush()
            
            return temp_file
            
        except Exception as e:
            print(f"\n⚠ 保存临时文件失败: {e}")
            return None
    
    def _merge_temp_h5_files(self, temp_files: List[Path]):
        """
        合并所有临时HDF5文件到主数据库
        
        Args:
            temp_files: 临时HDF5文件列表
        """
        if not temp_files:
            return
        
        try:
            import h5py
            metadata_file = self.faceset_path / "metadata.h5"
            
            # 关闭当前访问器
            if self._h5_accessor is not None:
                self._h5_accessor.close()
            
            # 打开主文件（追加模式）
            with h5py.File(metadata_file, 'a') as main_f:
                # 逐个合并临时文件
                for temp_file in temp_files:
                    if not temp_file.exists():
                        continue
                    
                    with h5py.File(temp_file, 'r') as temp_f:
                        for key in temp_f.keys():
                            # 如果主文件中已存在，先删除
                            if key in main_f:
                                del main_f[key]
                            
                            # 复制到主文件
                            temp_f.copy(key, main_f)
                    
                    main_f.flush()
            
        except Exception as e:
            print(f"\n⚠ 合并临时文件失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_new_features_to_hdf5(self, new_features: Dict[str, Dict]):
        """
        将新计算的特征增量保存到HDF5文件
        
        Args:
            new_features: {filename: {feature_key: feature_value}}
        """
        if not new_features:
            return
        
        try:
            import h5py
            metadata_file = self.faceset_path / "metadata.h5"
            
            # 关闭当前访问器
            if self._h5_accessor is not None:
                self._h5_accessor.close()
            
            # 以追加模式打开HDF5文件
            with h5py.File(metadata_file, 'a') as f:
                for filename, features in new_features.items():
                    safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                    
                    # 如果组已存在，更新它；否则创建新组
                    if safe_name in f:
                        grp = f[safe_name]
                    else:
                        grp = f.create_group(safe_name)
                        grp.attrs['__original_filename__'] = filename
                    
                    # 写入新特征
                    for key, value in features.items():
                        # 删除旧数据（如果存在）
                        if key in grp:
                            del grp[key]
                        
                        if isinstance(value, (list, np.ndarray)):
                            grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                        elif isinstance(value, (int, float, str)):
                            grp.attrs[key] = value
                        else:
                            grp.attrs[key] = str(value)
                
                # 强制flush确保数据写入磁盘
                f.flush()
            
        except Exception as e:
            print(f"\n⚠ 保存新特征失败: {e}")
            import traceback
            traceback.print_exc()
    
    @property
    def image_files(self) -> List[Path]:
        """获取图像文件列表（父类属性的别名）"""
        return self._image_files
    
    def sort_by_method(self, method: str, **kwargs):
        """
        根据指定方法排序
        
        Args:
            method: 排序方法
            **kwargs: 额外参数
            
        Returns:
            排序后的列表 [(filename, value), ...]
        """
        if method not in SortMethod.ALL_METHODS:
            raise ValueError(f"Unsupported sort method: {method}")
        
        print(S('SORTER_SORTING_BY', method))
        
        # 调用对应的排序方法
        if method == SortMethod.PHASH:
            return self._sort_by_phash(**kwargs)
        elif method == SortMethod.HIST:
            return self._sort_by_hist(**kwargs)
        elif method == SortMethod.BLUR:
            return self._sort_by_blur(**kwargs)
        elif method == SortMethod.FACE_POSE:
            return self._sort_by_face_pose(**kwargs)
        elif method == SortMethod.RESOLUTION:
            return self._sort_by_resolution(**kwargs)
        elif method == SortMethod.COLOR:
            return self._sort_by_color(**kwargs)
        elif method == SortMethod.NAME:
            return self._sort_by_name(**kwargs)
        else:
            raise NotImplementedError(f"Sort method '{method}' not implemented yet")
    
    def _sort_by_phash(self, workers: int = None) -> List[Tuple[str, int]]:
        """
        按感知哈希排序（相似的图片排在一起）
        优先从元数据中读取，缺失的才重新计算
        
        Args:
            workers: 工作进程数
            
        Returns:
            排序后的列表
        """
        if workers is None:
            workers = cpu_count()
        
        print(S('SORTER_CALCULATING_PHASH', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的 phash 值
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            # 使用父类的 get_field() 方法，只读取 phash 字段
            phash_value = self.get_field(filename, 'phash')
            
            if phash_value is not None:
                # 元数据中有 phash，直接使用
                try:
                    # 处理不同格式的 phash
                    if isinstance(phash_value, list):
                        # 如果是列表，转换为整数（假设是字节列表）
                        phash_int = int(''.join([str(b) for b in phash_value]), 2)
                    elif isinstance(phash_value, str):
                        # 字符串格式（十六进制）
                        phash_int = int(phash_value, 16)
                    else:
                        # 直接是整数
                        phash_int = int(phash_value)
                    cached_results.append((filename, phash_int))
                except Exception as e:
                    print(f"Warning: Invalid phash value for {filename}: {e}")
                    missing_files.append(img_path)
            else:
                # 元数据中没有 phash，需要计算
                missing_files.append(img_path)
        
        print(f"✓ From metadata: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        
        # 第二步：计算缺失的 phash 值（每10000个保存到临时文件，最后合并）
        newly_calculated = []
        save_interval = 10000
        last_save_count = 0
        temp_h5_files = []  # 记录临时HDF5文件
        
        if missing_files:
            tasks = [(str(img_path),) for img_path in missing_files]
            
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(
                    pool.imap_unordered(_calculate_phash_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing phash",
                    unit="img"
                )
                
                for result in pbar:
                    if result[1] is not None:
                        newly_calculated.append(result)
                        
                        # 每计算save_interval个就保存到临时文件
                        if len(newly_calculated) - last_save_count >= save_interval:
                            batch_features = {}
                            for filename, phash_value in newly_calculated[last_save_count:]:
                                phash_hex = format(phash_value, 'x')
                                batch_features[filename] = {'phash': phash_hex}
                            
                            # 保存到临时HDF5文件
                            temp_file = self._save_to_temp_h5(batch_features, len(temp_h5_files))
                            if temp_file:
                                temp_h5_files.append(temp_file)
                            
                            last_save_count = len(newly_calculated)
                            pbar.set_description(f"Calculating missing phash [{last_save_count} saved to temp]")
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
            
            # 保存剩余的数据到临时文件
            if len(newly_calculated) > last_save_count:
                batch_features = {}
                for filename, phash_value in newly_calculated[last_save_count:]:
                    phash_hex = format(phash_value, 'x')
                    batch_features[filename] = {'phash': phash_hex}
                temp_file = self._save_to_temp_h5(batch_features, len(temp_h5_files))
                if temp_file:
                    temp_h5_files.append(temp_file)
            
            # 合并所有临时文件到主HDF5
            if temp_h5_files:
                print(f"\n  正在合并 {len(temp_h5_files)} 个临时文件到主数据库...")
                self._merge_temp_h5_files(temp_h5_files)
                print(f"  ✓ 合并完成")
                
                # 清理临时文件
                for temp_file in temp_h5_files:
                    try:
                        temp_file.unlink()
                    except:
                        pass
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_SIMILARITY'))
        
        # 第四步：分块并行贪心算法排序
        phash_values = {r[0]: r[1] for r in all_results}
        file_list = [r[0] for r in all_results]
        
        print(f"  总图片数: {len(file_list)}")
        
        # 分块大小
        chunk_size = 5000
        n_chunks = (len(file_list) + chunk_size - 1) // chunk_size
        
        print(f"  分成 {n_chunks} 个块，每块 {chunk_size} 张图片")
        print(f"  使用 {workers} 个进程并行处理...")
        
        # 创建块
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(file_list))
            chunks.append(list(range(start, end)))
        
        # 准备任务参数
        tasks = [(chunk, file_list, phash_values) for chunk in chunks]
        
        # 并行处理每个块
        sorted_indices = []
        with Pool(processes=workers) as pool:
            for chunk_result in tqdm.tqdm(
                pool.imap(_sort_phash_chunk, tasks),
                total=len(chunks),
                desc="Sorting chunks",
                unit="chunk"
            ):
                sorted_indices.extend(chunk_result)
        
        print(f"\n✓ 分块排序完成，共 {len(sorted_indices)} 张图片")
        
        # 构建结果
        sorted_results = [(file_list[idx], phash_values[file_list[idx]]) for idx in sorted_indices]
        
        # 构建新特征数据（将新计算的 phash 保存到元数据）
        new_features = {}
        for filename, phash_value in newly_calculated:
            # 将整数 phash 转换为十六进制字符串
            phash_hex = format(phash_value, 'x')
            new_features[filename] = {'phash': phash_hex}
        
        return sorted_results, new_features
    
    def _sort_by_hist(self, workers: int = None) -> List[Tuple[str, Dict]]:
        """
        按直方图相似度排序（相似的图片排在一起）
        优先从元数据中读取，缺失的才重新计算
        
        Args:
            workers: 工作进程数
            
        Returns:
            排序后的列表
        """
        if workers is None:
            workers = cpu_count()
        
        print(S('SORTER_CALCULATING_HIST', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的 histogram 值
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            # 使用父类的 get_field() 方法，只读取 histogram 字段
            hist_data = self.get_field(filename, 'histogram')
            
            if hist_data is not None:
                # 如果 hist_data 是字符串，尝试解析为字典
                if isinstance(hist_data, str):
                    try:
                        import json
                        hist_data = json.loads(hist_data)
                    except:
                        # 解析失败，跳过该文件
                        missing_files.append(img_path)
                        continue
                
                cached_results.append((filename, hist_data))
            else:
                missing_files.append(img_path)
        
        print(f"✓ From metadata: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        
        # 第二步：计算缺失的直方图（每10000个保存到临时文件，最后合并）
        newly_calculated = []
        save_interval = 10000
        last_save_count = 0
        temp_h5_files = []
        
        if missing_files:
            tasks = [(str(img_path),) for img_path in missing_files]
            
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(
                    pool.imap_unordered(_calculate_histogram_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing histogram",
                    unit="img"
                )
                
                for result in pbar:
                    if result[1] is not None:
                        newly_calculated.append(result)
                        
                        # 每计算save_interval个就保存到临时文件
                        if len(newly_calculated) - last_save_count >= save_interval:
                            batch_features = {}
                            for filename, hist_data in newly_calculated[last_save_count:]:
                                batch_features[filename] = {'histogram': hist_data}
                            
                            temp_file = self._save_to_temp_h5(batch_features, len(temp_h5_files))
                            if temp_file:
                                temp_h5_files.append(temp_file)
                            
                            last_save_count = len(newly_calculated)
                            pbar.set_description(f"Calculating missing histogram [{last_save_count} saved to temp]")
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
            
            # 保存剩余的数据到临时文件
            if len(newly_calculated) > last_save_count:
                batch_features = {}
                for filename, hist_data in newly_calculated[last_save_count:]:
                    batch_features[filename] = {'histogram': hist_data}
                temp_file = self._save_to_temp_h5(batch_features, len(temp_h5_files))
                if temp_file:
                    temp_h5_files.append(temp_file)
            
            # 合并所有临时文件到主HDF5
            if temp_h5_files:
                print(f"\n  正在合并 {len(temp_h5_files)} 个临时文件到主数据库...")
                self._merge_temp_h5_files(temp_h5_files)
                print(f"  ✓ 合并完成")
                
                # 清理临时文件
                for temp_file in temp_h5_files:
                    try:
                        temp_file.unlink()
                    except:
                        pass
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_BY_HIST'))
        
        # 第四步：使用 Bhattacharyya 距离进行分块排序
        # 将直方图数据转换为 numpy 数组以便计算
        hist_arrays = {}
        file_list = [r[0] for r in all_results]
        
        for filename, hist_data in all_results:
            # 将直方图列表转换为 numpy 数组
            hist_b = np.array(hist_data['b'], dtype=np.float32)
            hist_g = np.array(hist_data['g'], dtype=np.float32)
            hist_r = np.array(hist_data['r'], dtype=np.float32)
            hist_arrays[filename] = (hist_b, hist_g, hist_r)
        
        print(S('SORTER_SORTING_BY_HIST'))
        print(f"  总图片数: {len(file_list)}")
        
        # 分块大小
        chunk_size = 5000
        n_chunks = (len(file_list) + chunk_size - 1) // chunk_size
        
        print(f"  分成 {n_chunks} 个块，每块 {chunk_size} 张图片")
        print(f"  使用 {workers} 个进程并行处理...")
        
        # 创建块
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(file_list))
            chunks.append(list(range(start, end)))
        
        # 准备任务参数（需要将 hist_arrays 转换为可序列化的字典）
        tasks = [(chunk, file_list, hist_arrays) for chunk in chunks]
        
        # 并行处理每个块
        sorted_indices = []
        with Pool(processes=workers) as pool:
            for chunk_result in tqdm.tqdm(
                pool.imap(_sort_hist_chunk, tasks),
                total=len(chunks),
                desc="Sorting chunks",
                unit="chunk"
            ):
                sorted_indices.extend(chunk_result)
        
        print(f"\n✓ 分块排序完成，共 {len(sorted_indices)} 张图片")
        
        # 构建结果
        sorted_results = [(file_list[idx], all_results[idx][1]) for idx in sorted_indices]
        
        # 构建新特征数据
        new_features = {}
        for filename, hist_data in newly_calculated:
            new_features[filename] = {'histogram': hist_data}
        
        return sorted_results, new_features
    
    def _sort_by_blur(self, workers: int = None, use_motion_blur: bool = False) -> List[Tuple[str, float]]:
        """
        按模糊度排序
        优先从元数据中读取，缺失的才重新计算
        
        Args:
            workers: 工作进程数
            use_motion_blur: 是否使用运动模糊检测
            
        Returns:
            排序后的列表（从高到低）
        """
        if workers is None:
            workers = cpu_count()
        
        print(S('SORTER_CALCULATING_BLUR', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的 blur/sharpness 值
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            # 分别读取需要的字段
            blur_value = self.get_field(filename, 'sharpness', 
                          self.get_field(filename, 'blur'))
            landmarks = self.get_field(filename, 'landmarks')
            
            if blur_value is not None:
                cached_results.append((filename, float(blur_value)))
            else:
                missing_files.append((img_path, landmarks))
        
        print(f"✓ From metadata: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        
        # 第二步：计算缺失的模糊度（每10000个保存一次）
        newly_calculated = []
        save_interval = 10000
        last_save_count = 0
        
        if missing_files:
            tasks = [(str(img_path), landmarks, use_motion_blur) for img_path, landmarks in missing_files]
            
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(
                    pool.imap_unordered(_calculate_blur_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing blur",
                    unit="img"
                )
                
                for result in pbar:
                    if result[1] > 0:
                        newly_calculated.append(result)
                        
                        # 每计算save_interval个就保存一次
                        if len(newly_calculated) - last_save_count >= save_interval:
                            batch_features = {}
                            for filename, blur_value in newly_calculated[last_save_count:]:
                                batch_features[filename] = {'sharpness': blur_value}
                            self._save_new_features_to_hdf5(batch_features)
                            last_save_count = len(newly_calculated)
                            pbar.set_description(f"Calculating missing blur [{last_save_count} saved]")
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
            
            # 保存剩余的数据
            if len(newly_calculated) > last_save_count:
                batch_features = {}
                for filename, blur_value in newly_calculated[last_save_count:]:
                    batch_features[filename] = {'sharpness': blur_value}
                self._save_new_features_to_hdf5(batch_features)
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_BY_BLUR'))
        
        # 第四步：按模糊度从高到低排序
        sorted_results = sorted(all_results, key=operator.itemgetter(1), reverse=True)
        
        # 构建新特征数据
        new_features = {}
        for filename, blur_value in newly_calculated:
            new_features[filename] = {'sharpness': blur_value}
        
        return sorted_results, new_features
    
    def _sort_by_face_pose(self, workers: int = None, pose_type: str = 'yaw') -> List[Tuple[str, float]]:
        """
        按人脸姿态排序
        优先从元数据中读取，缺失的才重新计算
        
        Args:
            workers: 工作进程数
            pose_type: 姿态类型 ('pitch', 'yaw', 'roll')
            
        Returns:
            排序后的列表
        """
        if workers is None:
            workers = cpu_count()
        
        print(S('SORTER_CALCULATING_POSE', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的 pose 值
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            pose_data = self.get_field(filename, 'pose')
            landmarks = self.get_field(filename, 'landmarks')
            
            if pose_data and pose_type in pose_data:
                cached_results.append((filename, float(pose_data[pose_type])))
            else:
                missing_files.append((img_path, landmarks))
        
        print(f"✓ From metadata: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        
        # 第二步：计算缺失的姿态（每10000个保存一次）
        newly_calculated = []
        save_interval = 10000
        last_save_count = 0
        
        if missing_files:
            tasks = [(str(img_path), landmarks) for img_path, landmarks in missing_files]
            
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(
                    pool.imap_unordered(_calculate_face_pose_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing pose",
                    unit="img"
                )
                
                for result in pbar:
                    if result[1] is not None:
                        newly_calculated.append((result[0], result[1].get(pose_type, 0.0)))
                        
                        # 每计算save_interval个就保存一次
                        if len(newly_calculated) - last_save_count >= save_interval:
                            batch_features = {}
                            for filename, pose_value in newly_calculated[last_save_count:]:
                                batch_features[filename] = {'pose': {pose_type: pose_value}}
                            self._save_new_features_to_hdf5(batch_features)
                            last_save_count = len(newly_calculated)
                            pbar.set_description(f"Calculating missing pose [{last_save_count} saved]")
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
            
            # 保存剩余的数据
            if len(newly_calculated) > last_save_count:
                batch_features = {}
                for filename, pose_value in newly_calculated[last_save_count:]:
                    batch_features[filename] = {'pose': {pose_type: pose_value}}
                self._save_new_features_to_hdf5(batch_features)
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_BY_POSE', pose_type))
        
        # 第四步：按姿态值排序
        sorted_results = sorted(all_results, key=operator.itemgetter(1))
        
        # 构建新特征数据
        new_features = {}
        for filename, pose_value in newly_calculated:
            # 需要重新获取完整的 pose 数据
            # 这里简化处理，只保存当前姿态类型
            new_features[filename] = {'pose': {pose_type: pose_value}}
        
        return sorted_results, new_features
    
    def _sort_by_resolution(self, workers: int = None) -> List[Tuple[str, int]]:
        """
        按分辨率排序
        优先从元数据中读取，缺失的才重新计算
        
        Args:
            workers: 工作进程数
            
        Returns:
            排序后的列表（从低到高）
        """
        if workers is None:
            workers = cpu_count()
        
        print(S('SORTER_CALCULATING_RESOLUTION', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的 resolution 值
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            resolution = self.get_field(filename, 'resolution')
            
            if resolution is not None:
                cached_results.append((filename, int(resolution)))
            else:
                missing_files.append(img_path)
        
        print(f"✓ From metadata: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        
        # 第二步：计算缺失的分辨率（每10000个保存一次）
        newly_calculated = []
        save_interval = 10000
        last_save_count = 0
        
        if missing_files:
            tasks = [(str(img_path),) for img_path in missing_files]
            
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(
                    pool.imap_unordered(_calculate_resolution_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing resolution",
                    unit="img"
                )
                
                for result in pbar:
                    if result[1] > 0:
                        newly_calculated.append(result)
                        
                        # 每计算save_interval个就保存一次
                        if len(newly_calculated) - last_save_count >= save_interval:
                            batch_features = {}
                            for filename, resolution in newly_calculated[last_save_count:]:
                                batch_features[filename] = {'resolution': resolution}
                            self._save_new_features_to_hdf5(batch_features)
                            last_save_count = len(newly_calculated)
                            pbar.set_description(f"Calculating missing resolution [{last_save_count} saved]")
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
            
            # 保存剩余的数据
            if len(newly_calculated) > last_save_count:
                batch_features = {}
                for filename, resolution in newly_calculated[last_save_count:]:
                    batch_features[filename] = {'resolution': resolution}
                self._save_new_features_to_hdf5(batch_features)
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_BY_RESOLUTION'))
        
        # 第四步：按分辨率从低到高排序
        sorted_results = sorted(all_results, key=operator.itemgetter(1))
        
        # 构建新特征数据
        new_features = {}
        for filename, resolution in newly_calculated:
            new_features[filename] = {'resolution': resolution}
        
        return sorted_results, new_features
    
    def _sort_by_color(self, workers: int = None) -> List[Tuple[str, float]]:
        """
        按颜色排序（平均色相）
        优先从元数据中读取，缺失的才重新计算
        
        Args:
            workers: 工作进程数
            
        Returns:
            排序后的列表
        """
        if workers is None:
            workers = cpu_count()
        
        print(S('SORTER_CALCULATING_COLOR', len(self.image_files)))
        
        # 第一步：从元数据中收集已有的 histogram 值（用于计算颜色）
        cached_results = []
        missing_files = []
        
        for img_path in self.image_files:
            filename = img_path.name
            # 使用父类的 get_field() 方法，只读取 histogram 字段
            hist_data = self.get_field(filename, 'histogram')
            
            if hist_data is not None:
                # 计算平均亮度作为颜色指标
                avg_brightness = (
                    sum(hist_data['r']) + 
                    sum(hist_data['g']) + 
                    sum(hist_data['b'])
                ) / 3
                cached_results.append((filename, avg_brightness))
            else:
                missing_files.append(img_path)
        
        print(f"✓ From metadata: {len(cached_results)}, Need to calculate: {len(missing_files)}")
        
        # 第二步：计算缺失的颜色值
        newly_calculated = []
        if missing_files:
            tasks = [(str(img_path),) for img_path in missing_files]
            
            with Pool(processes=workers) as pool:
                for result in tqdm.tqdm(
                    pool.imap_unordered(_calculate_histogram_worker, tasks),
                    total=len(tasks),
                    desc="Calculating missing color",
                    unit="img"
                ):
                    if result[1] is not None:
                        hist_data = result[1]
                        avg_brightness = (
                            sum(hist_data['r']) + 
                            sum(hist_data['g']) + 
                            sum(hist_data['b'])
                        ) / 3
                        newly_calculated.append((result[0], avg_brightness))
            
            print(f"✓ Newly calculated: {len(newly_calculated)}")
        
        # 第三步：合并结果
        all_results = cached_results + newly_calculated
        
        if not all_results:
            return [], {}
        
        print(S('SORTER_SORTING_BY_COLOR'))
        
        # 第四步：按颜色排序
        sorted_results = sorted(all_results, key=operator.itemgetter(1))
        
        # color 方法不保存新特征，因为它使用的是 histogram 数据
        return sorted_results, {}
    
    def _sort_by_name(self) -> List[Tuple[str, str]]:
        """
        按文件名排序
        
        Returns:
            排序后的列表
        """
        print(S('SORTER_SORTING_BY_NAME'))
        
        results = [(img_path.name, img_path.name) for img_path in self.image_files]
        sorted_results = sorted(results, key=operator.itemgetter(0))
        
        return sorted_results, {}
    
    def rename_sorted_files(self, sorted_list: List[Tuple[str, any]], prefix: str = "sorted", new_features: Dict[str, Dict] = None):
        """
        重命名排序后的文件，并更新元数据库
        使用 序号+16位随机码 作为最终名称，彻底避免冲突
        
        Args:
            sorted_list: 排序后的列表 [(filename, value), ...]
            prefix: 文件名前缀
            new_features: 新计算的特征数据 {filename: {feature_key: feature_value}}
        """
        import random
        import string
        
        print(S('SORTER_RENAMING_FILES', len(sorted_list)))
        
        # 生成16位随机字符串的函数
        def generate_random_suffix(length=16):
            chars = string.ascii_lowercase + string.digits
            return ''.join(random.choices(chars, k=length))
        
        # 第一阶段：收集所有需要重命名的文件
        rename_plan = []  # [(old_path, final_name, index), ...]
        
        for i, (filename, _) in enumerate(sorted_list):
            old_path = self.faceset_path / filename
            
            if not old_path.exists():
                print(f"\n⚠ 跳过 {filename}: 文件不存在")
                continue
            
            # 直接生成最终名称：prefix_序号_16位随机码.扩展名
            random_suffix = generate_random_suffix()
            final_name = f"{prefix}_{i:05d}_{random_suffix}{old_path.suffix}"
            
            rename_plan.append((old_path, final_name, i))
        
        print(f"  计划重命名 {len(rename_plan)} 个文件")
        print(f"  命名格式: {prefix}_序号_16位随机码.jpg (例如: {prefix}_00000_a3f7b9c2d4e5f6g8.jpg)")
        
        # 第二阶段：执行重命名 + 增量保存元数据
        success_count = 0
        failed_count = 0
        failed_files = []
        saved_metadata_count = 0
        
        # 创建备份
        metadata_file = self.faceset_path / "metadata.h5"
        backup_file = self.faceset_path / "metadata_backup.h5"
        
        if metadata_file.exists():
            try:
                import shutil
                shutil.copy2(metadata_file, backup_file)
                print(f"  ✓ 已创建元数据备份: {backup_file.name}")
            except Exception as e:
                print(f"  ⚠ 无法创建备份: {e}")
        
        pbar = tqdm.tqdm(total=len(rename_plan), desc="重命名文件", unit="file", ascii=True)
        
        # 增量元数据字典
        incremental_metadata = {}
        save_interval = 10000
        
        for old_path, final_name, i in rename_plan:
            final_path = self.faceset_path / final_name
            
            try:
                # 直接重命名为最终名称（带随机后缀，不会冲突）
                old_path.rename(final_path)
                success_count += 1
                
                # 构建元数据
                original_filename = old_path.name
                meta = self.get_metadata(original_filename).copy() if self.has_metadata(original_filename) else {}
                
                # 添加新特征
                if new_features and original_filename in new_features:
                    meta.update(new_features[original_filename])
                
                incremental_metadata[final_name] = meta
                saved_metadata_count += 1
                
                # 定期保存元数据
                if saved_metadata_count % save_interval == 0 and incremental_metadata:
                    try:
                        import h5py
                        
                        if self._h5_accessor is not None:
                            self._h5_accessor.close()
                        
                        with h5py.File(metadata_file, 'a') as f:
                            for filename, meta in incremental_metadata.items():
                                safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                                
                                # 如果组已存在，先删除（避免重复）
                                if safe_name in f:
                                    del f[safe_name]
                                
                                grp = f.create_group(safe_name)
                                grp.attrs['__original_filename__'] = filename
                                
                                for key, value in meta.items():
                                    if isinstance(value, (list, np.ndarray)):
                                        grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                                    elif isinstance(value, (int, float, str)):
                                        grp.attrs[key] = value
                                    else:
                                        grp.attrs[key] = str(value)
                        
                        incremental_metadata.clear()
                        pbar.set_description(f"重命名文件 [已保存: {saved_metadata_count}]")
                        
                    except Exception as e:
                        print(f"\n⚠ 元数据保存失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\n⚠ 重命名失败 {old_path.name}: {e}")
                failed_count += 1
                failed_files.append(old_path.name)
                pbar.update(1)
        
        pbar.close()
        
        print(f"\n✓ 成功重命名: {success_count}, 失败: {failed_count}")
        
        if failed_files:
            print(f"  失败的文件: {', '.join(failed_files[:5])}{'...' if len(failed_files) > 5 else ''}")
        
        # 第三阶段：保存剩余的元数据
        if incremental_metadata:
            print(f"  正在保存剩余的 {len(incremental_metadata)} 个元数据条目...")
            try:
                import h5py
                
                if self._h5_accessor is not None:
                    self._h5_accessor.close()
                
                with h5py.File(metadata_file, 'a') as f:
                    for filename, meta in incremental_metadata.items():
                        safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                        
                        if safe_name in f:
                            del f[safe_name]
                        
                        grp = f.create_group(safe_name)
                        grp.attrs['__original_filename__'] = filename
                        
                        for key, value in meta.items():
                            if isinstance(value, (list, np.ndarray)):
                                grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                            elif isinstance(value, (int, float, str)):
                                grp.attrs[key] = value
                            else:
                                grp.attrs[key] = str(value)
                
                print(f"  ✓ 元数据保存完成")
            except Exception as e:
                print(f"\n⚠ 最终元数据保存失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 清理旧文件名的元数据（防止HDF5文件无限增长）
        print(f"  正在清理旧元数据条目...")
        try:
            import h5py
            
            if self._h5_accessor is not None:
                self._h5_accessor.close()
            
            # 收集所有需要保留的新文件名
            new_filenames = set(final_name for _, final_name, _ in rename_plan)
            
            with h5py.File(metadata_file, 'a') as f:
                # 找出所有需要删除的旧条目
                keys_to_delete = []
                for key in f.keys():
                    # 解码key（h5py返回的是bytes或str）
                    key_str = key if isinstance(key, str) else key.decode('utf-8')
                    
                    # 如果不是新文件名，且不是特殊属性，就删除
                    if key_str not in new_filenames and not key_str.startswith('__'):
                        keys_to_delete.append(key_str)
                
                # 删除旧条目
                for key in keys_to_delete:
                    del f[key]
                
                if keys_to_delete:
                    print(f"  ✓ 已清理 {len(keys_to_delete)} 个旧元数据条目")
                else:
                    print(f"  ✓ 无需清理")
                    
        except Exception as e:
            print(f"\n⚠ 清理旧元数据失败: {e}")
            # 不中断流程，继续执行
        
        # 第四阶段：删除备份（如果一切成功）
        if backup_file.exists() and failed_count == 0:
            try:
                backup_file.unlink()
                print(f"  ✓ 已删除备份文件（重命名全部成功）")
            except:
                pass
        elif backup_file.exists():
            print(f"  ⚠ 保留备份文件以备恢复: {backup_file.name}")
        
        print(S('SORTER_RENAME_COMPLETE'))


def main():
    """主函数 - 命令行入口"""
    parser = argparse.ArgumentParser(description='Faceset Sorter - 人脸集排序工具')
    parser.add_argument('--input', '-i', type=str, required=True, help='人脸集路径')
    parser.add_argument('--method', '-m', type=str, default='phash',
                       choices=SortMethod.ALL_METHODS,
                       help='排序方法')
    parser.add_argument('--rename', action='store_true', help='重命名排序后的文件')
    parser.add_argument('--prefix', type=str, default='sorted', help='重命名前缀')
    parser.add_argument('--workers', type=int, default=None, help='工作进程数')
    parser.add_argument('--pose-type', type=str, default='yaw',
                       choices=['pitch', 'yaw', 'roll'],
                       help='姿态类型（仅用于face_pose排序）')
    parser.add_argument('--motion-blur', action='store_true', help='使用运动模糊检测')
    parser.add_argument('--reset', action='store_true', help='重置排序：将 sorted_XXXXX_随机码.jpg 还原为 XXXXX.jpg')
    
    args = parser.parse_args()
    
    # 创建排序器
    faceset_path = Path(args.input)
    if not faceset_path.exists():
        print(f"Error: Path not found: {faceset_path}")
        sys.exit(1)
    
    sorter = FaceSorter(faceset_path)
    
    # 如果指定了 --reset，执行重置操作
    if args.reset:
        import re
        
        prefix = args.prefix
        pattern = re.compile(rf'^{re.escape(prefix)}_(\d{{5}})_[a-z0-9]{{16}}(.+)$')
        
        sorted_files = [
            f for f in faceset_path.iterdir()
            if f.is_file() and pattern.match(f.name)
        ]
        
        if not sorted_files:
            print(f"未发现 {prefix}_XXXXX_随机码 格式的文件")
            sys.exit(0)
        
        print(f"发现 {len(sorted_files)} 个已排序的文件，开始重置...")
        print(f"  将还原为: XXXXX.jpg 格式（按序号重命名）")
        
        success_count = 0
        failed_count = 0
        metadata_updates = {}  # 记录需要更新的元数据 {新文件名: 元数据}
        
        pbar = tqdm.tqdm(total=len(sorted_files), desc="重置文件", unit="file", ascii=True)
        
        for sorted_file in sorted_files:
            match = pattern.match(sorted_file.name)
            if match:
                index = match.group(1)
                ext = match.group(2)
                new_name = f"{index}{ext}"
                new_path = faceset_path / new_name
                
                try:
                    # 如果目标文件已存在，先删除
                    if new_path.exists():
                        new_path.unlink()
                    
                    # 从HDF5读取旧文件名的元数据
                    old_meta = sorter.get_metadata(sorted_file.name)
                    if old_meta:
                        metadata_updates[new_name] = old_meta
                    
                    sorted_file.rename(new_path)
                    success_count += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"\n⚠ 重置失败 {sorted_file.name}: {e}")
                    failed_count += 1
                    pbar.update(1)
            else:
                pbar.update(1)
        
        pbar.close()
        
        # 更新元数据库
        if metadata_updates:
            print(f"\n  正在更新元数据库 ({len(metadata_updates)} 条)...")
            try:
                import h5py
                metadata_file = faceset_path / "metadata.h5"
                
                if sorter._h5_accessor is not None:
                    sorter._h5_accessor.close()
                
                with h5py.File(metadata_file, 'a') as f:
                    # 第1步：删除所有 sorted_ 开头的旧条目
                    keys_to_delete = []
                    for key in f.keys():
                        key_str = key if isinstance(key, str) else key.decode('utf-8')
                        if key_str.startswith('sorted_'):
                            keys_to_delete.append(key_str)
                    
                    for key in keys_to_delete:
                        del f[key]
                    
                    if keys_to_delete:
                        print(f"    已删除 {len(keys_to_delete)} 个旧条目")
                    
                    # 第2步：创建新条目
                    for new_name, meta in metadata_updates.items():
                        safe_name = new_name.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                        
                        grp = f.create_group(safe_name)
                        grp.attrs['__original_filename__'] = new_name
                        
                        for key, value in meta.items():
                            if isinstance(value, (list, np.ndarray)):
                                grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                            elif isinstance(value, (int, float, str)):
                                grp.attrs[key] = value
                            else:
                                grp.attrs[key] = str(value)
                
                print(f"  ✓ 元数据库已更新")
            except Exception as e:
                print(f"\n⚠ 元数据更新失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✓ 重置完成: 成功 {success_count}, 失败 {failed_count}")
        print(f"  文件已按序号重命名为: 00000.jpg, 00001.jpg, ...")
        print("\nDone!")
        sys.exit(0)
    
    # 执行排序
    kwargs = {'workers': args.workers}
    if args.method == SortMethod.FACE_POSE:
        kwargs['pose_type'] = args.pose_type
    elif args.method == SortMethod.BLUR:
        kwargs['use_motion_blur'] = args.motion_blur
    
    sorted_list = sorter.sort_by_method(args.method, **kwargs)
    
    # 处理返回的元组 (sorted_list, new_features)
    if isinstance(sorted_list, tuple):
        sorted_list, new_features = sorted_list
    else:
        new_features = {}
    
    if not sorted_list:
        print("No images to sort")
        sys.exit(1)
    
    print(f"\nSorted {len(sorted_list)} images by {args.method}")
    
    # 如果需要重命名（始终传递 new_features）
    if args.rename:
        sorter.rename_sorted_files(sorted_list, args.prefix, new_features)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
