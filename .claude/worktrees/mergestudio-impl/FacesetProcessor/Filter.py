"""
DeepFaceLab Torch - FacesetProcessor Filter Module
人脸集过滤器模块：基于质量过滤和人脸ID分组
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json
import shutil
import argparse
from typing import Dict, List, Optional, Tuple
import tqdm
import multiprocessing
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

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
from facelib.LandmarksProcessor import get_image_hull_mask

# 导入基类
from FacesetBaseProcessor import FacesetBaseProcessor

# InsightFace 相关 (可选,优先使用 ONNX)
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class ArcFaceONNXExtractor:
    """
    纯 ONNX 实现的 ArcFace embedding 提取器
    不需要安装 insightface 库，直接使用 .onnx 模型文件
    """
    
    def __init__(self, model_dir: Path, model_name: str = None):
        """
        初始化 ArcFace 提取器
        
        Args:
            model_dir: 模型目录路径
            model_name: 指定模型名称 (可选)，支持: w600k_mbf, mobilefacenet, w600k_r50
        """
        self.model_dir = Path(model_dir)
        self.model_name_used = None  # 记录实际使用的模型名称
        
        # 加载识别模型 (按优先级或指定名称)
        if model_name:
            # 使用指定的模型
            if not model_name.endswith('.onnx'):
                model_name += '.onnx'
            rec_model_path = self.model_dir / model_name
            if not rec_model_path.exists():
                raise FileNotFoundError(f"Specified model '{model_name}' not found in {self.model_dir}")
            self.model_name_used = model_name
        else:
            # 自动选择: w600k_mbf > w600k_r50
            rec_model_path = None
            for candidate_name in ["w600k_mbf.onnx", "w600k_r50.onnx"]:
                candidate = self.model_dir / candidate_name
                if candidate.exists():
                    rec_model_path = candidate
                    self.model_name_used = candidate_name
                    break
            
            if rec_model_path is None:
                raise FileNotFoundError(f"No recognition model found in {self.model_dir}")
        
        self.rec_session = onnxruntime.InferenceSession(
            str(rec_model_path), 
            providers=['CPUExecutionProvider']
        )
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        
        # 检测模型 (可选,已对齐的人脸不需要)
        # 注释掉检测模型加载,因为输入已经是裁剪好的人脸
    
    def detect_faces(self, img: np.ndarray) -> List[np.ndarray]:
        """
        检测人脸并返回边界框
        注意: 对于已对齐的人脸数据集，此方法不再使用
        
        Args:
            img: BGR 图像 (H, W, 3)
            
        Returns:
            人脸边界框列表 [(x1, y1, x2, y2), ...]
        """
        # 已对齐的人脸，返回整张图
        h, w = img.shape[:2]
        return [[0, 0, w, h]]
    
    def extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        从人脸图像提取 embedding
        
        Args:
            face_img: 人脸图像 (BGR格式)
            
        Returns:
            embedding 向量 (512维)，失败返回 None
        """
        try:
            # 预处理：resize 到 112x112
            face_resized = cv2.resize(face_img, (112, 112))
            
            # BGR -> RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # 归一化
            face_normalized = face_rgb.astype(np.float32) / 255.0
            
            # 标准化 (ArcFace 要求)
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            face_normalized = (face_normalized - mean) / std
            
            # HWC -> CHW
            face_transposed = np.transpose(face_normalized, (2, 0, 1))
            
            # 添加 batch 维度
            face_batch = np.expand_dims(face_transposed, axis=0)
            
            # 推理
            embedding = self.rec_session.run(
                None, 
                {self.rec_input_name: face_batch}
            )[0]
            
            # 归一化 embedding
            embedding = embedding.flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            print(S('FILTER_EMBEDDING_ERROR', 'unknown', e))
            return None


def _calculate_sharpness_worker(args):
    """
    多进程工作函数：计算单张图片的清晰度
    
    Args:
        args: (image_path_str, landmarks, faceset_path_str)
        
    Returns:
        (filename, sharpness, error_msg)
    """
    image_path_str, landmarks, faceset_path_str = args
    
    try:
        # 读取图像
        image = cv2.imread(image_path_str)
        if image is None:
            return (Path(image_path_str).name, 0.0, "Failed to read image")
        
        h, w = image.shape[:2]
        
        # 获取人脸凸包mask
        hull_mask = get_image_hull_mask(image.shape, np.array(landmarks))
        
        # 应用mask提取人脸区域
        face_region = image * hull_mask
        
        # 转换为灰度图
        gray = cv2.cvtColor(face_region.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # 计算拉普拉斯方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return (Path(image_path_str).name, float(variance), None)
    except Exception as e:
        return (Path(image_path_str).name, 0.0, str(e))


def _load_metadata_h5(metadata_file: Path) -> Dict:
    """
    通用函数：加载 HDF5 格式的元数据
    
    Args:
        metadata_file: 元数据文件路径
        
    Returns:
        元数据字典
    """
    import h5py
    
    if not metadata_file.exists():
        print(S('ANALYZER_NO_METADATA'))
        return {}
    
    try:
        # 显示文件大小
        file_size = metadata_file.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"\nLoading metadata from: {metadata_file.name}")
        print(f"File size: {file_size_mb:.2f} MB")
        
        print("Reading HDF5...", end=" ", flush=True)
        
        metadata_dict = {}
        with h5py.File(metadata_file, 'r') as f:
            # 遍历所有组（每个组对应一个文件）
            for safe_name in f.keys():
                grp = f[safe_name]
                meta = {}
                
                # 读取 datasets（数组类型）
                for key in grp.keys():
                    dataset = grp[key]
                    if isinstance(dataset, h5py.Dataset):
                        data = dataset[:]
                        # 对于 landmarks 和 embedding，保持为 numpy 数组
                        # 其他数组转换为 list
                        if key in ['landmarks', 'embedding'] and isinstance(data, np.ndarray):
                            meta[key] = data
                        elif isinstance(data, np.ndarray):
                            if data.ndim == 0:
                                meta[key] = data.item()
                            else:
                                meta[key] = data.tolist()
                        else:
                            meta[key] = data
                
                # 读取 attributes（标量类型）
                for key, value in grp.attrs.items():
                    # 跳过内部使用的属性
                    if key == '__original_filename__':
                        continue
                    meta[key] = value
                
                # 使用存储的原始文件名
                original_filename = grp.attrs.get('__original_filename__', safe_name)
                metadata_dict[original_filename] = meta
        
        print(f"\r{' ' * 80}\r", end="", flush=True)
        print(S('FILTER_METADATA_LOADED', len(metadata_dict)))
        return metadata_dict
        
    except ImportError:
        print("\n✗ 错误: 需要安装 h5py 库来读取 HDF5 格式")
        print("请运行: pip install h5py\n")
        return {}
    except Exception as e:
        print(S('ANALYZER_METADATA_LOAD_ERROR', e))
        import traceback
        traceback.print_exc()
        return {}


class QualityFilter(FacesetBaseProcessor):
    """质量过滤器 - 基于拉普拉斯方差检测模糊
    
    继承自 FacesetBaseProcessor，自动获得流式 HDF5 访问能力
    """
    
    def __init__(self, faceset_path: Path, metadata_file: Optional[Path] = None):
        """
        初始化质量过滤器
        
        Args:
            faceset_path: 人脸数据集路径
            metadata_file: 元数据文件路径（可选）
        """
        # 调用父类初始化（会自动初始化 HDF5 访问器和扫描文件）
        super().__init__(faceset_path, metadata_file)
    
    def calculate_face_sharpness(self, image: np.ndarray, landmarks: List[List[float]]) -> float:
        """
        计算人脸区域的清晰度（拉普拉斯方差）
        
        Args:
            image: BGR格式图像
            landmarks: 68个特征点坐标
            
        Returns:
            拉普拉斯方差值
        """
        try:
            h, w = image.shape[:2]
            
            # 获取人脸凸包mask
            hull_mask = get_image_hull_mask(image.shape, np.array(landmarks))
            
            # 应用mask提取人脸区域
            face_region = image * hull_mask
            
            # 转换为灰度图
            gray = cv2.cvtColor(face_region.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            
            # 计算拉普拉斯方差
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            return float(variance)
        except Exception as e:
            print(S('FILTER_SHARPNESS_ERROR', e))
            return 0.0
    
    def filter_by_quality(self, threshold: float = 20.0, workers: int = None) -> Dict:
        """
        根据清晰度过滤图片，移动到 highquality 或 lowquality 文件夹
        
        Args:
            threshold: 拉普拉斯方差阈值，默认20
            workers: 工作进程数，默认为 CPU 核心数
            
        Returns:
            过滤统计结果
        """
        print(S('FILTER_START_QUALITY_FILTER', threshold))
        
        # 创建输出目录
        high_dir = self.faceset_path / "highquality"
        low_dir = self.faceset_path / "lowquality"
        high_dir.mkdir(exist_ok=True)
        low_dir.mkdir(exist_ok=True)
        
        stats = {
            'total': 0,
            'high_quality': 0,
            'low_quality': 0,
            'errors': 0,
            'high_files': [],
            'low_files': []
        }
        
        # 扫描所有图片
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([
            f for f in self.faceset_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ])
        
        print(S('FILTER_FOUND_IMAGES', len(image_files)))
        
        # ========== 第一阶段：计算并标记（多进程） ==========
        print(S('FILTER_PHASE1_CALCULATING'))
        
        # 准备任务列表
        tasks = []
        valid_files = []
        
        for img_file in image_files:
            filename = img_file.name
            # 使用父类的 get_field() 方法，只读取需要的字段
            landmarks = self.get_field(filename, 'landmarks')
            
            if landmarks is None:
                print(S('FILTER_NO_LANDMARKS', filename))
                stats['errors'] += 1
                continue
            
            tasks.append((str(img_file), landmarks, str(self.faceset_path)))
            valid_files.append(img_file)
        
        # 设置工作进程数
        if workers is None:
            workers = min(multiprocessing.cpu_count(), len(tasks))
        
        print(S('FILTER_USING_WORKERS', workers))
        
        # 多进程计算
        sharpness_results = {}
        if tasks:
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(total=len(tasks), desc=S('FILTER_PROGRESS_QUALITY'), unit="img", ascii=True)
                
                for filename, sharpness, error in pool.imap_unordered(_calculate_sharpness_worker, tasks):
                    if error:
                        print(S('FILTER_PROCESS_ERROR', filename, error))
                        stats['errors'] += 1
                    else:
                        sharpness_results[filename] = sharpness
                        stats['total'] += 1
                    pbar.update(1)
                
                pbar.close()
        
        # ========== 第二阶段：分类并构建移动列表 ==========
        print(S('FILTER_PHASE2_CLASSIFYING'))
        move_list = []  # [(src_path, dst_path), ...]
        
        for img_file in valid_files:
            filename = img_file.name
            if filename not in sharpness_results:
                continue
            
            sharpness = sharpness_results[filename]
            
            # 标记目标路径（不立即移动）
            if sharpness >= threshold:
                # 高质量
                dest = high_dir / img_file.name
                move_list.append((img_file, dest))
                stats['high_quality'] += 1
                stats['high_files'].append(filename)
            else:
                # 低质量
                dest = low_dir / img_file.name
                move_list.append((img_file, dest))
                stats['low_quality'] += 1
                stats['low_files'].append(filename)
        
        # ========== 第三阶段：批量移动 ==========
        if move_list:
            print(S('FILTER_PHASE3_MOVING', len(move_list)))
            pbar = tqdm.tqdm(total=len(move_list), desc=S('FILTER_PROGRESS_MOVING'), unit="file", ascii=True)
            
            for src, dst in move_list:
                try:
                    shutil.move(str(src), str(dst))
                except Exception as e:
                    print(S('FILTER_MOVE_ERROR', src.name, e))
                    stats['errors'] += 1
                finally:
                    pbar.update(1)
            
            pbar.close()
        
        # 打印统计
        print(S('FILTER_QUALITY_COMPLETE'))
        print(f"  {S('FILTER_STAT_TOTAL')}: {stats['total']}")
        print(f"  {S('FILTER_STAT_HIGH')}: {stats['high_quality']}")
        print(f"  {S('FILTER_STAT_LOW')}: {stats['low_quality']}")
        print(f"  {S('FILTER_STAT_ERRORS')}: {stats['errors']}")
        
        return stats


class FaceIDFilter(FacesetBaseProcessor):
    """人脸ID过滤器 - 基于ArcFace embedding分组
    
    继承自 FacesetBaseProcessor，自动获得流式 HDF5 访问能力
    """
    
    def __init__(self, faceset_path: Path, metadata_file: Optional[Path] = None, model_name: str = None, skip_model_init: bool = False):
        """
        初始化人脸ID过滤器
        
        Args:
            faceset_path: 人脸数据集路径
            metadata_file: 元数据文件路径（可选）
            model_name: 指定使用的模型名称 (可选)，支持: w600k_mbf, mobilefacenet, w600k_r50
            skip_model_init: 跳过模型初始化（用于 merge-only 模式）
        """
        # 调用父类初始化（会自动初始化 HDF5 访问器和扫描文件）
        super().__init__(faceset_path, metadata_file)
        
        # 如果跳过模型初始化（merge-only 模式），直接返回
        if skip_model_init:
            self.arcface_extractor = None
            self.use_insightface = False
            return
        
        # 检查 embedding 覆盖情况，决定是否需要模型
        needs_model = False
        metadata_count = self.metadata_count
        if metadata_count > 0:
            # 统计有 embedding 的文件数量（流式检查）
            has_embedding_count = 0
            for filename in self.metadata_filenames:
                emb = self.get_field(filename, 'embedding')
                if emb is not None:
                    has_embedding_count += 1
            
            missing_count = metadata_count - has_embedding_count
            
            if has_embedding_count == metadata_count:
                print(f"✓ 所有 {metadata_count} 张图片都已有 embedding 数据")
                print(f"  → 将直接使用缓存的 embedding，无需重新计算\n")
                needs_model = False
            elif has_embedding_count > 0:
                print(f"⚠ {has_embedding_count}/{metadata_count} 张图片有 embedding，{missing_count} 张缺失")
                print(f"  → 将对缺失的图片重新提取 embedding\n")
                needs_model = True
            else:
                print(f"⚠ 所有 {metadata_count} 张图片都缺少 embedding 数据")
                print(f"  → 将重新提取所有图片的 embedding\n")
                needs_model = True
        else:
            print(f"⚠ 没有元数据或元数据为空")
            print(f"  → 将提取所有图片的 embedding\n")
            needs_model = True
        
        # 如果不需要模型（所有数据都有 embedding），直接返回
        if not needs_model:
            self.arcface_extractor = None
            self.use_insightface = False
            return
        
        # 初始化 ArcFace ONNX 提取器
        # 模型在项目根目录的 modelhub/ 下
        project_root = Path(__file__).parent.parent
        model_dir = project_root / "modelhub"
        
        if model_dir.exists():
            try:
                # 如果未指定模型且有多个可用，询问用户
                if model_name is None:
                    available_models = []
                    for candidate_name in ["w600k_mbf.onnx", "w600k_r50.onnx"]:
                        if (model_dir / candidate_name).exists():
                            available_models.append(candidate_name.replace('.onnx', ''))
                    
                    if len(available_models) > 1:
                        print(f"\n{S('FILTER_AVAILABLE_MODELS')}: {', '.join(available_models)}")
                        print(S('FILTER_SELECT_MODEL'))
                        for i, name in enumerate(available_models, 1):
                            marker = f" {S('FILTER_RECOMMENDED')}" if i == 1 else ""
                            print(f"  {i}. {name}{marker}")
                        
                        try:
                            choice = input(f"\n{S('FILTER_ENTER_CHOICE').format(1, len(available_models), 1)}").strip()
                            if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                                model_name = available_models[int(choice) - 1]
                            elif choice == '':
                                model_name = available_models[0]  # 默认第一个
                            else:
                                print(S('FILTER_INVALID_CHOICE').format(available_models[0]))
                                model_name = available_models[0]
                        except (EOFError, KeyboardInterrupt):
                            print(f"\n{S('FILTER_USING_DEFAULT').format(available_models[0])}")
                            model_name = available_models[0]
                        print()
                
                self.arcface_extractor = ArcFaceONNXExtractor(model_dir, model_name=model_name)
                self.use_insightface = False
                # 显示使用的模型
                if self.arcface_extractor.model_name_used:
                    print(f"Using model: {self.arcface_extractor.model_name_used}\n")
            except Exception as e:
                print(S('FILTER_ARCFACE_ONNX_FAILED', e))
                self.arcface_extractor = None
                self.use_insightface = False
        else:
            self.arcface_extractor = None
            self.use_insightface = False
        
        # 如果 ONNX 失败，尝试使用 InsightFace
        if not self.arcface_extractor and INSIGHTFACE_AVAILABLE:
            print(S('FILTER_INIT_INSIGHTFACE'))
            try:
                self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=0, det_size=(640, 640))
                self.use_insightface = True
                print(S('FILTER_INSIGHTFACE_READY'))
            except Exception as e:
                print(S('FILTER_INSIGHTFACE_FAILED', e))
                self.use_insightface = False
        
        if not self.arcface_extractor and not self.use_insightface:
            # 不打印警告，因为可以从元数据中读取 embedding
            pass
    
    def extract_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """
        提取人脸 embedding 向量
        
        Args:
            image_path: 图像路径
            
        Returns:
            embedding 向量 (512维)，失败返回 None
        """
        try:
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                print(S('FILTER_IMAGE_READ_ERROR', image_path.name))
                return None
            
            # 优先使用 ArcFace ONNX
            if self.arcface_extractor:
                # 对于已对齐的人脸，直接提取 embedding
                emb = self.arcface_extractor.extract_embedding(img)
                if emb is None:
                    print(S('FILTER_ARCFACE_RETURNED_NONE', image_path.name))
                return emb
            
            # 其次使用 InsightFace
            elif self.use_insightface and hasattr(self, 'app'):
                faces = self.app.get(img)
                if len(faces) == 0:
                    print(S('FILTER_INSIGHTFACE_NO_FACE', image_path.name))
                    return None
                best_face = max(faces, key=lambda x: x.det_score)
                return best_face.embedding
            
            else:
                # 没有任何可用的提取器
                if not hasattr(self, '_extraction_warning_shown'):
                    print(S('FILTER_NO_EXTRACTION_MODEL'))
                    print(f"  - ArcFace ONNX: {'✓' if self.arcface_extractor else '✗'}")
                    print(f"  - InsightFace: {'✓' if self.use_insightface else '✗'}")
                    print(f"\n{S('FILTER_CHECK_MODEL_PATH')}")
                    print(S('FILTER_SUPPORTED_MODELS'))
                    print(S('FILTER_CURRENT_MODEL_PATH', Path(__file__).parent.parent / 'modelhub'))
                    print()
                    self._extraction_warning_shown = True
                return None
            
        except Exception as e:
            print(S('FILTER_EMBEDDING_ERROR', image_path.name, e))
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个 embedding 的余弦相似度
        
        Args:
            emb1: 第一个 embedding 向量
            emb2: 第二个 embedding 向量
            
        Returns:
            余弦相似度 (0-1)，越接近1表示越相似
        """
        try:
            # 归一化
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
            
            # 余弦相似度
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # 转换到 0-1 范围
            similarity = (similarity + 1.0) / 2.0
            
            return float(similarity)
        except Exception as e:
            print(S('FILTER_SIMILARITY_ERROR', e))
            return 0.0
    
    def calculate_landmark_similarity(self, landmarks1: List[List[float]], 
                                     landmarks2: List[List[float]]) -> float:
        """
        计算两组特征点的相似度（归一化欧氏距离）
        
        Args:
            landmarks1: 第一组68个特征点
            landmarks2: 第二组68个特征点
            
        Returns:
            相似度分数 (0-1)，越接近1表示越相似
        """
        try:
            pts1 = np.array(landmarks1, dtype=np.float32)
            pts2 = np.array(landmarks2, dtype=np.float32)
            
            # 计算欧氏距离
            distances = np.sqrt(np.sum((pts1 - pts2) ** 2, axis=1))
            
            # 归一化：除以人脸尺寸（使用两眼之间的距离作为参考）
            eye_dist = np.sqrt(np.sum((pts1[36] - pts1[45]) ** 2))
            if eye_dist == 0:
                eye_dist = 1.0
            
            normalized_dist = np.mean(distances) / eye_dist
            
            # 转换为相似度（距离越小，相似度越高）
            similarity = 1.0 / (1.0 + normalized_dist)
            
            return float(similarity)
        except Exception as e:
            print(S('FILTER_SIMILARITY_ERROR', e))
            return 0.0
    
    def filter_by_face_id(self, eps: float = 0.3, min_samples: int = 2, merge_back: bool = False, output_dir: str = None) -> Dict:
        """
        根据人脸ID分组，将相似的人脸移动到同一文件夹
        使用 DBSCAN 聚类算法进行准确的人脸识别分组
        
        Args:
            eps: DBSCAN 的邻域半径参数（余弦距离空间）
                 默认 0.3 表示余弦相似度 >= 0.7 视为相似
                 值越小，分组越严格；值越大，分组越宽松
            min_samples: DBSCAN 的最小样本数，默认2
            merge_back: 是否将分组后的文件合并回一个文件夹
            output_dir: 合并后的输出目录名称（默认: aligned）
            
        Returns:
            分组统计结果
        """
        print(S('FILTER_START_FACEID_FILTER', f'eps={eps}'))
        print(S('FILTER_DBSCLAN_INFO', eps, min_samples))
        
        # 扫描所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([
            f for f in self.faceset_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ])
        
        if not image_files:
            print(S('FILTER_NO_IMAGES_FOUND'))
            return {}
        
        print(S('FILTER_FOUND_IMAGES', len(image_files)))
        
        # ========== 第一阶段：提取 embedding ==========
        print(S('FILTER_PHASE1_EXTRACTING_EMBEDDINGS'))
        
        embeddings_dict = {}  # {filename: embedding}
        filename_list = []  # 保持顺序
        failed_files = []
        
        pbar = tqdm.tqdm(total=len(image_files), desc=S('FILTER_PROGRESS_EXTRACTING'), unit="img", ascii=True)
        
        saved_count = 0  # 统计本次新保存的 embedding 数量
        
        for img_file in image_files:
            filename = img_file.name
            try:
                # 先尝试从元数据中获取 embedding（流式读取）
                emb_list = self.get_field(filename, 'embedding')
                if emb_list is not None:
                    # 使用已计算的 embedding
                    emb = np.array(emb_list, dtype=np.float32)
                    embeddings_dict[filename] = emb
                    filename_list.append(filename)
                else:
                    # 提取新的 embedding
                    emb = self.extract_embedding(img_file)
                    if emb is not None:
                        embeddings_dict[filename] = emb
                        filename_list.append(filename)
                        saved_count += 1
                    else:
                        failed_files.append(filename)
            except Exception as e:
                print(S('FILTER_PROCESS_ERROR', img_file.name, e))
                failed_files.append(filename)
            finally:
                pbar.update(1)
        
        pbar.close()
        
        # 保存剩余的 embedding 到元数据
        if saved_count > 0:
            try:
                with open(self.metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                print(f"\n✓ 已保存 {saved_count} 个新提取的 embedding 到元数据")
            except Exception as e:
                print(S('FILTER_SAVE_METADATA_WARNING', e))
        
        # 统计缓存使用情况
        cached_count = sum(1 for img_file in image_files if self.get_field(img_file.name, 'embedding') is not None)
        if cached_count > 0:
            print(S('FILTER_USING_CACHED_EMBEDDINGS'))
        
        print(S('FILTER_EMBEDDINGS_EXTRACTED', len(embeddings_dict), len(failed_files)))
        
        if not embeddings_dict:
            print(S('FILTER_NO_EMBEDDINGS'))
            return {}
        
        # ========== 第二阶段：DBSCAN 聚类 ==========
        print(S('FILTER_PHASE2_GROUPING'))
        print(S('FILTER_COMPUTING_SIMILARITY'))
        
        # 构建 embedding 矩阵
        embedding_matrix = np.array([embeddings_dict[f] for f in filename_list])
        print(f"  Embedding 矩阵形状: {embedding_matrix.shape}")
        print(f"  预计内存需求: {embedding_matrix.nbytes / (1024**2):.2f} MB")
        
        # 检查是否可以使用全量相似度矩阵（限制在 10GB 以内）
        n_samples = len(filename_list)
        estimated_memory_gb = (n_samples * n_samples * 4) / (1024**3)  # float32 = 4 bytes
        
        if estimated_memory_gb > 10:
            print(f"\n⚠ 警告: 全量相似度矩阵需要 {estimated_memory_gb:.2f} GB 内存，超过限制")
            print(f"  → 改用分批处理策略...\n")
            
            # 使用分批处理的 DBSCAN
            from sklearn.neighbors import NearestNeighbors
            
            print("  使用 NearestNeighbors 进行高效相似度搜索...")
            # 归一化 embedding 以便使用余弦距离
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # 避免除零
            embedding_normalized = embedding_matrix / norms
            
            # 转换 eps 从余弦距离到欧氏距离
            # 对于归一化向量：欧氏距离² = 2 × 余弦距离
            # 所以：欧氏距离 = √(2 × 余弦距离)
            euclidean_radius = np.sqrt(2 * eps)
            print(f"  余弦距离阈值: {eps}, 转换为欧氏距离: {euclidean_radius:.4f}")
            
            # 使用 ball_tree 算法进行高效搜索
            # 注意：对于归一化向量，欧氏距离等价于余弦距离的转换形式
            nn = NearestNeighbors(
                metric='euclidean',  # 使用欧氏距离（对归一化向量等价于余弦距离）
                algorithm='auto',  # 让 sklearn 自动选择最优算法（通常比 ball_tree 更快）
                radius=euclidean_radius  # 使用转换后的欧氏距离阈值
            )
            nn.fit(embedding_normalized)
            print(f"  ✓ NearestNeighbors 模型训练完成")
            
            # 查找每个点的邻居（分批处理以显示进度）
            print(f"  正在查找邻居（欧氏半径={euclidean_radius:.4f}）...")
            print(f"  提示: 首次调用可能需要较长时间构建索引，请耐心等待...")
            
            # 分批处理，每批 10000 个点（增大批次减少调用次数）
            batch_size = 10000
            neighbors_list = []
            n_batches = (n_samples + batch_size - 1) // batch_size
            
            pbar_neighbor = tqdm.tqdm(total=n_samples, desc="  查找邻居", unit="img", ascii=True)
            
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_embeddings = embedding_normalized[batch_start:batch_end]
                
                # 查找当前批次的邻居
                batch_neighbors = nn.radius_neighbors(batch_embeddings, radius=euclidean_radius, return_distance=False)
                neighbors_list.extend(batch_neighbors)
                
                pbar_neighbor.update(batch_end - batch_start)
            
            pbar_neighbor.close()
            
            # 转换为数组格式（与原始 API 保持一致）
            neighbors = np.array(neighbors_list, dtype=object)
            
            print(f"  ✓ 邻居查找完成")
            
            # 基于邻居关系进行连通分量聚类
            print("  正在进行连通分量聚类...")
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
            
            # 构建稀疏邻接矩阵并应用 min_samples 约束
            rows = []
            cols = []
            valid_points = set()  # 满足 min_samples 的核心点
            
            # 第一遍：找出核心点（邻居数 >= min_samples）
            print(f"  [1/3] 识别核心点（min_samples={min_samples}）...")
            neighbor_counts = {}
            pbar_core = tqdm.tqdm(total=n_samples, desc="  识别核心点", unit="img", ascii=True)
            for i, nbrs in enumerate(neighbors):
                neighbor_counts[i] = len(nbrs)
                if len(nbrs) >= min_samples:
                    valid_points.add(i)
                pbar_core.update(1)
            pbar_core.close()
            print(f"  ✓ 找到 {len(valid_points)} 个核心点")
            
            # 第二遍：只添加核心点的边（模拟 DBSCAN 的行为）
            print(f"  [2/3] 构建邻接矩阵...")
            pbar_adj = tqdm.tqdm(total=len(valid_points), desc="  构建邻接矩阵", unit="point", ascii=True)
            for i in valid_points:
                for j in neighbors[i]:
                    if i != j:  # 排除自己
                        rows.append(i)
                        cols.append(j)
                pbar_adj.update(1)
            pbar_adj.close()
            print(f"  ✓ 邻接矩阵构建完成（{len(rows)} 条边）")
            
            adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_samples, n_samples))
            
            # 计算连通分量（只在核心点之间）
            print(f"  [3/3] 计算连通分量...")
            n_components, labels_temp = connected_components(adjacency, directed=False)
            print(f"  ✓ 连通分量计算完成")
            
            # 初始化所有标签为 -1（噪声）
            labels = np.full(n_samples, -1, dtype=int)
            
            # 为核心点和它们的邻居分配标签
            print(f"  正在分配聚类标签...")
            pbar_label = tqdm.tqdm(total=n_components, desc="  分配标签", unit="cluster", ascii=True)
            for component_id in range(n_components):
                # 找到该连通分量中的所有点
                component_points = np.where(labels_temp == component_id)[0]
                
                # 检查这些点是否都是有效点（至少有一个核心点）
                has_core = any(p in valid_points for p in component_points)
                
                if has_core:
                    # 为该连通分量分配标签
                    for point in component_points:
                        labels[point] = component_id
                
                pbar_label.update(1)
            pbar_label.close()
            
            print(f"\n✓ 聚类完成")
            print(f"  发现 {n_components} 个连通分量")
        else:
            # 原始的全量矩阵方法（小数据集）
            print(f"  使用全量相似度矩阵（预计 {estimated_memory_gb:.2f} GB）...")
            
            # 计算余弦相似度矩阵
            similarity_matrix = cosine_similarity(embedding_matrix)
            
            # 转换为距离矩阵（DBSCAN 需要距离）
            distance_matrix = 1.0 - similarity_matrix
            distance_matrix = np.clip(distance_matrix, 0.0, None)
            
            print(S('FILTER_RUNNING_DBSCLAN'))
            
            # 执行 DBSCAN 聚类
            db = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='precomputed'
            )
            
            labels = db.fit_predict(distance_matrix)
        
        # 统计聚类结果
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)  # 排除噪声点
        n_noise = list(labels).count(-1)
        
        print(S('FILTER_CLUSTERING_RESULTS'))
        print(S('FILTER_TOTAL_CLUSTERS', n_clusters))
        print(S('FILTER_NOISE_POINTS', n_noise))
        
        # 构建分组字典
        groups = {}
        noise_files = []
        
        for i, label in enumerate(labels):
            filename = filename_list[i]
            if label == -1:
                # 噪声点（无法归类）
                noise_files.append(filename)
            else:
                if label not in groups:
                    groups[label] = []
                groups[label].append(filename)
        
        # 重新编号组（按成员数量降序排列，最多的组获得最小编号）
        print(f"  正在按组成员数量排序...")
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        renumbered_groups = {}
        for new_id, (old_id, filenames) in enumerate(sorted_groups):
            renumbered_groups[new_id] = filenames
        
        groups = renumbered_groups
        print(f"  ✓ 排序完成：最大的组有 {len(groups[0])} 张图片")
        
        # ========== 第三阶段：最终确认保存（确保所有 embedding 已保存）==========
        
        # ========== 第四阶段：批量移动 ==========
        print(S('FILTER_MOVING_FILES', len(groups)))
        
        stats = {
            'total_groups': len(groups),
            'total_files': sum(len(g) for g in groups.values()),
            'failed_files': len(failed_files),
            'noise_files': len(noise_files),
            'groups': {}
        }
        
        # 构建移动列表
        move_list = []  # [(src_path, dst_path), ...]
        
        for group_id, filenames in groups.items():
            # 创建组文件夹
            group_dir = self.faceset_path / f"faceid_group_{group_id:03d}"
            group_dir.mkdir(exist_ok=True)
            
            # 添加到移动列表
            for filename in filenames:
                src = self.faceset_path / filename
                dst = group_dir / filename
                move_list.append((src, dst))
            
            stats['groups'][f"group_{group_id:03d}"] = len(filenames)
        
        # 处理噪声点（无法归类的图片）
        if noise_files:
            noise_dir = self.faceset_path / "faceid_noise"
            noise_dir.mkdir(exist_ok=True)
            
            for filename in noise_files:
                src = self.faceset_path / filename
                dst = noise_dir / filename
                move_list.append((src, dst))
            
            print(S('FILTER_MOVING_NOISE', len(noise_files)))
        
        # 执行批量移动
        if move_list:
            pbar = tqdm.tqdm(total=len(move_list), desc=S('FILTER_PROGRESS_MOVING'), unit="file", ascii=True)
            
            for src, dst in move_list:
                try:
                    shutil.move(str(src), str(dst))
                except Exception as e:
                    print(S('FILTER_MOVE_ERROR', src.name, e))
                    stats['errors'] = stats.get('errors', 0) + 1
                finally:
                    pbar.update(1)
            
            pbar.close()
        
        # ========== 第四阶段：可选的合并操作 ==========
        if merge_back:
            stats = self._merge_groups_to_aligned(stats, output_dir)
        
        print(S('FILTER_FACEID_COMPLETE'))
        print(f"  {S('FILTER_STAT_GROUPS')}: {stats['total_groups']}")
        print(f"  {S('FILTER_STAT_TOTAL')}: {stats['total_files']}")
        print(f"  {S('FILTER_STAT_NOISE')}: {stats['noise_files']}")
        print(f"  {S('FILTER_STAT_FAILED')}: {stats['failed_files']}")
        
        # 显示前10个组的详细信息
        for i, (group_name, count) in enumerate(list(stats['groups'].items())[:10]):
            print(f"    {group_name}: {count} 张")
        
        if len(stats['groups']) > 10:
            print(f"    ... 还有 {len(stats['groups']) - 10} 个组")
        
        return stats
    
    def _merge_groups_to_aligned(self, stats: Dict, output_dir: str = None) -> Dict:
        """
        将所有分组中的人脸图片合并回一个文件夹
        处理文件名冲突，保留最新的文件
        
        Args:
            stats: 统计字典
            output_dir: 输出目录名称（默认: aligned）
            
        Returns:
            更新后的统计字典
        """
        output_dir_name = output_dir or "aligned"
        output_path = self.faceset_path / output_dir_name
        output_path.mkdir(exist_ok=True)
        
        print(f"\nMerging all groups to '{output_dir_name}/'...")
        
        # 收集所有分组和噪声文件夹中的文件
        all_source_dirs = []
        
        # 添加所有分组目录
        for group_id in range(stats['total_groups']):
            group_dir = self.faceset_path / f"faceid_group_{group_id:03d}"
            if group_dir.exists():
                all_source_dirs.append(group_dir)
        
        # 添加噪声目录
        noise_dir = self.faceset_path / "faceid_noise"
        if noise_dir.exists():
            all_source_dirs.append(noise_dir)
        
        if not all_source_dirs:
            print("No groups to merge")
            return stats
        
        # 统计信息
        merged_count = 0
        conflict_count = 0
        skipped_count = 0
        error_count = 0
        
        # 遍历所有源目录
        total_files = sum(len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png"))) for d in all_source_dirs)
        pbar = tqdm.tqdm(total=total_files, desc="Merging", unit="file", ascii=True)
        
        for source_dir in all_source_dirs:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [f for f in source_dir.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]
            
            for src_file in image_files:
                try:
                    dst_file = output_path / src_file.name
                    
                    # 检查是否存在冲突
                    if dst_file.exists():
                        # 比较元数据，保留较新的
                        if self._should_replace_file(src_file, dst_file):
                            # 目标文件存在但源文件更新，替换
                            shutil.move(str(src_file), str(dst_file))
                            conflict_count += 1
                        else:
                            # 目标文件更新或相同，跳过
                            skipped_count += 1
                    else:
                        # 没有冲突，直接移动
                        shutil.move(str(src_file), str(dst_file))
                        merged_count += 1
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error merging {src_file.name}: {e}")
                    error_count += 1
                    pbar.update(1)
        
        pbar.close()
        
        print(f"\nMerge complete:")
        print(f"  Merged: {merged_count}")
        print(f"  Conflicts (replaced): {conflict_count}")
        print(f"  Skipped (older): {skipped_count}")
        print(f"  Errors: {error_count}")
        
        # 更新统计
        stats['merged_files'] = merged_count
        stats['conflict_replaced'] = conflict_count
        stats['conflict_skipped'] = skipped_count
        stats['merge_errors'] = error_count
        
        return stats
    
    def _should_replace_file(self, src_file: Path, dst_file: Path) -> bool:
        """
        判断是否应该用源文件替换目标文件
        仅使用文件修改时间，不加载元数据（提高性能）
        
        Args:
            src_file: 源文件路径
            dst_file: 目标文件路径
            
        Returns:
            True 如果应该替换，False 否则
        """
        try:
            # 直接使用文件修改时间，避免加载元数据
            src_mtime = src_file.stat().st_mtime
            dst_mtime = dst_file.stat().st_mtime
            
            return src_mtime > dst_mtime
            
        except Exception as e:
            # 出错时默认不替换，保护已有数据
            print(f"Warning: Could not compare files {src_file.name}: {e}")
            return False
    
    def merge_subfolders_to_aligned(self, output_dir: str = "aligned") -> Dict:
        """
        将当前目录下所有一级子文件夹中的图片合并到指定文件夹
        安全限制：只遍历直接子文件夹，不递归到更深层
        
        Args:
            output_dir: 输出目录名称（默认: aligned）
                       注意：如果 faceset_path 本身就是 aligned 目录，
                       则直接合并到 faceset_path，不再创建子目录
            
        Returns:
            统计结果
        """
        # 判断 faceset_path 是否已经是目标目录
        # 如果路径以 'aligned' 结尾，说明用户已经指定了 aligned 目录
        path_name = self.faceset_path.name.lower()
        if path_name == output_dir.lower():
            # 用户输入的已经是 aligned 目录，直接使用该目录作为输出
            output_path = self.faceset_path
            print(f"\nMerging images from subfolders to current directory...")
        else:
            # 用户输入的是父目录，需要创建子目录
            output_path = self.faceset_path / output_dir
            output_path.mkdir(exist_ok=True)
            print(f"\nMerging images from subfolders to '{output_dir}/'...")
        
        print("Safety: Only scanning immediate subdirectories (not recursive)")
        
        # 获取所有一级子文件夹（排除输出目录本身）
        subdirs = [
            d for d in self.faceset_path.iterdir() 
            if d.is_dir() and d != output_path
        ]
        
        if not subdirs:
            print("No subdirectories found")
            return {'merged': 0, 'conflicts': 0, 'skipped': 0, 'errors': 0}
        
        print(f"Found {len(subdirs)} subdirectories")
        
        # 记录原本就是空的目录（合并后不删除）
        originally_empty_dirs = set()
        for subdir in subdirs:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            has_images = any(
                f.is_file() and f.suffix.lower() in image_extensions
                for f in subdir.iterdir()
            )
            if not has_images:
                originally_empty_dirs.add(subdir)
        
        if originally_empty_dirs:
            print(f"Found {len(originally_empty_dirs)} originally empty directories (will be preserved)")
        
        # 统计信息
        merged_count = 0
        conflict_count = 0
        skipped_count = 0
        error_count = 0
        
        # 收集所有要移动的文件
        all_files = []
        for subdir in subdirs:
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            # 只获取直接子文件夹中的文件，不递归
            image_files = [
                f for f in subdir.iterdir() 
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            all_files.extend(image_files)
        
        print(f"Total images to process: {len(all_files)}")
        
        if not all_files:
            print("No images found in subdirectories")
            return {'merged': 0, 'conflicts': 0, 'skipped': 0, 'errors': 0}
        
        # 执行移动
        pbar = tqdm.tqdm(total=len(all_files), desc="Merging", unit="file", ascii=True)
        
        for src_file in all_files:
            try:
                dst_file = output_path / src_file.name
                
                # 检查是否存在冲突
                if dst_file.exists():
                    # 比较元数据，保留较新的
                    if self._should_replace_file(src_file, dst_file):
                        # 目标文件存在但源文件更新，替换
                        shutil.move(str(src_file), str(dst_file))
                        conflict_count += 1
                    else:
                        # 目标文件更新或相同，跳过
                        skipped_count += 1
                else:
                    # 没有冲突，直接移动
                    shutil.move(str(src_file), str(dst_file))
                    merged_count += 1
                
                pbar.update(1)
                
            except Exception as e:
                print(f"Error merging {src_file.name}: {e}")
                error_count += 1
                pbar.update(1)
        
        pbar.close()
        
        print(f"\nMerge complete:")
        print(f"  Merged: {merged_count}")
        print(f"  Conflicts (replaced): {conflict_count}")
        print(f"  Skipped (older): {skipped_count}")
        print(f"  Errors: {error_count}")
        
        # 清理空目录：删除那些因为文件被移走而变空的目录
        print("\nCleaning up empty directories...")
        removed_dirs = []
        kept_empty_dirs = []
        
        for subdir in subdirs:
            # 跳过原本就是空的目录
            if subdir in originally_empty_dirs:
                kept_empty_dirs.append(subdir.name)
                continue
            
            # 检查目录是否为空（或只包含非图片文件）
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            remaining_files = [
                f for f in subdir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            if not remaining_files:
                # 目录已空，删除
                try:
                    shutil.rmtree(str(subdir))
                    removed_dirs.append(subdir.name)
                except Exception as e:
                    print(f"Warning: Could not remove directory {subdir.name}: {e}")
        
        if removed_dirs:
            print(f"Removed {len(removed_dirs)} empty directories: {', '.join(removed_dirs)}")
        if kept_empty_dirs:
            print(f"Preserved {len(kept_empty_dirs)} originally empty directories: {', '.join(kept_empty_dirs)}")
        if not removed_dirs and not kept_empty_dirs:
            print("No empty directories to clean up")
        
        return {
            'merged': merged_count,
            'conflicts': conflict_count,
            'skipped': skipped_count,
            'errors': error_count,
            'removed_empty_dirs': len(removed_dirs),
            'preserved_empty_dirs': len(kept_empty_dirs)
        }


class PositionFilter:
    """位置过滤器 - 基于人脸坐标聚类"""
    
    def __init__(self, faceset_path: Path, metadata_file: Optional[Path] = None):
        """
        初始化位置过滤器
        
        Args:
            faceset_path: 人脸数据集路径
            metadata_file: 元数据文件路径（可选）
        """
        self.faceset_path = Path(faceset_path)
        self.metadata_file = metadata_file or (self.faceset_path / "metadata.h5")
        self.metadata = {}
        
        # 加载元数据
        self._load_metadata()
    
    def _load_metadata(self):
        """加载 HDF5 格式的元数据"""
        self.metadata = _load_metadata_h5(self.metadata_file)
    
    def filter_by_position(self, eps: float = 50.0, min_samples: int = 2) -> Dict:
        """
        根据人脸在原始图像中的位置进行聚类分组
        适用于固定摄像头、视频会议等场景
        
        Args:
            eps: DBSCAN 邻域半径（像素），默认 50.0
                 表示位置距离小于 50 像素的人脸视为同一位置
            min_samples: DBSCAN 最小样本数，默认 2
            
        Returns:
            分组统计结果
        """
        print(S('FILTER_START_POSITION_FILTER', eps))
        
        # 扫描所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([
            f for f in self.faceset_path.iterdir()
            if f.suffix.lower() in image_extensions and f.is_file()
        ])
        
        if not image_files:
            print(S('FILTER_NO_IMAGES_FOUND'))
            return {}
        
        print(S('FILTER_FOUND_IMAGES', len(image_files)))
        
        # ========== 第一阶段：提取位置信息 ==========
        print(S('FILTER_PHASE1_EXTRACTING_POSITIONS'))
        
        positions_dict = {}  # {filename: (center_x, center_y)}
        filename_list = []
        failed_files = []
        
        pbar = tqdm.tqdm(total=len(image_files), desc=S('FILTER_PROGRESS_EXTRACTING'), unit="img", ascii=True)
        
        for img_file in image_files:
            filename = img_file.name
            try:
                meta = self.metadata.get(filename, {})
                source_rect = meta.get('source_rect', None)
                
                if source_rect is None:
                    failed_files.append(filename)
                    pbar.update(1)
                    continue
                
                # 计算人脸中心点
                l, t, r, b = source_rect[:4]
                center_x = (l + r) / 2.0
                center_y = (t + b) / 2.0
                
                positions_dict[filename] = (center_x, center_y)
                filename_list.append(filename)
                
            except Exception as e:
                print(S('FILTER_PROCESS_ERROR', img_file.name, e))
                failed_files.append(filename)
            finally:
                pbar.update(1)
        
        pbar.close()
        
        print(S('FILTER_POSITIONS_EXTRACTED', len(positions_dict), len(failed_files)))
        
        if not positions_dict:
            print(S('FILTER_NO_POSITIONS'))
            return {}
        
        # ========== 第二阶段：DBSCAN 聚类 ==========
        print(S('FILTER_PHASE2_CLUSTERING_POSITIONS'))
        
        # 构建位置矩阵
        position_matrix = np.array([positions_dict[f] for f in filename_list])
        
        print(S('FILTER_RUNNING_DBSCLAN_POSITION', eps, min_samples))
        
        # 执行 DBSCAN 聚类
        db = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        labels = db.fit_predict(position_matrix)
        
        # 统计聚类结果
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(S('FILTER_CLUSTERING_RESULTS'))
        print(S('FILTER_TOTAL_CLUSTERS', n_clusters))
        print(S('FILTER_NOISE_POINTS', n_noise))
        
        # 构建分组字典
        groups = {}
        noise_files = []
        
        for i, label in enumerate(labels):
            filename = filename_list[i]
            if label == -1:
                noise_files.append(filename)
            else:
                if label not in groups:
                    groups[label] = []
                groups[label].append(filename)
        
        # 重新编号组
        renumbered_groups = {}
        for new_id, (old_id, filenames) in enumerate(sorted(groups.items())):
            renumbered_groups[new_id] = filenames
        
        groups = renumbered_groups
        
        # ========== 第三阶段：批量移动 ==========
        print(S('FILTER_MOVING_FILES', len(groups)))
        
        stats = {
            'total_groups': len(groups),
            'total_files': sum(len(g) for g in groups.values()),
            'failed_files': len(failed_files),
            'noise_files': len(noise_files),
            'groups': {}
        }
        
        # 构建移动列表
        move_list = []
        
        for group_id, filenames in groups.items():
            # 创建组文件夹
            group_dir = self.faceset_path / f"position_group_{group_id:03d}"
            group_dir.mkdir(exist_ok=True)
            
            # 添加到移动列表
            for filename in filenames:
                src = self.faceset_path / filename
                dst = group_dir / filename
                move_list.append((src, dst))
            
            stats['groups'][f"group_{group_id:03d}"] = len(filenames)
        
        # 处理噪声点
        if noise_files:
            noise_dir = self.faceset_path / "position_noise"
            noise_dir.mkdir(exist_ok=True)
            
            for filename in noise_files:
                src = self.faceset_path / filename
                dst = noise_dir / filename
                move_list.append((src, dst))
            
            print(S('FILTER_MOVING_NOISE', len(noise_files)))
        
        # 执行批量移动
        if move_list:
            pbar = tqdm.tqdm(total=len(move_list), desc=S('FILTER_PROGRESS_MOVING'), unit="file", ascii=True)
            
            for src, dst in move_list:
                try:
                    shutil.move(str(src), str(dst))
                except Exception as e:
                    print(S('FILTER_MOVE_ERROR', src.name, e))
                    stats['errors'] = stats.get('errors', 0) + 1
                finally:
                    pbar.update(1)
            
            pbar.close()
        
        print(S('FILTER_POSITION_COMPLETE'))
        print(f"  {S('FILTER_STAT_GROUPS')}: {stats['total_groups']}")
        print(f"  {S('FILTER_STAT_TOTAL')}: {stats['total_files']}")
        print(f"  {S('FILTER_STAT_NOISE')}: {stats['noise_files']}")
        print(f"  {S('FILTER_STAT_FAILED')}: {stats['failed_files']}")
        
        # 显示前10个组的详细信息
        for i, (group_name, count) in enumerate(list(stats['groups'].items())[:10]):
            print(f"    {group_name}: {count} 张")
        
        if len(stats['groups']) > 10:
            print(f"    ... 还有 {len(stats['groups']) - 10} 个组")
        
        return stats


def _calculate_phash_worker(args):
    """
    多进程工作函数：计算单张图片的 phash
    
    Args:
        args: (image_path_str, hash_method)
        
    Returns:
        (filename, hash_array_or_None, error_msg_or_None)
    """
    image_path_str, hash_method = args
    
    try:
        # 读取图像
        img = cv2.imread(image_path_str)
        if img is None:
            return (Path(image_path_str).name, None, "Failed to read image")
        
        # 根据方法计算哈希
        if hash_method == 'phash':
            hash_value = RepeatedFilter.phash(img)
        elif hash_method == 'ahash':
            hash_value = RepeatedFilter.ahash(img)
        elif hash_method == 'dhash':
            hash_value = RepeatedFilter.dhash(img)
        else:
            hash_value = RepeatedFilter.phash(img)
        
        return (Path(image_path_str).name, hash_value, None)
    except Exception as e:
        return (Path(image_path_str).name, None, str(e))


class RepeatedFilter:
    """重复图片过滤器 - 基于感知哈希去重"""

    def __init__(self, faceset_path: Path):
        self.faceset_path = Path(faceset_path)
        self.metadata_file = self.faceset_path / "metadata.json"
        self.metadata = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """加载 HDF5 格式的元数据"""
        self.metadata = _load_metadata_h5(self.metadata_file)

    @staticmethod
    def phash(img: np.ndarray, hash_size: int = 8) -> Optional[np.ndarray]:
        try:
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(np.float32(resized))
            dct_low_freq = dct[:hash_size, :hash_size]
            avg = np.mean(dct_low_freq)
            hash_value = (dct_low_freq > avg).astype(int)
            return hash_value.flatten()
        except Exception as e:
            return None

    @staticmethod
    def ahash(img: np.ndarray, hash_size: int = 8) -> Optional[np.ndarray]:
        try:
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
            avg = np.mean(resized)
            hash_value = (resized > avg).astype(int)
            return hash_value.flatten()
        except Exception as e:
            return None

    @staticmethod
    def dhash(img: np.ndarray, hash_size: int = 8) -> Optional[np.ndarray]:
        try:
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
            hash_value = []
            for row in range(resized.shape[0]):
                for col in range(resized.shape[1] - 1):
                    if resized[row, col] > resized[row, col + 1]:
                        hash_value.append(1)
                    else:
                        hash_value.append(0)
            return np.array(hash_value)
        except Exception as e:
            return None

    @staticmethod
    def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
        return int(np.sum(hash1 != hash2))

    @staticmethod
    def similarity(hash1, hash2) -> float:
        """
        计算两个哈希值的相似度
        
        Args:
            hash1: 可以是 np.ndarray, list, 或其他可转换为数组的对象
            hash2: 同上
        
        Returns:
            相似度 (0-1)
        """
        # 确保转换为 numpy 数组
        try:
            if not isinstance(hash1, np.ndarray):
                hash1 = np.array(hash1)
            if not isinstance(hash2, np.ndarray):
                hash2 = np.array(hash2)
            
            # 检查是否为有效数组
            if hash1.ndim == 0 or hash2.ndim == 0:
                # 标量值，无法计算相似度
                return 0.0
            
            total_bits = len(hash1)
            if total_bits == 0:
                return 0.0
            
            distance = RepeatedFilter.hamming_distance(hash1, hash2)
            return 1.0 - (distance / total_bits)
        except Exception as e:
            print(f"Warning: similarity calculation error: {e}")
            print(f"  hash1 type: {type(hash1)}, shape: {getattr(hash1, 'shape', 'N/A')}")
            print(f"  hash2 type: {type(hash2)}, shape: {getattr(hash2, 'shape', 'N/A')}")
            return 0.0

    def filter_repeated(self, threshold: float = 0.98, hash_method: str = 'phash', workers: int = None) -> Dict:
        """
        使用顺序扫描策略找出重复图片
        算法逻辑：
        1. 第一张图片作为起点A
        2. 向后扫描，找到第一个不与A相似的图片B
        3. A和A后面所有相似的都保留，其他的标记为重复
        4. B成为新起点，继续向后扫描
        5. 复杂度 O(n)，只需记录当前起点
        """
        import multiprocessing
        from multiprocessing import Pool

        print(S('FILTER_START_REPEATED_FILTER', threshold, hash_method))
        hash_functions = {'phash': self.phash, 'ahash': self.ahash, 'dhash': self.dhash}
        hash_func = hash_functions.get(hash_method, self.phash)

        if hash_method == 'phash':
            print(S('FILTER_HASH_METHOD_PHASH'))
        elif hash_method == 'ahash':
            print(S('FILTER_HASH_METHOD_AHASH'))
        elif hash_method == 'dhash':
            print(S('FILTER_HASH_METHOD_DHASH'))

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted(
            [f for f in self.faceset_path.iterdir() if f.suffix.lower() in image_extensions and f.is_file()])

        if not image_files:
            print(S('FILTER_NO_IMAGES_FOUND'))
            return {}

        total_images = len(image_files)
        print(S('FILTER_FOUND_IMAGES', total_images))

        repeated_dir = self.faceset_path / "repeated"
        repeated_dir.mkdir(exist_ok=True)

        if workers is None:
            workers = multiprocessing.cpu_count()

        # ========== 第一阶段：计算哈希并保存到元数据 ==========
        print(S('FILTER_PHASE1_CALCULATING_HASHES'))
        
        # 准备任务列表
        tasks = []
        valid_files = []
        
        for img_file in image_files:
            filename = img_file.name
            # 检查元数据中是否已有 phash
            meta = self.metadata.get(filename, {})
            if 'phash' in meta and meta['phash']:
                # 已有 phash，跳过计算
                continue
            else:
                # 需要计算
                tasks.append((str(img_file), hash_method))
                valid_files.append(img_file)
        
        # 多进程计算 phash（只计算缺失的）
        phash_results = {}  # {filename: hash_array}
        new_phash_count = 0
        
        if tasks:
            with Pool(processes=workers) as pool:
                pbar = tqdm.tqdm(total=len(tasks), desc=S('FILTER_PROGRESS_HASHING'), unit="img", ascii=True)
                
                # 使用 imap_unordered 获取结果
                for filename, hash_value, error in pool.imap_unordered(_calculate_phash_worker, tasks):
                    if error:
                        print(S('FILTER_HASH_ERROR', filename, error))
                    else:
                        if hash_value is not None:
                            phash_results[filename] = hash_value
                            
                            # 检查是否需要保存到元数据
                            meta = self.metadata.get(filename, {})
                            if 'phash' not in meta or not meta['phash']:
                                meta['phash'] = hash_value.tolist()
                                self.metadata[filename] = meta
                                new_phash_count += 1
                    
                    pbar.update(1)
                
                pbar.close()
        
        # 保存新的 phash 到元数据
        if new_phash_count > 0:
            print(S('FILTER_SAVING_EMBEDDINGS').replace('embedding', 'phash'))
            try:
                import h5py
                with h5py.File(self.metadata_file, 'w') as f:
                    for filename, meta in self.metadata.items():
                        safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                        grp = f.create_group(safe_name)
                        # 存储原始文件名
                        grp.attrs['__original_filename__'] = filename
                        for key, value in meta.items():
                            if isinstance(value, (list, np.ndarray)):
                                grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                            elif isinstance(value, (int, float, str)):
                                grp.attrs[key] = value
                            else:
                                grp.attrs[key] = str(value)
                print(S('FILTER_EMBEDDINGS_SAVED', new_phash_count).replace('embedding', 'phash'))
            except Exception as e:
                print(S('FILTER_SAVE_METADATA_WARNING', e))
        else:
            print(S('FILTER_EMBEDDINGS_ALREADY_SAVED').replace('embedding', 'phash'))
        
        # 将所有图片（包括已有phash的）加入 valid_files
        valid_files = image_files
        
        # ========== 第二阶段：顺序扫描找出重复图片 ==========
        print(S('FILTER_PHASE2_SEQUENTIAL_SCAN'))
        
        # 收集所有图片的哈希值（按顺序）
        file_hash_list = []  # [(filename, hash_array, file_size), ...]
        
        pbar = tqdm.tqdm(total=len(valid_files), desc=S('FILTER_PROGRESS_LOADING_HASHES'), unit="img", ascii=True)
        
        for img_file in valid_files:
            filename = img_file.name
            
            # 从 phash_results 或元数据中获取 phash
            if filename in phash_results:
                hash_value = phash_results[filename]
            else:
                # 从元数据读取
                meta = self.metadata.get(filename, {})
                if 'phash' in meta and meta['phash']:
                    phash_data = meta['phash']
                    # 确保转换为 numpy 数组
                    if isinstance(phash_data, str):
                        # 字符串格式（十六进制），转换为二进制数组
                        try:
                            # 将十六进制字符串转换为整数，再转换为二进制数组
                            phash_int = int(phash_data, 16)
                            # 转换为 64 位二进制数组（phash 通常是 64 位）
                            hash_value = np.array([(phash_int >> i) & 1 for i in range(63, -1, -1)], dtype=np.int32)
                        except Exception as e:
                            print(f"Warning: Failed to convert phash string for {filename}: {e}")
                            pbar.update(1)
                            continue
                    elif isinstance(phash_data, list):
                        hash_value = np.array(phash_data, dtype=np.int32)
                    elif isinstance(phash_data, np.ndarray):
                        hash_value = phash_data
                    else:
                        # 其他类型（可能是标量），跳过
                        print(f"Warning: Invalid phash format for {filename}: {type(phash_data)}")
                        pbar.update(1)
                        continue
                else:
                    # 没有哈希值，跳过
                    pbar.update(1)
                    continue
            
            file_size = img_file.stat().st_size
            file_hash_list.append((filename, hash_value, file_size))
            pbar.update(1)
        
        pbar.close()
        
        if not file_hash_list:
            print("No valid images with hash found")
            return {}
        
        # 顺序扫描：O(n) 复杂度，直接移动重复文件
        print(S('FILTER_PHASE3_MARKING_DUPLICATES'))
        
        anchors = []  # 起点列表（保留的图片）
        moved_count = 0
        skipped_count = 0
        
        # 第一张图片作为第一个起点
        if file_hash_list:
            current_anchor_idx = 0
            anchors.append(file_hash_list[0])
        else:
            return {}
        
        # 从第二张图片开始扫描
        pbar = tqdm.tqdm(total=len(file_hash_list) - 1, desc=S('FILTER_PROGRESS_SCANNING'), unit="img", ascii=True)
        
        for i in range(1, len(file_hash_list)):
            current_file, current_hash, current_size = file_hash_list[i]
            
            # 只与当前起点对比
            anchor_file, anchor_hash, anchor_size = file_hash_list[current_anchor_idx]
            sim = self.similarity(current_hash, anchor_hash)
            
            if sim >= threshold:
                # 与当前起点相似，直接移动到 repeated 目录
                src = self.faceset_path / current_file
                dst = repeated_dir / current_file
                
                try:
                    if src.exists():
                        shutil.move(str(src), str(dst))
                        moved_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(S('FILTER_MOVE_ERROR', current_file, e))
                    skipped_count += 1
            else:
                # 不相似，成为新起点
                current_anchor_idx = i
                anchors.append((current_file, current_hash, current_size))
            
            pbar.update(1)
        
        pbar.close()
        
        duplicate_count = moved_count
        anchor_count = len(anchors)
        print(S('FILTER_FOUND_DUPLICATES', duplicate_count))
        print(f"  Anchors (kept): {anchor_count}")
        
        if skipped_count > 0:
            print(f"  跳过 {skipped_count} 个不存在的文件")
        
        remaining_images = total_images - moved_count
        reduction_percentage = (moved_count / total_images) * 100 if total_images > 0 else 0
        print(S('FILTER_REPEATED_COMPLETE'))
        print(f"  {S('FILTER_STAT_TOTAL')}: {total_images}")
        print(f"  {S('FILTER_STAT_MOVED')}: {moved_count}")
        print(f"  {S('FILTER_STAT_REMAINING')}: {remaining_images}")
        print(f"  {S('FILTER_STAT_REDUCTION')}: {reduction_percentage:.2f}%")
        print(f"  Anchor groups: {anchor_count}")
        
        return {
            'total': total_images,
            'moved': moved_count,
            'remaining': remaining_images,
            'reduction_percentage': reduction_percentage,
            'duplicate_count': duplicate_count,
            'anchor_count': anchor_count
        }

    @staticmethod
    def _compare_with_group(args):
        hash_value, group, threshold = args
        representative_hash = group[0]['hash']
        sim = RepeatedFilter.similarity(hash_value, representative_hash)
        return sim >= threshold, sim

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description=S('FILTER_DESCRIPTION'),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python Filter.py --input ".\\workspace\\faces" --mode blur
  python Filter.py --input ".\\workspace\\faces" --mode faceid --eps 0.3 --min-samples 2
  python Filter.py --input ".\\workspace\\faces" --mode position --threshold 50 --min-samples 2
  python Filter.py --input ".\\workspace\\faces" --mode repeated --threshold 0.98
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help=S('FILTER_ARG_INPUT')
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['blur', 'faceid', 'position', 'repeated'],
        required=False,  # 改为可选，因为 merge-only 模式不需要
        default=None,
        help=S('FILTER_ARG_MODE')
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help=S('FILTER_ARG_THRESHOLD')  # Used for blur and position modes only
    )
    
    parser.add_argument(
        '--eps',
        type=float,
        default=0.3,
        help=S('FILTER_ARG_EPS')
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=2,
        help=S('FILTER_ARG_MIN_SAMPLES')
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help=S('FILTER_ARG_WORKERS')
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['w600k_mbf', 'w600k_r50'],
        default=None,
        help=S('FILTER_ARG_MODEL')
    )
    
    parser.add_argument(
        '--merge-back',
        action='store_true',
        help='Merge all groups back to a single folder after grouping'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='aligned',
        help='Output directory name for merged files (default: aligned)'
    )
    
    parser.add_argument(
        '--merge-only',
        action='store_true',
        help='Only merge subfolders without performing face ID grouping'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    #依旧雷霆大字
    print("""
=======================================================================================================================
███████╗ █████╗  ██████╗███████╗███████╗██╗     ██╗████████╗███████╗██████╗     ████████╗ ██████╗  ██████╗ ██╗     
██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██║     ██║╚══██╔══╝██╔════╝██╔══██╗    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     
█████╗  ███████║██║     █████╗  █████╗  ██║     ██║   ██║   █████╗  ██████╔╝       ██║   ██║   ██║██║   ██║██║     
██╔══╝  ██╔══██║██║     ██╔══╝  ██╔══╝  ██║     ██║   ██║   ██╔══╝  ██╔══██╗       ██║   ██║   ██║██║   ██║██║     
██║     ██║  ██║╚██████╗███████╗██║     ███████╗██║   ██║   ███████╗██║  ██║       ██║   ╚██████╔╝╚██████╔╝███████╗
╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝     ╚══════╝╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
=======================================================================================================================
    """)
    
    args = parse_args()
    
    # 打印用户输入参数
    print("\n" + "="*80)
    print("FacesetProcessor Filter - 参数配置")
    print("="*80)
    print(f"输入路径: {args.input}")
    print(f"过滤模式: {args.mode if args.mode else 'merge-only'}")
    
    if args.mode == 'faceid':
        print(f"  - eps: {args.eps}")
        print(f"  - min_samples: {args.min_samples}")
        print(f"  - model: {args.model if args.model else 'auto-select'}")
        print(f"  - merge_back: {args.merge_back}")
        print(f"  - output_dir: {args.output_dir}")
    elif args.mode == 'blur':
        print(f"  - threshold: {args.threshold if args.threshold is not None else 20.0}")
        print(f"  - workers: {args.workers if args.workers else 'auto'}")
    elif args.mode == 'position':
        print(f"  - eps (threshold): {args.threshold if args.threshold is not None else 50.0}")
        print(f"  - min_samples: {args.min_samples if args.min_samples else 2}")
    elif args.mode == 'repeated':
        print(f"  - threshold: {args.threshold if args.threshold is not None else 0.98}")
        print(f"  - hash_method: phash")
        print(f"  - workers: {args.workers if args.workers else 'auto'}")
    elif args.merge_only:
        print(f"  - output_dir: {args.output_dir}")
    
    print("="*80 + "\n")
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(S('PATH_NOT_EXIST', input_path))
        return
    
    if not input_path.is_dir():
        print(S('INVALID_PATH_TYPE', input_path))
        return
    
    # 根据模式执行不同的过滤
    if args.merge_only:
        # 仅合并子文件夹模式（不需要加载模型）
        filter_obj = FaceIDFilter(input_path, skip_model_init=True)
        filter_obj.merge_subfolders_to_aligned(output_dir=args.output_dir)
    
    elif args.mode is None:
        # 如果没有指定 mode 且不是 merge-only，报错
        print("错误: 必须指定 --mode 参数或使用 --merge-only")
        return
    
    elif args.mode == 'blur':
        # 模糊过滤模式
        threshold = args.threshold if args.threshold is not None else 20.0
        filter_obj = QualityFilter(input_path)
        filter_obj.filter_by_quality(threshold=threshold, workers=args.workers)
    
    elif args.mode == 'faceid':
        # 人脸ID过滤模式
        filter_obj = FaceIDFilter(input_path, model_name=args.model)
        filter_obj.filter_by_face_id(
            eps=args.eps if args.eps is not None else 0.3,
            min_samples=args.min_samples if args.min_samples else 2,
            merge_back=args.merge_back,
            output_dir=args.output_dir
        )
    
    elif args.mode == 'position':
        # 位置过滤模式
        eps = args.threshold if args.threshold is not None else 50.0
        filter_obj = PositionFilter(input_path)
        filter_obj.filter_by_position(
            eps=eps,
            min_samples=args.min_samples if args.min_samples else 2
        )
    
    elif args.mode == 'repeated':
        # 重复图片过滤模式
        threshold = args.threshold if args.threshold is not None else 0.98
        hash_method = 'phash'  # 默认使用phash
        filter_obj = RepeatedFilter(input_path)
        filter_obj.filter_repeated(
            threshold=threshold,
            hash_method=hash_method,
            workers=args.workers
        )


if __name__ == '__main__':
    main()
