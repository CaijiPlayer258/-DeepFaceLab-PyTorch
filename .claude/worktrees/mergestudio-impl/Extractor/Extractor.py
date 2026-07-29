"""
DeepFaceLab Torch - Extractor Module
人脸提取器模块：从视频或图片中提取对齐后的人脸
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import traceback
import argparse
from typing import List, Tuple, Optional, Dict
import tqdm
from multiprocessing import cpu_count
import concurrent.futures
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 减少ONNX Runtime警告
import onnxruntime
onnxruntime.set_default_logger_severity(3)
import warnings
warnings.filterwarnings('ignore', module='onnxruntime')

# 导入modelhub模块
from modelhub.onnx import (
    BlazeFace, CenterFace, S3FD, YoloV5Face,
    InsightFace2D106, FaceMesh
)
from modelhub.onnx.YoloV8Face import YoloV8Face
from xlib.onnxruntime import get_cpu_device_info, get_available_devices_info
from facelib.LandmarksProcessor import get_transform_mat, get_canonical_68
import facelib

# 导入多语言支持
from strings import S


class DetectorFactory:
    """人脸检测器工厂类"""
    
    DETECTORS = {
        'BlazeFace': BlazeFace,
        'CenterFace': CenterFace,
        'S3FD': S3FD,
        'YoloV5Face': YoloV5Face,
        'YoloV8Face': YoloV8Face,  # Added: faster and more accurate than YoloV5
        # TODO: Implement these detectors
        # 'RetinaFace': None,
        # 'MTCNN': None,
    }
    
    @classmethod
    def create_detector(cls, detector_name: str, device_info):
        """Create face detector instance"""
        detector_class = cls.DETECTORS.get(detector_name)
        if detector_class is None:
            raise ValueError(f"Unsupported detector: {detector_name}")
        
        try:
            detector = detector_class(device_info)
            print(S('DETECTOR_LOADED', detector_name))
            return detector
        except Exception as e:
            print(S('LOAD_DETECTOR_FAILED', detector_name, e))
            raise


class LandmarkFactory:
    """特征点标记器工厂类"""
    
    LANDMARKS = {
        'insightface-2d106det': InsightFace2D106,
        '2DFAN-4': None,  # TODO: 使用facelib.FANExtractor
        'Google-mediapipe': FaceMesh,
    }
    
    @classmethod
    def create_landmarker(cls, landmark_name: str, device_info):
        """创建特征点标记器实例"""
        landmark_class = cls.LANDMARKS.get(landmark_name)
        if landmark_class is None:
            raise ValueError(f"不支持的特征点标记器: {landmark_name}")
        
        try:
            landmarker = landmark_class(device_info)
            print(S('LANDMARKER_LOADED', landmark_name))
            return landmarker
        except Exception as e:
            print(S('LOAD_LANDMARKER_FAILED', landmark_name, e))
            raise


def detect_faces_multi_angle(detector, image: np.ndarray, angles: List[int] = [0]) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces from multiple rotation angles and merge results
    
    Args:
        detector: Face detector instance
        image: Input image (BGR)
        angles: List of rotation angles in degrees (clockwise): [0, 90, 180, 270]
        
    Returns:
        List of face rects with angle info: [(angle, l, t, r, b), ...]
    """
    all_detections = []
    h, w = image.shape[:2]
    
    for angle in angles:
        if angle == 0:
            rotated_img = image
        else:
            # Rotate image clockwise
            if angle == 90:
                rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_img = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                rotated_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                print(f"WARNING: Unsupported angle {angle}, skipping")
                continue
        
        # Detect faces on rotated image
        try:
            # YoloV8Face doesn't support fixed_window parameter
            if isinstance(detector, YoloV8Face):
                results = detector.extract(rotated_img)
            else:
                results = detector.extract(rotated_img, fixed_window=0)
            if not results or len(results) == 0:
                continue
            
            faces = results[0] if isinstance(results[0], list) else results
            
            # Convert coordinates back to original image space
            for face_rect in faces:
                l, t, r, b = face_rect[:4]
                
                # Rotate coordinates back to original orientation
                if angle == 90:
                    # Rotated 90 CW: new_w=h, new_h=w
                    # Original: (x, y) -> Rotated: (y, new_w-x)
                    # Back: (x', y') -> (new_h-y', x')
                    orig_l = t
                    orig_t = w - r
                    orig_r = b
                    orig_b = w - l
                elif angle == 180:
                    # Rotated 180: same dimensions
                    # Back: (x', y') -> (w-x', h-y')
                    orig_l = w - r
                    orig_t = h - b
                    orig_r = w - l
                    orig_b = h - t
                elif angle == 270:
                    # Rotated 270 CW (90 CCW): new_w=h, new_h=w
                    # Back: (x', y') -> (y', new_w-x')
                    orig_l = h - b
                    orig_t = l
                    orig_r = h - t
                    orig_b = r
                else:
                    orig_l, orig_t, orig_r, orig_b = l, t, r, b
                
                # Ensure coordinates are valid
                orig_l = max(0, int(orig_l))
                orig_t = max(0, int(orig_t))
                orig_r = min(w, int(orig_r))
                orig_b = min(h, int(orig_b))
                
                if orig_r > orig_l and orig_b > orig_t:
                    all_detections.append((angle, orig_l, orig_t, orig_r, orig_b))
        except Exception as e:
            print(f"WARNING: Detection failed at angle {angle}: {e}")
    
    # Remove duplicate detections using IoU
    unique_detections = remove_duplicate_detections(all_detections, iou_threshold=0.5)
    
    return unique_detections


def remove_duplicate_detections(detections: List[Tuple[int, int, int, int, int]], 
                                iou_threshold: float = 0.5) -> List[Tuple[int, int, int, int, int]]:
    """
    Remove duplicate face detections based on IoU (Intersection over Union)
    
    Args:
        detections: List of (angle, l, t, r, b)
        iou_threshold: IoU threshold for considering duplicates
        
    Returns:
        Filtered list of unique detections
    """
    if not detections:
        return []
    
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes (angle, l, t, r, b)"""
        _, l1, t1, r1, b1 = box1
        _, l2, t2, r2, b2 = box2
        
        # Calculate intersection
        inter_l = max(l1, l2)
        inter_t = max(t1, t2)
        inter_r = min(r1, r2)
        inter_b = min(b1, b2)
        
        if inter_r <= inter_l or inter_b <= inter_t:
            return 0.0
        
        inter_area = (inter_r - inter_l) * (inter_b - inter_t)
        
        # Calculate union
        area1 = (r1 - l1) * (b1 - t1)
        area2 = (r2 - l2) * (b2 - t2)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    # Sort by confidence (use box area as proxy, larger boxes first)
    sorted_detections = sorted(detections, key=lambda x: (x[3]-x[1])*(x[4]-x[2]), reverse=True)
    
    keep = []
    for detection in sorted_detections:
        is_duplicate = False
        for kept in keep:
            iou = calculate_iou(detection, kept)
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep.append(detection)
    
    return keep


def detect_and_align_on_resized(detector, landmarker, image: np.ndarray, fixed_window: int = 0, 
                                image_size_fixed: Optional[int] = None,
                                face_type_str: str = 'whole_face',
                                detection_angles: List[int] = None) -> List[Dict]:
    """
    Complete face processing pipeline on (possibly resized) image
    Returns normalized results that can be mapped to original image
    
    Args:
        detector: Face detector
        landmarker: Landmark detector
        image: Input image (will be resized if needed)
        fixed_window: Pre-resize width (0 = no resize)
        image_size_fixed: Fixed output size for alignment
        face_type_str: Face type string
        detection_angles: List of angles for multi-angle detection [0, 90, 180, 270]
        
    Returns:
        List of dicts with normalized coordinates and transformation matrices
    """
    h_orig, w_orig = image.shape[:2]
    scale_factor = 1.0
    working_image = image
    
    # Step 1: Resize if needed
    if fixed_window > 0 and w_orig > fixed_window:
        scale_factor = w_orig / fixed_window
        new_h = int(h_orig / scale_factor)
        new_w = fixed_window
        working_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Step 2: Detect faces on working image (with multi-angle support)
    if detection_angles is None:
        detection_angles = [0]  # Default: only 0 degree
    
    if len(detection_angles) == 1 and detection_angles[0] == 0:
        # Single angle detection (fast path)
        # YoloV8Face doesn't support fixed_window parameter
        if isinstance(detector, YoloV8Face):
            results = detector.extract(working_image)
        elif isinstance(detector, (BlazeFace, YoloV5Face, CenterFace, S3FD)):
            results = detector.extract(working_image, fixed_window=0)
        else:
            results = detector.extract(working_image)
        
        if not results or len(results) == 0:
            return []
        
        faces = results[0] if isinstance(results[0], list) else results
        # Add angle info (all 0 degrees)
        faces_with_angle = [(0, *face[:4]) for face in faces]
    else:
        # Multi-angle detection
        faces_with_angle = detect_faces_multi_angle(detector, working_image, detection_angles)
        
        if not faces_with_angle:
            return []
    
    h_work, w_work = working_image.shape[:2]
    
    # Step 3: Extract landmarks on working image
    # Note: faces_with_angle contains coordinates in working_image space,
    # but we need to rotate the image back to the detection angle to extract landmarks correctly
    landmarks_list = []
    for face_idx, (angle, l, t, r, b) in enumerate(faces_with_angle):
        if angle == 0:
            # No rotation needed - extract directly
            face_rect_for_landmark = (l, t, r, b)
            lmks = extract_landmarks(landmarker, working_image, [face_rect_for_landmark])
            
            if lmks and lmks[0] is not None:
                landmarks_list.append(lmks[0])
            else:
                landmarks_list.append(None)
        else:
            # For non-zero angles, we need to:
            # 1. Rotate the working image to match the detection angle
            # 2. Transform the face rect to the rotated image space
            # 3. Extract landmarks on the rotated image
            # 4. Transform landmarks back to original working_image space
            
            if angle == 90:
                rotated_working = cv2.rotate(working_image, cv2.ROTATE_90_CLOCKWISE)
                rot_h, rot_w = rotated_working.shape[:2]
                # Transform face rect from original to rotated space
                # Original (l,t,r,b) -> Rotated: (t, w-r, b, w-l)
                rot_l, rot_t, rot_r, rot_b = t, w_work - r, b, w_work - l
            elif angle == 180:
                rotated_working = cv2.rotate(working_image, cv2.ROTATE_180)
                rot_h, rot_w = h_work, w_work
                # Transform face rect 180 degrees
                rot_l, rot_t, rot_r, rot_b = w_work - r, h_work - b, w_work - l, h_work - t
            elif angle == 270:
                rotated_working = cv2.rotate(working_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rot_h, rot_w = rotated_working.shape[:2]
                # Transform face rect 270 CW
                rot_l, rot_t, rot_r, rot_b = h_work - b, l, h_work - t, r
            else:
                rotated_working = working_image
                rot_h, rot_w = h_work, w_work
                rot_l, rot_t, rot_r, rot_b = l, t, r, b
            
            # Extract landmarks on rotated image with transformed face rect
            face_rect_for_landmark = (rot_l, rot_t, rot_r, rot_b)
            lmks = extract_landmarks(landmarker, rotated_working, [face_rect_for_landmark])
            
            if lmks and lmks[0] is not None:
                pts = lmks[0].copy()
                # Transform landmarks back to original working_image space
                if angle == 90:
                    # Rotated 90 CW: (x', y') where x'=y, y'=w-x
                    # Reverse: x = rot_w - y', y = x'
                    temp_x = pts[:, 0].copy()
                    pts[:, 0] = pts[:, 1]
                    pts[:, 1] = rot_w - temp_x
                elif angle == 180:
                    # Rotated 180: (x', y') where x'=w-x, y'=h-y
                    # Reverse: x = rot_w - x', y = rot_h - y'
                    pts[:, 0] = rot_w - pts[:, 0]
                    pts[:, 1] = rot_h - pts[:, 1]
                elif angle == 270:
                    # Rotated 270 CW: (x', y') where x'=h-y, y'=x
                    # Reverse: x = y', y = rot_h - x'
                    temp_x = pts[:, 0].copy()
                    pts[:, 0] = rot_h - pts[:, 1]
                    pts[:, 1] = temp_x
                landmarks_list.append(pts)
            else:
                landmarks_list.append(None)
    
    # Step 4: Process each face - get normalized transformation matrix
    results_list = []
    
    for face_idx, ((angle, l, t, r, b), landmarks) in enumerate(zip(faces_with_angle, landmarks_list)):
        if landmarks is None:
            continue
        
        # Convert landmarks to standard 68 points
        if len(landmarks) == 106:
            landmarks_for_align = landmark106to68(landmarks)
        elif len(landmarks) == 468:
            landmarks_for_align = landmark468to68(landmarks)
        elif len(landmarks) > 68:
            landmarks_for_align = landmarks[:68]
        else:
            landmarks_for_align = landmarks
        
        # Calculate output size based on landmarks in original image
        if image_size_fixed is not None and image_size_fixed > 0:
            out_size = image_size_fixed
        else:
            # Map landmarks to original image coordinates
            landmarks_orig_temp = landmarks_for_align.copy()
            landmarks_orig_temp[:, 0] *= scale_factor
            landmarks_orig_temp[:, 1] *= scale_factor
            
            # Calculate what size get_transform_mat will extract from original image
            # We need to replicate the logic inside get_transform_mat to predict the extracted region size
            import numpy.linalg as npla
            
            # Use same landmarks subset as get_transform_mat (points 17:49 and 54:55)
            lm_subset = np.concatenate([landmarks_orig_temp[17:49], landmarks_orig_temp[54:55]])
            
            # Estimate transform to unit space
            from facelib.LandmarksProcessor import umeyama, transform_points, landmarks_2D_new
            mat_unit = umeyama(lm_subset, landmarks_2D_new, True)[0:2]
            
            # Get corner points in original image space
            g_p = transform_points(np.float32([(0,0),(1,0),(1,1),(0,1),(0.5,0.5)]), mat_unit, True)
            
            # Calculate diagonal length
            diag_vec = g_p[2] - g_p[0]
            diag_len = npla.norm(diag_vec)
            
            # Get padding based on face type
            face_type_map = {
                'half_face': 0.0,
                'midfull_face': 0.0675,
                'full_face': 0.2109375,
                'whole_face': 0.40,
                'head': 0.70
            }
            padding = face_type_map.get(face_type_str, 0.40)  # Default to whole_face
            
            # Calculate mod (half-diagonal of the extracted square)
            mod = diag_len * (padding * np.sqrt(2.0) + 0.5)
            
            # The extracted square has diagonal = 2 * mod
            # So side length = (2 * mod) / sqrt(2) = mod * sqrt(2)
            extracted_size = mod * np.sqrt(2.0)
            
            # Use the predicted extracted size directly as output size
            out_size = int(extracted_size)
            out_size = (out_size // 2) * 2  # Ensure even number
        
        try:
            # Convert face_type string to FaceType enum
            face_type_enum_map = {
                'half_face': facelib.FaceType.HALF,
                'midfull_face': facelib.FaceType.MID_FULL,
                'full_face': facelib.FaceType.FULL,
                'whole_face': facelib.FaceType.WHOLE_FACE,
                'head': facelib.FaceType.HEAD
            }
            face_type_enum = face_type_enum_map.get(face_type_str, facelib.FaceType.WHOLE_FACE)
            
            # Get transformation matrix (in working image coordinates)
            mat_work = get_transform_mat(landmarks_for_align, out_size, face_type_enum)
            
            # Normalize the transformation matrix to [0, 1] range relative to working image
            # The affine transform maps from source (working image) to destination (out_size x out_size)
            # We need to store it in a way that can be applied to original image
            
            # Store normalized data
            result = {
                'face_rect': (l, t, r, b),  # In working image coordinates
                'detection_angle': angle,  # Detection angle used
                'landmarks': landmarks_for_align,  # In working image coordinates
                'transform_mat': mat_work,  # Affine matrix for working image
                'out_size': out_size,
                'scale_factor': scale_factor,
                'orig_size': (w_orig, h_orig),
                'work_size': (w_work, h_work),
                'face_type': face_type_str,  # Store face type string for alignment
                'face_type_enum': face_type_enum  # Store face type enum for metadata
            }
            results_list.append(result)
            
        except Exception as e:
            print(S('ALIGN_SAVE_FAILED', f"face {face_idx}", 0, e))
    
    return results_list


def apply_alignment_to_original(image: np.ndarray, face_data: Dict) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Apply stored transformation to original resolution image
    
    Args:
        image: Original resolution image
        face_data: Normalized face data from detect_and_align_on_resized
        
    Returns:
        Tuple of (aligned_face, aligned_landmarks, source_rect_in_original)
    """
    scale_factor = face_data['scale_factor']
    orig_w, orig_h = face_data['orig_size']
    work_w, work_h = face_data['work_size']
    out_size = face_data['out_size']
    
    # Scale landmarks from working image to original image
    landmarks_work = face_data['landmarks']
    landmarks_orig = landmarks_work.copy()
    landmarks_orig[:, 0] *= scale_factor
    landmarks_orig[:, 1] *= scale_factor
    
    # Scale face rect from working image to original image (use round for better precision)
    l, t, r, b = face_data['face_rect']
    face_rect_orig = (
        round(l * scale_factor),
        round(t * scale_factor),
        round(r * scale_factor),
        round(b * scale_factor)
    )
    
    # Recompute transformation matrix using original image landmarks
    face_type_str = face_data.get('face_type', 'whole_face')
    face_type_enum_map = {
        'half_face': facelib.FaceType.HALF,
        'midfull_face': facelib.FaceType.MID_FULL,
        'full_face': facelib.FaceType.FULL,
        'whole_face': facelib.FaceType.WHOLE_FACE,
        'head': facelib.FaceType.HEAD
    }
    face_type_enum = face_type_enum_map.get(face_type_str, facelib.FaceType.WHOLE_FACE)
    
    mat_orig = get_transform_mat(landmarks_orig, out_size, face_type_enum)
    
    # Apply affine transform on ORIGINAL image
    aligned_face = cv2.warpAffine(image, mat_orig, (out_size, out_size), 
                                 flags=cv2.INTER_LANCZOS4)
    
    # Transform landmarks to aligned space
    aligned_landmarks = facelib.LandmarksProcessor.transform_points(landmarks_orig, mat_orig)
    
    return aligned_face, aligned_landmarks, face_rect_orig


def extract_landmarks(landmarker, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """
    Extract facial landmarks
    
    Args:
        landmarker: Landmark detector instance
        image: BGR image (original)
        faces: Face bounding box list (MUST be in original image coordinates)
        
    Returns:
        Landmark list, each element is (N, 2) array (original image coordinates)
    """
    landmarks_list = []
    
    for face_idx, face_rect in enumerate(faces):
        # Use float coordinates to preserve precision
        l, t, r, b = face_rect[:4]
        if not isinstance(l, int):
            l, t, r, b = int(l), int(t), int(r), int(b)
        
        # Verify face rect is within image bounds
        h, w = image.shape[:2]
        if l < 0 or t < 0 or r > w or b > h:
            print(f"WARNING: Face {face_idx} bbox ({l},{t},{r},{b}) out of image bounds ({w}x{h}), clipping...")
            l = max(0, l)
            t = max(0, t)
            r = min(w, r)
            b = min(h, b)
        
        # Crop face region with margin (only for landmark detector input)
        # Use round() instead of int() to reduce bias
        margin = round((r - l) * 0.2)
        l_crop = max(0, l - margin)
        t_crop = max(0, t - margin)
        r_crop = min(w, r + margin)
        b_crop = min(h, b + margin)
        
        face_img = image[t_crop:b_crop, l_crop:r_crop]
        
        if face_img.size == 0:
            print(f"WARNING: Face {face_idx} cropped region is empty")
            landmarks_list.append(None)
            continue
        
        try:
            if isinstance(landmarker, InsightFace2D106):
                # Model returns coordinates relative to face_img
                lmks = landmarker.extract(face_img)
                if lmks is not None and len(lmks) > 0:
                    # Convert to original image coordinates: add crop offset
                    pts = lmks[0].copy()  # (106, 2)
                    pts[:, 0] += l_crop
                    pts[:, 1] += t_crop
                    landmarks_list.append(pts)
                else:
                    landmarks_list.append(None)
                    
            elif isinstance(landmarker, FaceMesh):
                # Model returns coordinates relative to face_img
                lmks = landmarker.extract(face_img)
                if lmks is not None and len(lmks) > 0:
                    pts = lmks[0][:, :2].copy()  # (468, 2)
                    pts[:, 0] += l_crop
                    pts[:, 1] += t_crop
                    landmarks_list.append(pts)
                else:
                    landmarks_list.append(None)
            else:
                landmarks_list.append(None)
        except Exception as e:
            print(S('LANDMARK_EXTRACT_ERROR', e))
            import traceback
            traceback.print_exc()
            landmarks_list.append(None)
    
    return landmarks_list


def sort_faces_by_distance(
    prev_faces: List[Tuple[int, int, int, int]],
    curr_faces: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Sort faces based on Euclidean distance to maintain consistency with previous frame
    
    Args:
        prev_faces: Previous frame faces (sorted)
        curr_faces: Current frame faces (unsorted)
        
    Returns:
        Sorted current frame faces
    """
    if not prev_faces or not curr_faces:
        return curr_faces
    
    if len(curr_faces) == 1:
        return curr_faces
    
    # Calculate center point for each face
    def get_center(face):
        l, t, r, b = face[:4]
        return ((l + r) / 2, (t + b) / 2)
    
    prev_centers = [get_center(f) for f in prev_faces]
    curr_centers = [get_center(f) for f in curr_faces]
    
    # Greedy matching: find nearest previous face for each current face
    matched_indices = set()
    sorted_faces = []
    
    for i, curr_center in enumerate(curr_centers):
        best_idx = -1
        best_dist = float('inf')
        
        for j, prev_center in enumerate(prev_centers):
            if j in matched_indices:
                continue
            
            dist = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                          (curr_center[1] - prev_center[1])**2)
            
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        
        if best_idx != -1:
            matched_indices.add(best_idx)
            sorted_faces.append(curr_faces[i])
    
    # Append unmatched faces to the end
    for i, face in enumerate(curr_faces):
        if i not in [list(matched_indices).index(j) if j in matched_indices else -1 
                     for j in range(len(prev_faces))]:
            if face not in sorted_faces:
                sorted_faces.append(face)
    
    return sorted_faces if sorted_faces else curr_faces


def sort_faces_by_distance_for_data(
    prev_faces: List[Tuple[int, int, int, int]],
    curr_faces: List[Tuple[int, int, int, int]]
) -> List[int]:
    """
    Sort indices based on face distance for face_data list sorting
    
    Args:
        prev_faces: Previous frame faces (sorted)
        curr_faces: Current frame faces (unsorted)
        
    Returns:
        List of indices to reorder curr_faces to match prev_faces order
    """
    if not prev_faces or not curr_faces:
        return list(range(len(curr_faces)))
    
    if len(curr_faces) == 1:
        return [0]
    
    # Calculate center point for each face
    def get_center(face):
        l, t, r, b = face[:4]
        return ((l + r) / 2, (t + b) / 2)
    
    prev_centers = [get_center(f) for f in prev_faces]
    curr_centers = [get_center(f) for f in curr_faces]
    
    # Greedy matching: find nearest previous face for each current face
    matched_prev = set()
    sorted_indices = [-1] * len(curr_faces)
    used_curr = set()
    
    # Match each previous face to nearest current face
    for j, prev_center in enumerate(prev_centers):
        best_idx = -1
        best_dist = float('inf')
        
        for i, curr_center in enumerate(curr_centers):
            if i in used_curr:
                continue
            
            dist = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                          (curr_center[1] - prev_center[1])**2)
            
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx != -1:
            sorted_indices[j] = best_idx
            used_curr.add(best_idx)
    
    # Fill in unmatched positions
    unused_curr = [i for i in range(len(curr_faces)) if i not in used_curr]
    fill_idx = 0
    for j in range(len(sorted_indices)):
        if sorted_indices[j] == -1 and fill_idx < len(unused_curr):
            sorted_indices[j] = unused_curr[fill_idx]
            fill_idx += 1
    
    return sorted_indices


# Global variables for worker processes (initialized once per process)
_worker_detector = None
_worker_landmarker = None
_worker_device_info = None
_worker_detector_name = None
_worker_landmark_name = None
_worker_fixed_window = 0


def init_worker_process(detector_name, landmark_name, device_info, fixed_window=0):
    """
    Initialize worker process with detector and landmarker (called once per process)
    """
    global _worker_detector, _worker_landmarker, _worker_device_info
    global _worker_detector_name, _worker_landmark_name, _worker_fixed_window
    
    _worker_detector_name = detector_name
    _worker_landmark_name = landmark_name
    _worker_device_info = device_info
    _worker_fixed_window = fixed_window
    
    try:
        _worker_detector = DetectorFactory.create_detector(detector_name, device_info)
        _worker_landmarker = LandmarkFactory.create_landmarker(landmark_name, device_info)
    except Exception as e:
        print(S('INIT_WORKER_FAILED', e))
        import traceback
        traceback.print_exc()


def process_single_frame(args):
    """
    Process single frame (for multiprocessing)
    Args:
        args: tuple (frame_idx, frame, image_size, output_path, input_path_name)
    Returns:
        tuple (frame_idx, saved_count, metadata_dict)
    """
    global _worker_detector, _worker_landmarker, _worker_fixed_window
    
    frame_idx, frame, image_size, output_path, input_path_name = args
    
    try:
        # Use pre-initialized detector and landmarker
        if _worker_detector is None or _worker_landmarker is None:
            print(S('WORKER_NOT_INIT', frame_idx))
            return (frame_idx, 0, {})
        
        # NEW APPROACH: Complete pipeline on resized image, then apply to original
        face_data_list = detect_and_align_on_resized(
            _worker_detector, _worker_landmarker, frame, 
            _worker_fixed_window, image_size
        )
        
        if not face_data_list:
            return (frame_idx, 0, {})
        
        # Align and save each face using original resolution image
        saved_count = 0
        metadata_dict = {}
        
        for face_idx, face_data in enumerate(face_data_list):
            try:
                # Apply alignment to ORIGINAL image
                aligned_face, aligned_landmarks, face_rect_orig = apply_alignment_to_original(
                    frame, face_data
                )
                
                out_size = face_data['out_size']
                
                # Generate filename
                filename = f"{frame_idx:05d}_{face_idx}.jpg"
                filepath = output_path / filename
                
                # Save JPG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                cv2.imwrite(str(filepath), aligned_face, encode_param)
                
                # Store metadata
                metadata_dict[filename] = {
                    'face_type': facelib.FaceType.toString(facelib.FaceType.WHOLE_FACE),
                    'landmarks': aligned_landmarks.tolist(),
                    'source_landmarks': face_data['landmarks'].tolist(),  # In working image coords
                    'source_rect': face_rect_orig,  # In original image coords
                    'image_to_face_mat': get_transform_mat(
                        face_data['landmarks'] * face_data['scale_factor'], 
                        out_size, 
                        facelib.FaceType.WHOLE_FACE
                    ).tolist(),
                    'source_filename': str(input_path_name)
                }
                
                saved_count += 1
            except Exception as e:
                print(S('ALIGN_SAVE_FAILED', f"frame {frame_idx}", face_idx, e))
        
        return (frame_idx, saved_count, metadata_dict)
    except Exception as e:
        print(S('PROCESS_IMAGE_FAILED', f"frame {frame_idx}", e))
        import traceback
        traceback.print_exc()
        return (frame_idx, 0, {})


def visualize_extraction_stages(original_image: np.ndarray, face_data_list: List[Dict], 
                               debug_dir: Path, img_idx: int, fixed_window: int = 0):
    """
    Visualize all stages of face extraction for debugging
    
    Args:
        original_image: Original input image
        face_data_list: List of face data from detect_and_align_on_resized
        debug_dir: Directory to save debug images
        img_idx: Image index
        fixed_window: Pre-resize width used
    """
    import os
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    h_orig, w_orig = original_image.shape[:2]
    
    # Stage 1: Show pre-resized image with detection boxes
    if fixed_window > 0 and w_orig > fixed_window:
        scale_factor = w_orig / fixed_window
        new_h = int(h_orig / scale_factor)
        resized_img = cv2.resize(original_image, (fixed_window, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_img = original_image.copy()
        scale_factor = 1.0
    
    # Draw detection boxes on resized image
    vis_resized = resized_img.copy()
    for face_data in face_data_list:
        l, t, r, b = face_data['face_rect']
        angle = face_data.get('detection_angle', 0)
        
        # Color based on detection angle
        color_map = {0: (0, 255, 0), 90: (255, 0, 0), 180: (0, 0, 255), 270: (255, 255, 0)}
        color = color_map.get(angle, (255, 255, 255))
        
        cv2.rectangle(vis_resized, (l, t), (r, b), color, 2)
        cv2.putText(vis_resized, f'{angle}°', (l, t-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    stage1_path = debug_dir / f"{img_idx:05d}_stage1_detection.png"
    cv2.imwrite(str(stage1_path), vis_resized)
    print(f"[DEBUG] Saved stage 1 (detection): {stage1_path}")
    
    # Stage 2: Show landmarks on resized image
    vis_landmarks = resized_img.copy()
    for face_data in face_data_list:
        landmarks = face_data['landmarks']
        l, t, r, b = face_data['face_rect']
        angle = face_data.get('detection_angle', 0)
        
        color_map = {0: (0, 255, 0), 90: (255, 0, 0), 180: (0, 0, 255), 270: (255, 255, 0)}
        color = color_map.get(angle, (255, 255, 255))
        
        # Draw landmarks
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis_landmarks, (int(x), int(y)), 2, color, -1)
            if i % 10 == 0:  # Label every 10th point
                cv2.putText(vis_landmarks, str(i), (int(x)+3, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw face box
        cv2.rectangle(vis_landmarks, (l, t), (r, b), color, 2)
    
    stage2_path = debug_dir / f"{img_idx:05d}_stage2_landmarks.png"
    cv2.imwrite(str(stage2_path), vis_landmarks)
    print(f"[DEBUG] Saved stage 2 (landmarks): {stage2_path}")
    
    # Stage 3: Show aligned faces and transformation on original image
    vis_original = original_image.copy()
    for face_idx, face_data in enumerate(face_data_list):
        angle = face_data.get('detection_angle', 0)
        color_map = {0: (0, 255, 0), 90: (255, 0, 0), 180: (0, 0, 255), 270: (255, 255, 0)}
        color = color_map.get(angle, (255, 255, 255))
        
        # Get source rect in original image coordinates
        scale_factor_fd = face_data['scale_factor']
        l_work, t_work, r_work, b_work = face_data['face_rect']
        l_orig = round(l_work * scale_factor_fd)
        t_orig = round(t_work * scale_factor_fd)
        r_orig = round(r_work * scale_factor_fd)
        b_orig = round(b_work * scale_factor_fd)
        
        # Draw detection box on original image
        cv2.rectangle(vis_original, (l_orig, t_orig), (r_orig, b_orig), color, 3)
        cv2.putText(vis_original, f'Face {face_idx} ({angle}°)', 
                   (l_orig, t_orig-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw landmarks on original image
        landmarks_work = face_data['landmarks']
        landmarks_orig = landmarks_work.copy()
        landmarks_orig[:, 0] *= scale_factor_fd
        landmarks_orig[:, 1] *= scale_factor_fd
        
        for x, y in landmarks_orig:
            cv2.circle(vis_original, (int(x), int(y)), 3, color, -1)
        
        # Draw the extracted region rectangle
        out_size = face_data['out_size']
        mat_orig = get_transform_mat(landmarks_orig, out_size, 
                                    face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE))
        
        # Calculate corners of the extracted region
        corners = np.array([[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]], dtype=np.float32)
        inv_mat = cv2.invertAffineTransform(mat_orig)
        orig_corners = cv2.transform(corners.reshape(1, -1, 2), inv_mat).reshape(-1, 2)
        
        # Draw polygon showing extraction area
        pts = orig_corners.astype(np.int32)
        cv2.polylines(vis_original, [pts], True, (255, 255, 255), 2)
        
        # Save individual aligned face
        aligned_face, aligned_landmarks, _ = apply_alignment_to_original(original_image, face_data)
        aligned_path = debug_dir / f"{img_idx:05d}_face_{face_idx}_aligned.png"
        cv2.imwrite(str(aligned_path), aligned_face)
        
        # Draw landmarks on aligned face
        vis_aligned = aligned_face.copy()
        for x, y in aligned_landmarks:
            cv2.circle(vis_aligned, (int(x), int(y)), 2, (0, 0, 255), -1)
        aligned_lm_path = debug_dir / f"{img_idx:05d}_face_{face_idx}_aligned_lm.png"
        cv2.imwrite(str(aligned_lm_path), vis_aligned)
    
    stage3_path = debug_dir / f"{img_idx:05d}_stage3_original_with_boxes.png"
    cv2.imwrite(str(stage3_path), vis_original)
    print(f"[DEBUG] Saved stage 3 (original with boxes): {stage3_path}")
    print(f"[DEBUG] Total faces detected: {len(face_data_list)}")
    print(f"[DEBUG] Debug images saved to: {debug_dir}\n")


def landmark106to68(pt106: np.ndarray) -> np.ndarray:#这个我能确保是包对的
    """
    Convert 106 landmarks to standard 68 landmarks
    """
    if len(pt106) != 106:
        return pt106[:68] if len(pt106) >= 68 else pt106
    
    landmark106to68 = [
        1, 10, 12, 14, 16, 3, 5, 7, 0,  # Chin 9 points
        23, 21, 19, 32, 30, 28, 26, 17,  # Eyebrows 8 points (should be 17 total for chin+brows)
        43, 48, 49, 51, 50,  # Left eyebrow 5 points
        102, 103, 104, 105, 101,  # Right eyebrow 5 points
        72, 73, 74, 86, 78, 79, 80, 85, 84,  # Nose 9 points
        35, 41, 42, 39, 37, 36,  # Left eye 6 points
        89, 95, 96, 93, 91, 90,  # Right eye 6 points
        52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57, 60, 54  # Mouth 20 points
    ]
    
    pts68 = np.array([pt106[i] for i in landmark106to68])
    return pts68


def landmark468to68(pt468: np.ndarray) -> np.ndarray:#不一定对，但是能水平对齐，我没怎么测试过
    """
    Convert MediaPipe 468 landmarks to standard 68 landmarks
    Based on MediaPipe Face Mesh topology
    """
    if len(pt468) < 468:
        return pt468[:68] if len(pt468) >= 68 else pt468
    
    # MediaPipe 468 to standard 68 landmark mapping
    # Indices based on MediaPipe Face Mesh topology
    landmark468to68 = [
        # Chin contour (0-16): 152, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356
        152, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356,
        
        # Left eyebrow (17-21): 46, 53, 52, 65, 55
        46, 53, 52, 65, 55,
        
        # Right eyebrow (22-26): 276, 283, 282, 295, 285
        276, 283, 282, 295, 285,
        
        # Nose bridge (27-30): 168, 6, 197, 195
        168, 6, 197, 195,
        
        # Nose bottom (31-35): 5, 4, 98, 131, 134
        5, 4, 98, 131, 134,
        
        # Left eye (36-41): 33, 160, 158, 133, 153, 144
        33, 160, 158, 133, 153, 144,
        
        # Right eye (42-47): 362, 385, 387, 263, 373, 380
        362, 385, 387, 263, 373, 380,
        
        # Mouth outer (48-59): 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
        
        # Mouth inner (60-67): 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324
    ]
    
    # Take only first 68 indices and extract 2D coordinates (ignore z)
    pts68 = np.array([pt468[i][:2] for i in landmark468to68[:68]])
    return pts68


def check_and_adjust_resize(input_path: Path, fixed_window: int) -> int:
    """
    Smart check: if media width <= resize value, disable pre-resize
    
    Args:
        input_path: Input file or directory path
        fixed_window: User-specified resize value
        
    Returns:
        Adjusted resize value (0 if should be disabled)
    """
    if fixed_window <= 0:
        return 0
    
    try:
        import cv2
        
        # Check if it's an image file
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        if input_path.is_file() and input_path.suffix.lower() in image_extensions:
            img = cv2.imread(str(input_path))
            if img is not None:
                width = img.shape[1]
                if width <= fixed_window:
                    print(S('RESIZE_AUTO_DISABLED', width, fixed_window))
                    return 0
                return fixed_window
        
        # Check if it's a video file
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        if input_path.is_file() and input_path.suffix.lower() in video_extensions:
            cap = cv2.VideoCapture(str(input_path))
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap.release()
                if width <= fixed_window:
                    print(S('RESIZE_AUTO_DISABLED', width, fixed_window))
                    return 0
                return fixed_window
        
        # If it's a directory, check the first file
        if input_path.is_dir():
            # Try to find video files first
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                video_files = list(input_path.glob(f'*{ext}'))
                if video_files:
                    cap = cv2.VideoCapture(str(video_files[0]))
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        cap.release()
                        if width <= fixed_window:
                            print(S('RESIZE_AUTO_DISABLED', width, fixed_window))
                            return 0
                        return fixed_window
            
            # Then try image files
            for ext in image_extensions:
                image_files = list(input_path.glob(f'*{ext}'))
                if image_files:
                    img = cv2.imread(str(image_files[0]))
                    if img is not None:
                        width = img.shape[1]
                        if width <= fixed_window:
                            print(S('RESIZE_AUTO_DISABLED', width, fixed_window))
                            return 0
                        return fixed_window
        
        # Cannot determine, keep original value
        return fixed_window
        
    except Exception as e:
        print(S('RESIZE_CHECK_ERROR', e))
        return fixed_window  # Keep original value on error


def process_single_image(
    img_path: Path,
    img_idx: int,
    detector,
    landmarker,
    output_path: Path,
    image_size_fixed: Optional[int] = None,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    fixed_window: int = 0,
    face_type: str = 'whole_face',
    detection_angles: List[int] = None
) -> int:
    """
    Process single image using new approach: detect+landmark on resized, align on original
    Returns number of faces saved
    
    Args:
        detection_angles: List of angles for multi-angle detection [0, 90, 180, 270]
    """
    try:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            return 0
        
        # NEW APPROACH: Complete pipeline on resized image
        face_data_list = detect_and_align_on_resized(
            detector, landmarker, image, fixed_window, image_size_fixed, face_type, detection_angles
        )
        
        if not face_data_list:
            return 0
        
        # Align and save each face using original resolution image
        saved_count = 0
        for face_idx, face_data in enumerate(face_data_list):
            try:
                # Apply alignment to ORIGINAL image
                aligned_face, aligned_landmarks, face_rect_orig = apply_alignment_to_original(
                    image, face_data
                )
                
                out_size = face_data['out_size']
                
                # Generate filename
                filename = f"{img_idx:05d}_{face_idx}.jpg"
                filepath = output_path / filename
                
                # Save JPG with highest quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                cv2.imwrite(str(filepath), aligned_face, encode_param)
                
                # Store metadata in global cache (旧版本，非线程安全)
                if not hasattr(process_single_image, 'metadata_cache'):
                    process_single_image.metadata_cache = {}
                
                process_single_image.metadata_cache[filename] = {
                    'face_type': facelib.FaceType.toString(face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)),
                    'landmarks': aligned_landmarks.tolist(),
                    'source_landmarks': face_data['landmarks'].tolist(),
                    'source_rect': face_rect_orig,
                    'image_to_face_mat': get_transform_mat(
                        face_data['landmarks'] * face_data['scale_factor'],
                        out_size,
                        face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)
                    ).tolist(),
                    'source_filename': str(img_path.name)
                }
                
                saved_count += 1
            except Exception as e:
                print(S('ALIGN_SAVE_FAILED', img_path.name, face_idx, e))
        
        # Debug visualization if enabled
        if debug and debug_dir and face_data_list:
            visualize_extraction_stages(image, face_data_list, debug_dir, img_idx, fixed_window)
        
        return saved_count
    except Exception as e:
        print(S('PROCESS_IMAGE_FAILED', img_path.name, e))
        import traceback
        traceback.print_exc()
        return 0


def process_single_image_threadsafe(
    img_path: Path,
    img_idx: int,
    detector,
    landmarker,
    output_path: Path,
    image_size_fixed: Optional[int] = None,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
    fixed_window: int = 0,
    face_type: str = 'whole_face',
    detection_angles: List[int] = None,
    metadata_cache: Dict = None,
    metadata_lock = None
) -> int:
    """
    Thread-safe version of process_single_image
    使用共享字典和线程锁保护元数据写入
    """
    try:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            return 0
        
        # NEW APPROACH: Complete pipeline on resized image
        face_data_list = detect_and_align_on_resized(
            detector, landmarker, image, fixed_window, image_size_fixed, face_type, detection_angles
        )
        
        if not face_data_list:
            return 0
        
        # Align and save each face using original resolution image
        saved_count = 0
        local_metadata = {}  # 本地缓存，最后统一加锁写入
        
        for face_idx, face_data in enumerate(face_data_list):
            try:
                # Apply alignment to ORIGINAL image
                aligned_face, aligned_landmarks, face_rect_orig = apply_alignment_to_original(
                    image, face_data
                )
                
                out_size = face_data['out_size']
                
                # Generate filename
                filename = f"{img_idx:05d}_{face_idx}.jpg"
                filepath = output_path / filename
                
                # Save JPG with highest quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                cv2.imwrite(str(filepath), aligned_face, encode_param)
                
                # 存储到本地缓存（避免频繁加锁）
                local_metadata[filename] = {
                    'face_type': facelib.FaceType.toString(face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)),
                    'landmarks': aligned_landmarks.tolist(),
                    'source_landmarks': face_data['landmarks'].tolist(),
                    'source_rect': face_rect_orig,
                    'image_to_face_mat': get_transform_mat(
                        face_data['landmarks'] * face_data['scale_factor'],
                        out_size,
                        face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)
                    ).tolist(),
                    'source_filename': str(img_path.name)
                }
                
                saved_count += 1
            except Exception as e:
                print(S('ALIGN_SAVE_FAILED', img_path.name, face_idx, e))
        
        # 一次性将本地缓存合并到共享缓存（加锁保护）
        if local_metadata and metadata_cache is not None and metadata_lock is not None:
            with metadata_lock:
                metadata_cache.update(local_metadata)
        
        # Debug visualization if enabled
        if debug and debug_dir and face_data_list:
            visualize_extraction_stages(image, face_data_list, debug_dir, img_idx, fixed_window)
        
        return saved_count
    except Exception as e:
        print(S('PROCESS_IMAGE_FAILED', img_path.name, e))
        import traceback
        traceback.print_exc()
        return 0



def process_images(
    input_path: Path,
    output_path: Path,
    detector_name: str,
    landmark_name: str,
    device_info,
    image_size: int = None,  # Changed default to None for dynamic size
    debug: bool = False,
    fixed_window: int = 0,  # Pre-resize parameter
    face_type: str = 'whole_face',  # Face type for extraction
    detection_angles: List[int] = None  # Multi-angle detection
):
    """处理图片文件夹 - 并行处理"""
    print(S('PROCESSING_IMAGES', input_path))
    
    # 创建检测器和标记器
    detector = DetectorFactory.create_detector(detector_name, device_info)
    landmarker = LandmarkFactory.create_landmarker(landmark_name, device_info)
    
    # 获取所有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = sorted([
        f for f in input_path.iterdir() 
        if f.suffix.lower() in image_extensions
    ])
    
    if not image_files:
        print(S('NO_IMAGES'))
        return
    
    if debug:
        image_files = image_files[:1]
        print(f"[DEBUG] Processing only first image: {image_files[0].name}\n")  # Keep DEBUG in English
    else:
        print(S('FOUND_IMAGES', len(image_files)))
        print(S('USING_THREADS', cpu_count()))
        print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create debug directory
    debug_dir = output_path / "debug" if debug else None
    
    if debug:
        total_faces = process_single_image(
            image_files[0], 0, detector, landmarker, output_path,
            image_size,  # Pass image_size directly
            debug=True,
            debug_dir=debug_dir,
            fixed_window=fixed_window,
            face_type=face_type,
            detection_angles=detection_angles
        )
        print(S('DEBUG_COMPLETE', debug_dir))
    else:
        from threading import Lock
        
        max_workers = cpu_count()
        total_faces = 0
        processed_files = 0
        metadata_cache = {}  # 共享元数据缓存
        metadata_lock = Lock()  # 线程锁保护元数据写入
        
        pbar = tqdm.tqdm(total=len(image_files), desc="Processing", unit="img", ascii=True)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    process_single_image_threadsafe,
                    img_path, idx, detector, landmarker, output_path,
                    image_size,  # Always pass image_size, None means dynamic calculation
                    False,  # debug
                    None,   # debug_dir
                    fixed_window,  # Pre-resize parameter
                    face_type,  # Face type for extraction
                    detection_angles,  # Multi-angle detection
                    metadata_cache,  # 共享缓存
                    metadata_lock  # 线程锁
                ): (img_path, idx)
                for idx, img_path in enumerate(image_files)
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_idx):
                img_path, idx = future_to_idx[future]
                try:
                    saved_count = future.result(timeout=60)  # 60秒超时
                    total_faces += saved_count
                    processed_files += 1
                    pbar.update(1)
                    pbar.set_postfix({"extracted": total_faces})
                except concurrent.futures.TimeoutError:
                    print(S('TIMEOUT', img_path.name))
                    processed_files += 1
                    pbar.update(1)
                except Exception as e:
                    print(S('FAILED', img_path.name, e))
                    processed_files += 1
                    pbar.update(1)
        
        pbar.close()
        print(S('COMPLETE', processed_files, total_faces, output_path))
        
        # Save metadata cache to HDF5 file
        if metadata_cache:
            import h5py
            
            metadata_file = output_path / "metadata.h5"
            with h5py.File(metadata_file, 'w') as f:
                # Create groups for each image
                for filename, meta in metadata_cache.items():
                    # Replace invalid characters in filename for HDF5 group names
                    safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                    grp = f.create_group(safe_name)
                    
                    # 存储原始文件名
                    grp.attrs['__original_filename__'] = filename
                    
                    # Store each metadata field
                    for key, value in meta.items():
                        if isinstance(value, (list, np.ndarray)):
                            grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                        elif isinstance(value, (int, float, str)):
                            grp.attrs[key] = value
                        else:
                            grp.attrs[key] = str(value)
            
            print(S('METADATA_SAVED', metadata_file))
            print(S('METADATA_ENTRIES', len(metadata_cache)))


def process_video(
    input_path: Path,
    output_path: Path,
    detector_name: str,
    landmark_name: str,
    device_info,
    image_size: int = None,
    fixed_window: int = 0,  # Pre-resize parameter
    face_type: str = 'whole_face',  # Face type for extraction
    detection_angles: List[int] = None,  # Multi-angle detection
    debug: bool = False  # Debug mode: only process first frame with visualization
):
    """Process video file - single process with GPU acceleration"""
    print(S('PROCESSING_VIDEO', input_path))
    
    # Initialize detector and landmarker once
    detector = DetectorFactory.create_detector(detector_name, device_info)
    landmarker = LandmarkFactory.create_landmarker(landmark_name, device_info)
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(S('CANNOT_OPEN_VIDEO', input_path))
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(S('VIDEO_INFO_FULL', total_frames, fps, width, height))
    print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create debug directory if needed
    debug_dir = output_path / "debug" if debug else None
    
    total_saved = 0
    processed_frames = 0
    all_metadata = {}
    prev_faces = None  # For inter-frame face sorting
    frame_idx = 0
    
    pbar = tqdm.tqdm(total=total_frames, desc="Processing", unit="frame", ascii=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
    
    # 用于计算人脸提取速率
    import time
    start_time = time.time()
    last_update_time = start_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # NEW APPROACH: Complete pipeline on resized image
            face_data_list = detect_and_align_on_resized(
                detector, landmarker, frame, fixed_window, image_size, face_type, detection_angles
            )
            
            # Extract face rects for inter-frame sorting
            faces = [face_data['face_rect'] for face_data in face_data_list]
            
            # Inter-frame face sorting (on working image coordinates)
            if faces and prev_faces is not None:
                sorted_indices = sort_faces_by_distance_for_data(prev_faces, faces)
                face_data_list = [face_data_list[i] for i in sorted_indices]
                faces = [face_data['face_rect'] for face_data in face_data_list]
            prev_faces = faces.copy() if faces else None
            
            faces_in_this_frame = 0
            
            # Align and save each face using original resolution image
            for face_idx, face_data in enumerate(face_data_list):
                try:
                    # Apply alignment to ORIGINAL image
                    aligned_face, aligned_landmarks, face_rect_orig = apply_alignment_to_original(
                        frame, face_data
                    )
                    
                    out_size = face_data['out_size']
                    
                    # Generate filename
                    filename = f"{frame_idx:05d}_{face_idx}.jpg"
                    filepath = output_path / filename
                    
                    # Save JPG
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                    cv2.imwrite(str(filepath), aligned_face, encode_param)
                    
                    # Store metadata
                    all_metadata[filename] = {
                        'face_type': facelib.FaceType.toString(face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)),
                        'landmarks': aligned_landmarks.tolist(),
                        'source_landmarks': face_data['landmarks'].tolist(),
                        'source_rect': face_rect_orig,
                        'image_to_face_mat': get_transform_mat(
                            face_data['landmarks'] * face_data['scale_factor'],
                            out_size,
                            face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)
                        ).tolist(),
                        'source_filename': str(input_path.name)
                    }
                    
                    total_saved += 1
                    faces_in_this_frame += 1
                except Exception as e:
                    print(S('ALIGN_SAVE_FAILED', f"frame {frame_idx}", face_idx, e))
            
            processed_frames += 1
            pbar.update(1)
            
            # Debug mode: visualize first frame and exit
            if debug and face_data_list and debug_dir:
                visualize_extraction_stages(frame, face_data_list, debug_dir, frame_idx, fixed_window)
                print(S('DEBUG_COMPLETE', debug_dir))
                break
            
            # 每秒更新一次速率显示（只显示人脸提取速率）
            current_time = time.time()
            if current_time - last_update_time >= 1.0:
                elapsed = current_time - start_time
                if elapsed > 0:
                    faces_per_sec = total_saved / elapsed
                    pbar.set_postfix({"faces/sec": f"{faces_per_sec:.1f}", "saved": total_saved})
                last_update_time = current_time
            
        except Exception as e:
            print(S('PROCESS_IMAGE_FAILED', f"frame {frame_idx}", e))
            import traceback
            traceback.print_exc()
        
        frame_idx += 1
    
    pbar.close()
    cap.release()
    
    print(S('COMPLETE', processed_frames, total_saved, output_path))
    
    # Save metadata to HDF5
    if all_metadata:
        import h5py
        
        metadata_file = output_path / "metadata.h5"
        with h5py.File(metadata_file, 'w') as f:
            # Create groups for each image
            for filename, meta in all_metadata.items():
                # Replace invalid characters in filename for HDF5 group names
                safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                grp = f.create_group(safe_name)
                
                # 存储原始文件名
                grp.attrs['__original_filename__'] = filename
                
                # Store each metadata field
                for key, value in meta.items():
                    if isinstance(value, (list, np.ndarray)):
                        grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                    elif isinstance(value, (int, float, str)):
                        grp.attrs[key] = value
                    else:
                        grp.attrs[key] = str(value)
        
        print(S('METADATA_SAVED', metadata_file))
        print(S('METADATA_ENTRIES', len(all_metadata)))



def consumer_worker(worker_id, detector_name, landmark_name, device_info, 
                   image_size, output_path, input_path_name, 
                   frame_queue, result_queue):
    """
    Consumer worker process: extract faces from frames
    """
    try:
        # Initialize detector and landmarker once per process
        detector = DetectorFactory.create_detector(detector_name, device_info)
        landmarker = LandmarkFactory.create_landmarker(landmark_name, device_info)
        
        while True:
            # Get frame from queue
            item = frame_queue.get()
            
            if item is None:
                # Stop signal
                result_queue.put(None)
                break
            
            frame_idx, frame = item
            
            try:
                # Face detection
                faces, scale_factor = detect_faces_in_image(detector, frame)
                
                if not faces:
                    result_queue.put((frame_idx, 0, {}))
                    continue
                
                # Landmark extraction
                landmarks_list = extract_landmarks(landmarker, frame, faces)
                
                # Align and save each face
                saved_count = 0
                metadata_dict = {}
                
                for face_idx, (face_rect, landmarks) in enumerate(zip(faces, landmarks_list)):
                    if landmarks is None:
                        continue
                    
                    # Convert landmarks to standard 68 points
                    if len(landmarks) == 106:
                        landmarks_for_align = landmark106to68(landmarks)
                    elif len(landmarks) == 468:
                        landmarks_for_align = landmark468to68(landmarks)
                    elif len(landmarks) > 68:
                        landmarks_for_align = landmarks[:68]
                    else:
                        landmarks_for_align = landmarks
                    
                    # Calculate output size
                    if image_size is not None and image_size > 0:
                        out_size = image_size
                    else:
                        out_size = calculate_face_size(face_rect)
                    
                    try:
                        # Get transformation matrix
                        mat = get_transform_mat(landmarks_for_align, out_size, facelib.FaceType.WHOLE_FACE)
                        
                        # Affine transform
                        aligned_face = cv2.warpAffine(frame, mat, (out_size, out_size), 
                                                    flags=cv2.INTER_LANCZOS4)
                        
                        # Transform landmarks
                        aligned_landmarks = facelib.LandmarksProcessor.transform_points(landmarks_for_align, mat)
                        
                        # Generate filename
                        filename = f"{frame_idx:05d}_{face_idx}.jpg"
                        filepath = output_path / filename
                        
                        # Save JPG
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                        cv2.imwrite(str(filepath), aligned_face, encode_param)
                        
                        # Store metadata
                        metadata_dict[filename] = {
                            'face_type': facelib.FaceType.toString(facelib.FaceType.WHOLE_FACE),
                            'landmarks': aligned_landmarks.tolist(),
                            'source_landmarks': landmarks_for_align.tolist(),
                            'source_rect': face_rect,
                            'image_to_face_mat': mat.tolist(),
                            'source_filename': str(input_path_name)
                        }
                        
                        saved_count += 1
                    except Exception as e:
                        print(S('ALIGN_SAVE_FAILED', f"frame {frame_idx}", face_idx, e))
                
                result_queue.put((frame_idx, saved_count, metadata_dict))
            except Exception as e:
                print(S('FAILED', f"Worker {worker_id} on frame {frame_idx}", e))
                import traceback
                traceback.print_exc()
                result_queue.put((frame_idx, 0, {}))
    
    except Exception as e:
        print(S('INIT_WORKER_FAILED', f"Worker {worker_id}: {e}"))
        import traceback
        traceback.print_exc()


def process_batch_frames(batch_args_list):
    """
    Process a batch of frames in worker process
    Args:
        batch_args_list: list of args tuples for each frame
    Returns:
        list of (frame_idx, saved_count, metadata_dict)
    """
    results = []
    for args in batch_args_list:
        result = process_single_frame(args)
        results.append(result)
    return results


def process_video_directory(
    input_dir: Path,
    output_path: Path,
    detector_name: str,
    landmark_name: str,
    device_info,
    image_size: int = None,
    fixed_window: int = 0,
    face_type: str = 'whole_face',
    detection_angles: List[int] = None,  # Multi-angle detection
    debug: bool = False  # Debug mode: only process first frame of first video
):
    """Process all video files in a directory - batch mode"""
    print(S('PROCESSING_VIDEO_DIR', input_dir))
    
    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ])
    
    if not video_files:
        print(S('NO_VIDEOS_FOUND'))
        return
    
    print(S('FOUND_VIDEOS', len(video_files)))
    print()
    
    # Debug mode: only process first video
    if debug:
        video_files = video_files[:1]
        print(f"[DEBUG] Processing only first video: {video_files[0].name}\n")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create debug directory if needed
    debug_dir = output_path / "debug" if debug else None
    
    total_videos_processed = 0
    total_frames_processed = 0
    total_faces_saved = 0
    all_metadata = {}
    
    # Track used base names to handle duplicates
    used_base_names = {}  # Maps base_name -> count
    
    for video_idx, video_path in enumerate(video_files, 1):
        print(f"\n[{video_idx}/{len(video_files)}] Processing: {video_path.name}")
        
        # Generate unique base name for this video
        base_name = video_path.stem  # filename without extension
        ext_lower = video_path.suffix.lower().lstrip('.')
        
        # Check for duplicate base names
        if base_name in used_base_names:
            # Duplicate found, use format: name_ext_frame_XXXXX_XX.jpg
            unique_base = f"{base_name}_{ext_lower}"
            used_base_names[base_name] += 1
        else:
            # First occurrence, check if extension is needed
            unique_base = base_name
            used_base_names[base_name] = 1
        
        try:
            # Initialize detector and landmarker once per video
            detector = DetectorFactory.create_detector(detector_name, device_info)
            landmarker = LandmarkFactory.create_landmarker(landmark_name, device_info)
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(S('CANNOT_OPEN_VIDEO', video_path))
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(S('VIDEO_INFO_FULL', total_frames, fps, width, height))
            
            faces_in_video = 0
            frames_in_video = 0
            prev_faces = None
            frame_idx = 0
            
            pbar = tqdm.tqdm(total=total_frames, desc=f"Video {video_idx}", unit="frame", ascii=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
            
            import time
            start_time = time.time()
            last_update_time = start_time
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Complete pipeline on resized image
                    face_data_list = detect_and_align_on_resized(
                        detector, landmarker, frame, fixed_window, image_size, face_type, detection_angles
                    )
                    
                    # Extract face rects for inter-frame sorting
                    faces = [face_data['face_rect'] for face_data in face_data_list]
                    
                    # Inter-frame face sorting
                    if faces and prev_faces is not None:
                        sorted_indices = sort_faces_by_distance_for_data(prev_faces, faces)
                        face_data_list = [face_data_list[i] for i in sorted_indices]
                        faces = [face_data['face_rect'] for face_data in face_data_list]
                    prev_faces = faces.copy() if faces else None
                    
                    # Align and save each face
                    for face_idx, face_data in enumerate(face_data_list):
                        try:
                            # Apply alignment to ORIGINAL image
                            aligned_face, aligned_landmarks, face_rect_orig = apply_alignment_to_original(
                                frame, face_data
                            )
                            
                            out_size = face_data['out_size']
                            
                            # Generate filename with unique base
                            filename = f"{unique_base}_frame_{frame_idx:05d}_{face_idx}.jpg"
                            filepath = output_path / filename
                            
                            # Save JPG
                            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                            cv2.imwrite(str(filepath), aligned_face, encode_param)
                            
                            # Store metadata
                            all_metadata[filename] = {
                                'face_type': facelib.FaceType.toString(face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)),
                                'landmarks': aligned_landmarks.tolist(),
                                'source_landmarks': face_data['landmarks'].tolist(),
                                'source_rect': face_rect_orig,
                                'image_to_face_mat': get_transform_mat(
                                    face_data['landmarks'] * face_data['scale_factor'],
                                    out_size,
                                    face_data.get('face_type_enum', facelib.FaceType.WHOLE_FACE)
                                ).tolist(),
                                'source_filename': str(video_path.name)
                            }
                            
                            faces_in_video += 1
                            total_faces_saved += 1
                        except Exception as e:
                            print(S('ALIGN_SAVE_FAILED', f"frame {frame_idx}", face_idx, e))
                    
                    frames_in_video += 1
                    total_frames_processed += 1
                    pbar.update(1)
                    
                    # Debug mode: visualize first frame and exit
                    if debug and face_data_list and debug_dir:
                        visualize_extraction_stages(frame, face_data_list, debug_dir, frame_idx, fixed_window)
                        print(f"[DEBUG] Visualization saved to: {debug_dir}")
                        print(S('DEBUG_COMPLETE', debug_dir))
                        # Close video and return early
                        pbar.close()
                        cap.release()
                        print(f"  [DEBUG] Debug completed for video: {video_path.name}")
                        return  # Exit function after first frame visualization
                    
                    # Update rate display
                    current_time = time.time()
                    if current_time - last_update_time >= 1.0:
                        elapsed = current_time - start_time
                        if elapsed > 0:
                            faces_per_sec = faces_in_video / elapsed
                            pbar.set_postfix({"faces": faces_in_video, "faces/sec": f"{faces_per_sec:.1f}"})
                        last_update_time = current_time
                    
                except Exception as e:
                    print(S('PROCESS_IMAGE_FAILED', f"frame {frame_idx}", e))
                    import traceback
                    traceback.print_exc()
                
                frame_idx += 1
            
            pbar.close()
            cap.release()
            
            print(f"  Video completed: {frames_in_video} frames, {faces_in_video} faces extracted")
            total_videos_processed += 1
            
        except Exception as e:
            print(S('FAILED', video_path.name, e))
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(S('BATCH_COMPLETE_SUMMARY', total_videos_processed, total_frames_processed, total_faces_saved, output_path))
    
    # Save metadata to HDF5
    if all_metadata:
        import h5py
        
        metadata_file = output_path / "metadata.h5"
        with h5py.File(metadata_file, 'w') as f:
            # Create groups for each image
            for filename, meta in all_metadata.items():
                # Replace invalid characters in filename for HDF5 group names
                safe_name = filename.replace('/', '_SLASH_').replace('\\', '_BSLASH_')
                grp = f.create_group(safe_name)
                
                # 存储原始文件名
                grp.attrs['__original_filename__'] = filename
                
                # Store each metadata field
                for key, value in meta.items():
                    if isinstance(value, (list, np.ndarray)):
                        grp.create_dataset(key, data=np.array(value), compression='gzip', compression_opts=4)
                    elif isinstance(value, (int, float, str)):
                        grp.attrs[key] = value
                    else:
                        grp.attrs[key] = str(value)
        
        print(S('METADATA_SAVED', metadata_file))
        print(S('METADATA_ENTRIES', len(all_metadata)))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='DeepFaceLab Torch - 人脸提取器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python Extractor.py --input ".\\workspace\\data_dst.mp4" --output ".\\workspace\\data_dst\\aligned"
  python Extractor.py -i ".\\workspace\\images" -o ".\\workspace\\faces" -d BlazeFace -l insightface-2d106det
  python Extractor.py  # 交互式模式
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='输入路径（视频文件或图片文件夹）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='输出路径（保存对齐后的人脸）'
    )
    
    parser.add_argument(
        '-d', '--FaceDetector',
        type=str,
        choices=list(DetectorFactory.DETECTORS.keys()),
        help=f'人脸检测器: {", ".join(DetectorFactory.DETECTORS.keys())}'
    )
    
    parser.add_argument(
        '-l', '--FaceMarker',
        type=str,
        choices=list(LandmarkFactory.LANDMARKS.keys()),
        help=f'特征点标记器: {", ".join(LandmarkFactory.LANDMARKS.keys())}'
    )
    
    parser.add_argument(
        '-s', '--size',
        type=int,
        default=None,  # None means dynamic calculation based on bbox
        help='Output image size (default: None = dynamic based on face bbox)'
    )
    
    parser.add_argument(
        '-r', '--resize',
        type=int,
        default=0,
        help='Pre-resize input image width before face detection (0 = no resize, improves performance for high-res images)'
    )
    
    parser.add_argument(
        '-t', '--face-type',
        type=str,
        choices=['half_face', 'midfull_face', 'full_face', 'whole_face', 'head'],
        default='whole_face',
        help='Face type for extraction (controls padding): half_face, midfull_face, full_face, whole_face, head'
    )
    
    parser.add_argument(
        '-a', '--angles',
        type=str,
        default='0',
        help='Detection angles in degrees (comma-separated), e.g., "0,90,180,270". Default: "0"'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode: only process first image with debug visualization'
    )
    
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['auto', 'video', 'image'],
        default='auto',
        help='Processing mode: auto (detect from input), video (force video mode), image (force image mode)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    #雷霆大字
    print("""
====================================================================================================================
███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗ ██████╗ ██████╗     ████████╗ ██████╗  ██████╗ ██╗     
██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     
█████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║   ██║   ██║██████╔╝       ██║   ██║   ██║██║   ██║██║     
██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║   ██║   ██║██╔══██╗       ██║   ██║   ██║██║   ██║██║     
███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║       ██║   ╚██████╔╝╚██████╔╝███████╗
╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝                                                                                                          
====================================================================================================================
    """)
    args = parse_args()

    print()
    
    # Check if input and output are provided
    if not args.input or not args.output:
        # Interactive mode for missing paths
        if not args.input:
            input_path_str = input(S('ENTER_INPUT_PATH')).strip()
            input_path = Path(input_path_str)
        else:
            input_path = Path(args.input)
        
        if not args.output:
            output_path_str = input(S('ENTER_OUTPUT_PATH')).strip()
            output_path = Path(output_path_str)
        else:
            output_path = Path(args.output)
        
        if not input_path.exists():
            print(S('PATH_NOT_EXIST', input_path))
            return
        
        # Ask for detector if not provided
        if not args.FaceDetector:
            print(S('AVAILABLE_DETECTORS'))
            detectors = list(DetectorFactory.DETECTORS.keys())
            for i, det in enumerate(detectors, 1):
                print(f"  {i}. {det}")
            
            det_choice = input(S('SELECT_DETECTOR', len(detectors))).strip()
            det_idx = int(det_choice) - 1 if det_choice.isdigit() else 4
            detector_name = detectors[det_idx] if 0 <= det_idx < len(detectors) else 'BlazeFace'
        else:
            detector_name = args.FaceDetector
        
        # Ask for landmarker if not provided
        if not args.FaceMarker:
            print(S('AVAILABLE_LANDMARKERS'))
            landmarks = list(LandmarkFactory.LANDMARKS.keys())
            for i, lm in enumerate(landmarks, 1):
                print(f"  {i}. {lm}")
            
            lm_choice = input(S('SELECT_LANDMARKER', len(landmarks))).strip()
            lm_idx = int(lm_choice) - 1 if lm_choice.isdigit() else 0
            landmark_name = landmarks[lm_idx] if 0 <= lm_idx < len(landmarks) else 'insightface-2d106det'
        else:
            landmark_name = args.FaceMarker
        
        # Ask for pre-resize if not provided
        if args.resize == 0 and not args.input:  # Only ask in fully interactive mode
            resize_input = input(S('ENTER_RESIZE_SIZE', '1920')).strip()
            fixed_window = int(resize_input) if resize_input.isdigit() else 0
        else:
            fixed_window = args.resize  # Pre-resize parameter
        
        # Ask for detection angles if not explicitly provided via command line
        import sys
        if '-a' not in sys.argv and '--angles' not in sys.argv:
            angles_input = input("Enter detection angles (comma-separated, e.g., 0,90,180,270) [default: 0]: ").strip()
            if angles_input:
                try:
                    detection_angles = [int(x.strip()) for x in angles_input.split(',')]
                except:
                    detection_angles = [0]
            else:
                detection_angles = [0]
        else:
            try:
                detection_angles = [int(x.strip()) for x in args.angles.split(',')]
            except:
                detection_angles = [0]
        
        # Quick test mode - 默认禁用
        if hasattr(args, 'quick_test') and args.quick_test:
            quick_test = True
        else:
            quick_test = False  # 默认禁用
        
        # Get face type
        face_type = args.face_type if hasattr(args, 'face_type') else 'whole_face'
        
        image_size = args.size  # None means dynamic
        
        print(S('CONFIGURATION'))
        print(f"  {S('INPUT_PATH')}: {input_path}")
        print(f"  {S('OUTPUT_PATH')}: {output_path}")
        print(f"  {S('DETECTOR')}: {detector_name}")
        print(f"  {S('LANDMARKER')}: {landmark_name}")
        print(f"  {S('OUTPUT_SIZE')}: {S('OUTPUT_SIZE_DYNAMIC') if image_size is None else f'{image_size}x{image_size}'}")
        print(f"  {S('PRE_RESIZE')}: {f'{fixed_window}px' if fixed_window > 0 else S('PRE_RESIZE_DISABLED')}")
        print(f"  Detection Angles: {detection_angles}")
        print(f"  Quick Test: {quick_test}")
        print()
        
        confirm = input(S('CONFIRM_START')).strip().lower()
        # 空输入或 y/yes 都视为确认
        if confirm == '' or confirm == 'y' or confirm == 'yes':
            pass  # 继续执行
        else:
            print(S('CANCELLED'))
            return
    elif not args.FaceDetector or not args.FaceMarker:
        # Paths provided but missing detector/landmarker - interactive selection
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if not input_path.exists():
            print(S('PATH_NOT_EXIST', input_path))
            return
        
        # Ask for detector if not provided
        if not args.FaceDetector:
            print(S('AVAILABLE_DETECTORS'))
            detectors = list(DetectorFactory.DETECTORS.keys())
            for i, det in enumerate(detectors, 1):
                print(f"  {i}. {det}")
            
            det_choice = input(S('SELECT_DETECTOR', len(detectors))).strip()
            det_idx = int(det_choice) - 1 if det_choice.isdigit() else 4
            detector_name = detectors[det_idx] if 0 <= det_idx < len(detectors) else 'BlazeFace'
        else:
            detector_name = args.FaceDetector
        
        # Ask for landmarker if not provided
        if not args.FaceMarker:
            print(S('AVAILABLE_LANDMARKERS'))
            landmarks = list(LandmarkFactory.LANDMARKS.keys())
            for i, lm in enumerate(landmarks, 1):
                print(f"  {i}. {lm}")
            
            lm_choice = input(S('SELECT_LANDMARKER', len(landmarks))).strip()
            lm_idx = int(lm_choice) - 1 if lm_choice.isdigit() else 0
            landmark_name = landmarks[lm_idx] if 0 <= lm_idx < len(landmarks) else 'insightface-2d106det'
        else:
            landmark_name = args.FaceMarker
        
        # Ask for pre-resize if not provided
        if args.resize == 0 and not args.FaceDetector:  # Only ask when detector was also selected interactively
            resize_input = input(S('ENTER_RESIZE_SIZE', '1920')).strip()
            fixed_window = int(resize_input) if resize_input.isdigit() else 0
        else:
            fixed_window = args.resize  # Pre-resize parameter
        
        # Ask for detection angles if not explicitly provided via command line
        import sys
        if '-a' not in sys.argv and '--angles' not in sys.argv:
            angles_input = input("Enter detection angles (comma-separated, e.g., 0,90,180,270) [default: 0]: ").strip()
            if angles_input:
                try:
                    detection_angles = [int(x.strip()) for x in angles_input.split(',')]
                except:
                    detection_angles = [0]
            else:
                detection_angles = [0]
        else:
            try:
                detection_angles = [int(x.strip()) for x in args.angles.split(',')]
            except:
                detection_angles = [0]
        
        # Quick test mode - 默认禁用
        if hasattr(args, 'quick_test') and args.quick_test:
            quick_test = True
        else:
            quick_test = False  # 默认禁用
        
        # Get face type
        face_type = args.face_type if hasattr(args, 'face_type') else 'whole_face'
        
        image_size = args.size
        
        print(S('CONFIGURATION'))
        print(f"  {S('INPUT_PATH')}: {input_path}")
        print(f"  {S('OUTPUT_PATH')}: {output_path}")
        print(f"  {S('DETECTOR')}: {detector_name}")
        print(f"  {S('LANDMARKER')}: {landmark_name}")
        print(f"  {S('OUTPUT_SIZE')}: {S('OUTPUT_SIZE_DYNAMIC') if image_size is None else f'{image_size}x{image_size}'}")
        print(f"  {S('PRE_RESIZE')}: {f'{fixed_window}px' if fixed_window > 0 else S('PRE_RESIZE_DISABLED')}")
        print(f"  Detection Angles: {detection_angles}")
        print(f"  Quick Test: {quick_test}")
        print()
        
        confirm = input(S('CONFIRM_START')).strip().lower()
        # 空输入或 y/yes 都视为确认
        if confirm == '' or confirm == 'y' or confirm == 'yes':
            pass  # 继续执行
        else:
            print(S('CANCELLED'))
            return
    else:
        # Full command line mode (all params provided)
        input_path = Path(args.input)
        output_path = Path(args.output)
        detector_name = args.FaceDetector
        landmark_name = args.FaceMarker
        image_size = args.size
        fixed_window = args.resize  # Pre-resize parameter
        face_type = args.face_type if hasattr(args, 'face_type') else 'whole_face'
        
        # Parse detection angles - ask if not provided
        if hasattr(args, 'angles') and args.angles:
            try:
                detection_angles = [int(x.strip()) for x in args.angles.split(',')]
            except:
                detection_angles = [0]
        else:
            # Not provided via command line, ask user
            angles_input = input("Enter detection angles (comma-separated, e.g., 0,90,180,270) [default: 0]: ").strip()
            if angles_input:
                try:
                    detection_angles = [int(x.strip()) for x in angles_input.split(',')]
                except:
                    detection_angles = [0]
            else:
                detection_angles = [0]
        
        # Quick test mode - UI模式强制禁用
        # 当所有参数都通过命令行提供时（UI模式），默认禁用快速测试
        import sys
        if hasattr(args, 'quick_test') and args.quick_test:
            # 显式指定了 --quick-test 参数
            quick_test = True
        else:
            # UI调用或命令行未指定：默认禁用
            quick_test = False
        
        print(S('CMD_MODE'))
        print(f"  {S('INPUT_PATH')}: {input_path}")
        print(f"  {S('OUTPUT_PATH')}: {output_path}")
        print(f"  {S('DETECTOR')}: {detector_name}")
        print(f"  {S('LANDMARKER')}: {landmark_name}")
        print(f"  {S('OUTPUT_SIZE')}: {S('OUTPUT_SIZE_DYNAMIC') if image_size is None else f'{image_size}x{image_size}'}")
        print(f"  {S('PRE_RESIZE')}: {f'{fixed_window}px' if fixed_window > 0 else S('PRE_RESIZE_DISABLED')}")
        print(f"  Detection Angles: {detection_angles}")
        print(f"  Quick Test: {quick_test}")
        print()
        
        if not input_path.exists():
            print(S('PATH_NOT_EXIST', input_path))
            return
    
    # 初始化设备（按优先级：CUDA > DX12 > CPU）
    print(S('INIT_DEVICE'))
    devices = get_available_devices_info()
    
    device_info = None
    selected_device_name = S('NO_DEVICE')
    
    # 优先级1: CUDA (GPU)
    for device in devices:
        if 'CUDA' in str(device).upper() or 'cuda' in str(device).lower():
            device_info = device
            selected_device_name = str(device)
            print(S('DEVICE_CUDA', selected_device_name))
            break
    
    # 优先级2: DirectML/DX12 (GPU)
    if device_info is None:
        for device in devices:
            if 'DML' in str(device).upper() or 'directml' in str(device).lower() or 'dx12' in str(device).lower():
                device_info = device
                selected_device_name = str(device)
                print(S('DEVICE_DIRECTML', selected_device_name))
                break
    
    # 优先级3: CPU
    if device_info is None:
        device_info = get_cpu_device_info()
        selected_device_name = str(device_info)
        print(S('DEVICE_CPU', selected_device_name))
    
    print()
    
    # Smart check and adjust resize parameter
    if fixed_window > 0:
        fixed_window = check_and_adjust_resize(input_path, fixed_window)
    
    # Process input
    try:
        # Determine processing mode
        processing_mode = args.mode if hasattr(args, 'mode') else 'auto'
        
        if input_path.is_file():
            # Single video file
            process_video(input_path, output_path, detector_name, landmark_name, device_info, image_size, fixed_window, face_type, detection_angles, debug=quick_test)
        elif input_path.is_dir():
            # Directory - check mode
            if processing_mode == 'video':
                # Force video directory mode
                process_video_directory(input_path, output_path, detector_name, landmark_name, device_info, image_size, fixed_window, face_type, detection_angles, debug=quick_test)
            elif processing_mode == 'image':
                # Force image mode
                process_images(input_path, output_path, detector_name, landmark_name, device_info, 
                             image_size, debug=quick_test, fixed_window=fixed_window, 
                             face_type=face_type, detection_angles=detection_angles)
            else:
                # Auto mode - detect content type
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
                
                video_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
                image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
                
                if video_files and not image_files:
                    # Only videos found
                    print(S('AUTO_DETECT_VIDEO_MODE'))
                    process_video_directory(input_path, output_path, detector_name, landmark_name, device_info, image_size, fixed_window, face_type, detection_angles, debug=quick_test)
                elif image_files and not video_files:
                    # Only images found
                    print(S('AUTO_DETECT_IMAGE_MODE'))
                    process_images(input_path, output_path, detector_name, landmark_name, device_info, 
                                 image_size, debug=quick_test, fixed_window=fixed_window, 
                                 face_type=face_type, detection_angles=detection_angles)
                elif video_files and image_files:
                    # Both found - ask user
                    print(S('MIXED_CONTENT_DETECTED', len(video_files), len(image_files)))
                    choice = input(S('SELECT_PROCESSING_MODE')).strip().lower()
                    if choice == 'v' or choice == 'video':
                        process_video_directory(input_path, output_path, detector_name, landmark_name, device_info, image_size, fixed_window, face_type, detection_angles, debug=quick_test)
                    else:
                        process_images(input_path, output_path, detector_name, landmark_name, device_info, 
                                     image_size, debug=quick_test, fixed_window=fixed_window, 
                                     face_type=face_type, detection_angles=detection_angles)
                else:
                    # No supported files
                    print(S('NO_SUPPORTED_FILES'))
        else:
            print(S('INVALID_PATH_TYPE', input_path))
    except Exception as e:
        print(S('PROCESSING_ERROR'))
        traceback.print_exc()


if __name__ == '__main__':
    main()
