"""
Complete face detection + landmark + alignment pipeline.
Reference: Extractor/Extractor.py
"""
import cv2
import numpy as np
import facelib
from facelib.LandmarksProcessor import get_transform_mat
from modelhub.onnx.YoloV8Face import YoloV8Face
from MergeStudio.core.detector.landmarks import landmark106to68


def detect_and_align(detector, landmarker, image: np.ndarray,
                     face_type_str: str = 'whole_face',
                     fixed_window: int = 0):
    """
    Complete pipeline: detect faces -> extract landmarks -> align.
    Returns list of face_data dicts.
    """
    h_orig, w_orig = image.shape[:2]
    scale_factor = 1.0
    working_image = image

    if fixed_window > 0 and w_orig > fixed_window:
        scale_factor = w_orig / fixed_window
        new_h = int(h_orig / scale_factor)
        working_image = cv2.resize(image, (fixed_window, new_h),
                                   interpolation=cv2.INTER_AREA)

    if isinstance(detector, YoloV8Face):
        results = detector.extract(working_image)
    else:
        results = detector.extract(working_image, fixed_window=0)

    if not results or len(results) == 0:
        return []

    faces = results[0] if isinstance(results[0], list) else results

    face_data_list = []
    for face_rect in faces:
        l, t, r, b = face_rect[:4]
        l, t, r, b = int(l), int(t), int(r), int(b)

        margin = int((r - l) * 0.2)
        l_crop = max(0, l - margin)
        t_crop = max(0, t - margin)
        r_crop = min(working_image.shape[1], r + margin)
        b_crop = min(working_image.shape[0], b + margin)

        face_img = working_image[t_crop:b_crop, l_crop:r_crop]
        if face_img.size == 0:
            continue

        lmks = None
        try:
            landmark_results = landmarker.extract(face_img)
            if landmark_results is not None and len(landmark_results) > 0:
                pts = landmark_results[0].copy()
                pts[:, 0] += l_crop
                pts[:, 1] += t_crop
                lmks = pts
        except Exception:
            continue

        if lmks is None:
            continue

        if len(lmks) == 106:
            lmks = landmark106to68(lmks)

        lmks_orig = lmks.copy()
        lmks_orig[:, 0] *= scale_factor
        lmks_orig[:, 1] *= scale_factor

        face_type_map = {
            'half_face': facelib.FaceType.HALF,
            'midfull_face': facelib.FaceType.MID_FULL,
            'full_face': facelib.FaceType.FULL,
            'whole_face': facelib.FaceType.WHOLE_FACE,
            'head': facelib.FaceType.HEAD,
        }
        face_type_enum = face_type_map.get(face_type_str, facelib.FaceType.WHOLE_FACE)
        out_size = 256
        mat = get_transform_mat(lmks_orig, out_size, face_type_enum)

        face_data_list.append({
            'face_rect': (int(l * scale_factor), int(t * scale_factor),
                          int(r * scale_factor), int(b * scale_factor)),
            'landmarks': lmks_orig,
            'transform_mat': mat,
            'out_size': out_size,
            'face_type': face_type_str,
            'face_type_enum': face_type_enum,
        })

    return face_data_list


def apply_alignment(image: np.ndarray, face_data: dict) -> np.ndarray:
    """Apply stored transform to original resolution image."""
    mat = get_transform_mat(face_data['landmarks'], face_data['out_size'],
                            face_data['face_type_enum'])
    return cv2.warpAffine(image, mat, (face_data['out_size'], face_data['out_size']),
                          flags=cv2.INTER_LANCZOS4)
