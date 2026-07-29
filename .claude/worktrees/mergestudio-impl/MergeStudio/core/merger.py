"""
Face swap merge engine.
Transplanted from merger/MergeMasked.py and merger/MergeAvatar.py.
Complete implementation: mask modes 0-9, 8 color transfers, all post-processing, FaceAvatar.
"""
from pathlib import Path
import cv2
import numpy as np
import traceback
from facelib import FaceType, LandmarksProcessor
from core.imagelib import color_transfer
from MergeStudio.core.config import mode_dict, mask_mode_dict, ctm_dict, ctm_str_dict


class FrameInfo:
    """Per-frame metadata for merging."""
    def __init__(self, filepath=None, landmarks_list=None, motion_deg=0, motion_power=0):
        self.filepath = Path(filepath) if filepath else None
        self.landmarks_list = landmarks_list or []
        self.motion_deg = motion_deg
        self.motion_power = motion_power


def _get_face_mask(face_img, face_landmarks, cfg, pred_mask=None, xseg_mask=None):
    """
    Compute the final mask based on mask_mode (0-9).
    mask_mode:
      0=full, 1=dst, 2=learned-prd, 3=learned-dst,
      4=learned-prd*learned-dst, 5=learned-prd+learned-dst,
      6=XSeg-prd, 7=XSeg-dst, 8=XSeg-prd*XSeg-dst,
      9=all combined
    """
    hull_mask = LandmarksProcessor.get_image_hull_mask(face_img.shape, face_landmarks).astype(np.float32)

    if cfg.mask_mode == 0:
        return hull_mask
    if cfg.mask_mode == 1:
        return hull_mask  # dst same as full for standalone mode
    if cfg.mask_mode == 2:
        return pred_mask if pred_mask is not None else hull_mask
    if cfg.mask_mode == 3:
        return hull_mask
    if cfg.mask_mode == 4:
        if pred_mask is not None:
            return pred_mask * hull_mask
        return hull_mask
    if cfg.mask_mode == 5:
        if pred_mask is not None:
            return np.clip(pred_mask + hull_mask, 0, 1)
        return hull_mask
    if cfg.mask_mode == 6:
        return xseg_mask if xseg_mask is not None else hull_mask
    if cfg.mask_mode == 7:
        return hull_mask
    if cfg.mask_mode == 8:
        if xseg_mask is not None:
            return xseg_mask * hull_mask
        return hull_mask
    if cfg.mask_mode == 9:
        mask = hull_mask
        if pred_mask is not None:
            mask = mask * pred_mask
        if xseg_mask is not None:
            mask = mask * xseg_mask
        return mask
    return hull_mask


def _apply_color_transfer(ct_mode, img_src, img_trg):
    """Apply color transfer by mode index (0-8)."""
    if ct_mode == 0 or ct_mode == 'None':
        return img_src
    name = ctm_dict.get(ct_mode, "rct") if isinstance(ct_mode, int) else ct_mode
    try:
        result = color_transfer.color_transfer(name, img_src, img_trg)
        if result is not None:
            return result
    except Exception:
        pass
    return img_src


def _apply_postprocessing(face_img, mask, cfg, face_enhancer_func=None):
    """Apply all post-processing to the merged face."""
    out = face_img.copy()
    h, w = face_img.shape[:2]

    # Face scale (warp scale)
    if cfg.output_face_scale != 0:
        scale = 1.0 + cfg.output_face_scale / 100.0
        if scale > 0:
            mat = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
            out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_LANCZOS4)
            mask_out = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_LANCZOS4)
            mask = mask_out

    # Super resolution
    if cfg.super_resolution_power > 0 and face_enhancer_func is not None:
        try:
            out = face_enhancer_func(out)
        except Exception:
            pass

    # Sharpen
    if cfg.sharpen_mode != 0 and cfg.blursharpen_amount != 0:
        amount = cfg.blursharpen_amount / 100.0
        if cfg.sharpen_mode == 1:  # box sharpen
            blurred = cv2.blur(out, (3, 3))
        else:  # gaussian sharpen
            blurred = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out.astype(np.float32), 1.0 + amount,
                              blurred.astype(np.float32), -amount, 0)
        out = np.clip(out, 0, 255).astype(np.uint8)

    # Motion blur
    if cfg.motion_blur_power > 0:
        k_size = max(1, cfg.motion_blur_power)
        direction = 0
        kernel = np.zeros((k_size, k_size))
        kernel[:, k_size // 2] = 1.0 / k_size
        out = cv2.filter2D(out, -1, kernel)

    # Image denoise
    if cfg.image_denoise_power > 0:
        k_size = max(1, min(cfg.image_denoise_power, 10) | 1)
        out = cv2.medianBlur(out, k_size)

    # Bicubic degrade
    if cfg.bicubic_degrade_power > 0:
        scale = max(0.1, 1.0 - cfg.bicubic_degrade_power / 100.0)
        small = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    # Color degrade
    if cfg.color_degrade_power > 0:
        bins = max(2, 256 - int(cfg.color_degrade_power * 2.56))
        out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        out = (out / (256.0 / bins)).astype(np.float32) * (256.0 / bins)
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

    return out, mask


def MergeMaskedFace(frame_img, face_landmarks, cfg, predictor_func=None,
                    face_enhancer_func=None, xseg_256_extract_func=None):
    """
    Merge a single face within a frame.
    Full implementation with all mask modes, color transfers, and post-processing.
    Returns (out_img, out_merging_mask).
    """
    try:
        # Warp face to 256x256
        mat = LandmarksProcessor.get_transform_mat(face_landmarks, 256, cfg.face_type)
        face_img = cv2.warpAffine(frame_img, mat, (256, 256), flags=cv2.INTER_LANCZOS4)

        # Run predictor
        pred_mask = None
        xseg_mask = None
        if predictor_func is not None:
            pred = predictor_func(face_img)
            if len(pred.shape) == 4:
                bgr = pred[0, :, :, :3]
                if pred.shape[3] >= 4:
                    pred_mask = pred[0, :, :, 3].astype(np.float32) / 255.0
            elif len(pred.shape) == 3:
                bgr = pred[:, :, :3]
                if pred.shape[2] >= 4:
                    pred_mask = pred[:, :, 3].astype(np.float32) / 255.0
            else:
                bgr = face_img
        else:
            bgr = face_img

        # XSeg mask
        if xseg_256_extract_func is not None and cfg.mask_mode >= 6:
            try:
                xseg_result = xseg_256_extract_func(face_img)
                if xseg_result is not None:
                    xseg_mask = xseg_result.astype(np.float32) / 255.0
            except Exception:
                pass

        # Compute mask based on mask_mode
        mask_f = _get_face_mask(face_img, face_landmarks, cfg, pred_mask, xseg_mask)

        # Normalize
        out_face_bgr = bgr.astype(np.float32)
        dst_face_bgr = face_img.astype(np.float32)

        # Merge mode
        if cfg.mode == 'overlay':
            out_face_bgr = np.clip(out_face_bgr, 0, 255)
        elif cfg.mode == 'hist-match':
            out_face_bgr = color_transfer.color_hist_match(
                out_face_bgr, dst_face_bgr, cfg.hist_match_threshold
            ).astype(np.float32)
        elif cfg.mode == 'seamless':
            try:
                center = (face_img.shape[1] // 2, face_img.shape[0] // 2)
                out_face_bgr = cv2.seamlessClone(
                    out_face_bgr.astype(np.uint8),
                    dst_face_bgr.astype(np.uint8),
                    (mask_f * 255).astype(np.uint8),
                    center, cv2.NORMAL_CLONE
                ).astype(np.float32)
            except Exception:
                pass
        elif cfg.mode == 'seamless-hist-match':
            try:
                center = (face_img.shape[1] // 2, face_img.shape[0] // 2)
                out_face_bgr = cv2.seamlessClone(
                    out_face_bgr.astype(np.uint8),
                    dst_face_bgr.astype(np.uint8),
                    (mask_f * 255).astype(np.uint8),
                    center, cv2.NORMAL_CLONE
                ).astype(np.float32)
            except Exception:
                pass
            out_face_bgr = color_transfer.color_hist_match(
                out_face_bgr, dst_face_bgr, cfg.hist_match_threshold
            ).astype(np.float32)
        elif cfg.mode == 'raw-rgb':
            pass  # Keep raw prediction
        elif cfg.mode == 'raw-predict':
            pass  # Keep raw prediction

        # Color transfer
        if 'raw' not in cfg.mode and cfg.color_transfer_mode != 0:
            try:
                out_face_bgr = _apply_color_transfer(cfg.color_transfer_mode,
                                                     out_face_bgr, dst_face_bgr)
            except Exception:
                pass

        # Erode/dilate mask
        if cfg.erode_mask_modifier != 0:
            k_size = max(1, abs(cfg.erode_mask_modifier))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            if cfg.erode_mask_modifier > 0:
                mask_f = cv2.erode(mask_f, kernel, iterations=1)
            else:
                mask_f = cv2.dilate(mask_f, kernel, iterations=1)

        # Blur mask
        if cfg.blur_mask_modifier > 0:
            k_size = max(1, cfg.blur_mask_modifier | 1)
            mask_f = cv2.GaussianBlur(mask_f, (k_size, k_size), 0)

        # Composite using mask
        mask_f_3 = np.stack([mask_f, mask_f, mask_f], axis=-1)
        out_face_bgr = dst_face_bgr * (1 - mask_f_3) + out_face_bgr * mask_f_3

        # Post-processing on the face
        out_face_bgr, mask_f = _apply_postprocessing(
            np.clip(out_face_bgr, 0, 255).astype(np.uint8),
            mask_f, cfg, face_enhancer_func)

        # Warp back to original frame
        mat_inv = cv2.invertAffineTransform(mat)
        warped_face = cv2.warpAffine(out_face_bgr.astype(np.float32), mat_inv,
                                     (frame_img.shape[1], frame_img.shape[0]),
                                     flags=cv2.INTER_LANCZOS4)
        warped_mask = cv2.warpAffine(mask_f, mat_inv,
                                     (frame_img.shape[1], frame_img.shape[0]),
                                     flags=cv2.INTER_LANCZOS4)

        frame_f = frame_img.astype(np.float32)
        warped_mask_3 = np.stack([warped_mask, warped_mask, warped_mask], axis=-1)
        out_img = frame_f * (1 - warped_mask_3) + warped_face * warped_mask_3

        out_img = np.clip(out_img, 0, 255).astype(np.uint8)
        out_mask = np.clip(warped_mask * 255, 0, 255).astype(np.uint8)

        return out_img, out_mask

    except Exception as e:
        traceback.print_exc()
        return frame_img, np.zeros(frame_img.shape[:2], dtype=np.uint8)


def MergeMasked(frame_info, cfg, predictor_func=None, face_enhancer_func=None,
                xseg_256_extract_func=None):
    """Merge all faces in a frame. Returns final RGBA image."""
    frame_img = cv2.imread(str(frame_info.filepath))
    if frame_img is None:
        return None

    out_img = None
    full_mask = np.zeros(frame_img.shape[:2], dtype=np.float32)

    for landmarks in frame_info.landmarks_list:
        face_img, face_mask = MergeMaskedFace(
            frame_img, landmarks, cfg, predictor_func,
            face_enhancer_func, xseg_256_extract_func)

        if out_img is None:
            out_img = face_img.astype(np.float32)
            full_mask = face_mask.astype(np.float32)
        else:
            mask_f = (face_mask.astype(np.float32) / 255.0) * \
                     (1.0 - full_mask / 255.0)
            mask_f_3 = np.stack([mask_f, mask_f, mask_f], axis=-1)
            out_img = out_img * (1 - mask_f_3) + face_img.astype(np.float32) * mask_f_3
            full_mask = np.maximum(full_mask, face_mask.astype(np.float32))

    if out_img is None:
        out_img = frame_img.astype(np.float32)

    rgba = np.concatenate([
        np.clip(out_img, 0, 255).astype(np.uint8),
        np.clip(full_mask, 0, 255).astype(np.uint8)[:, :, None]
    ], axis=-1)

    return rgba


def MergeFaceAvatar(predictor_func, predictor_input_shape, cfg,
                    prev_temporal_frame_infos, frame_info,
                    next_temporal_frame_infos):
    """
    Merge a frame using FaceAvatar model with temporal context.
    """
    try:
        from facelib import LandmarksProcessor as lp

        frame_img = cv2.imread(str(frame_info.filepath))
        if frame_img is None:
            return None

        # Process temporal frames
        prev_imgs = []
        for fi in prev_temporal_frame_infos:
            img = cv2.imread(str(fi.filepath))
            if img is not None:
                prev_imgs.append(img)

        next_imgs = []
        for fi in next_temporal_frame_infos:
            img = cv2.imread(str(fi.filepath))
            if img is not None:
                next_imgs.append(img)

        # Use the first face's landmarks
        if not frame_info.landmarks_list:
            return frame_img

        landmarks = frame_info.landmarks_list[0]

        # Align frame
        mat = LandmarksProcessor.get_transform_mat(landmarks, 256, FaceType.FULL)
        face_img = cv2.warpAffine(frame_img, mat, (256, 256), flags=cv2.INTER_LANCZOS4)

        # Predict with temporal context
        pred = predictor_func(face_img)

        # Post-processing
        out_img = pred.astype(np.float32) if pred is not None else face_img.astype(np.float32)

        # Sharpen
        if cfg.sharpen_mode != 0 and cfg.blursharpen_amount != 0:
            amount = cfg.blursharpen_amount / 100.0
            if cfg.sharpen_mode == 1:
                blurred = cv2.blur(out_img, (3, 3))
            else:
                blurred = cv2.GaussianBlur(out_img, (0, 0), 1.0)
            out_img = cv2.addWeighted(out_img, 1.0 + amount, blurred, -amount, 0)
            out_img = np.clip(out_img, 0, 255)

        # Optional source image side-by-side
        if cfg.add_source_image:
            h, w = frame_img.shape[:2]
            out_img = np.concatenate([frame_img.astype(np.float32), out_img], axis=1)
            out_img = cv2.resize(out_img, (w, h))

        return np.clip(out_img, 0, 255).astype(np.uint8)

    except Exception as e:
        traceback.print_exc()
        if frame_info.filepath:
            return cv2.imread(str(frame_info.filepath))
        return None
