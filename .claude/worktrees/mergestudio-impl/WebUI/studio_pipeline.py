"""
Single-frame face swap pipeline for Merger Studio preview.

Provides on-demand DFM inference via ONNX Runtime and lightweight
alpha-blend compositing (no full MergeMasked pipeline).
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from facelib import FaceDetector, LandmarksProcessor, FaceType
from merger.MergeMasked import MergeMasked
from merger.MergerConfig import MergerConfigMasked

# ---------------------------------------------------------------------------
#  Global session cache
# ---------------------------------------------------------------------------
_sessions: Dict[str, ort.InferenceSession] = {}
_input_shape_cache: Dict[str, Tuple[int, int]] = {}
_last_inference: dict = {}
_xseg_session: ort.InferenceSession = None  # XSegLite ONNX session  # dfm_path -> (H, W)


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------
def _make_ort_providers(device: str, verbose: bool = False) -> List[str]:
    """Build the ONNX Runtime provider list from a device string.

    Supported formats
    -----------------
    ``"cpu"``           → CPU only.
    ``"0"``, ``"1"``, ...  → CUDAExecutionProvider with the given device id.
    ``"cuda"``          → CUDAExecutionProvider (default device 0).
    ``"dml"``           → DmlExecutionProvider.
    """
    device = (device or "0").strip().lower()

    if device == "cpu":
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    cuda_avail = "CUDAExecutionProvider" in available
    dml_avail = "DmlExecutionProvider" in available

    use_gpu = device not in ("cpu", "")
    if not use_gpu:
        return ["CPUExecutionProvider"]

    # Try to parse device index
    try:
        gpu_idx = int(device)
    except ValueError:
        gpu_idx = 0

    if device.startswith("dml"):
        if dml_avail:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if verbose:
            print("[studio_pipeline] DML not available, falling back to CPU")
        return ["CPUExecutionProvider"]

    if cuda_avail:
        return [
            ("CUDAExecutionProvider", {"device_id": gpu_idx}),
            "CPUExecutionProvider",
        ]
    if dml_avail:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    if verbose:
        print("[studio_pipeline] No GPU provider available, falling back to CPU")
    return ["CPUExecutionProvider"]


def _session_input_size(sess: ort.InferenceSession) -> Tuple[int, int]:
    """Return (H, W) of the first input, assuming NCHW or NHWC."""
    inp = sess.get_inputs()[0]
    shape = inp.shape  # e.g. [1, 3, H, W]  or  [1, H, W, 3]
    if len(shape) == 4:
        if shape[1] in (1, 3):
            # NCHW
            return int(shape[2]), int(shape[3])
        else:
            # NHWC
            return int(shape[1]), int(shape[2])
    # Fallback: assume square
    side = int(np.sqrt(np.prod(shape) // 3)) if shape else 256
    return side, side


def _img_to_jpeg_base64(img_bgr: np.ndarray, quality: int = 90) -> str:
    """Encode BGR image → base64-encoded JPEG string. Handles float32 [0,1] or uint8."""
    # Ensure uint8 range
    if img_bgr.dtype == np.float32 or img_bgr.dtype == np.float64:
        img_bgr = (np.clip(img_bgr, 0.0, 1.0) * 255).astype(np.uint8)
    elif img_bgr.dtype != np.uint8:
        img_bgr = img_bgr.astype(np.uint8)
    success, buf = cv2.imencode(".jpg", img_bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return ""
    return base64.b64encode(buf).decode("ascii")


# ---------------------------------------------------------------------------
#  1.  Model loading
# ---------------------------------------------------------------------------
def load_dfm(dfm_path: str, device: str = "0") -> dict:
    """Load an ONNX DFM model and cache the session.

    Parameters
    ----------
    dfm_path : str
        Path to the ``.dfm`` (ONNX) file.
    device : str
        Device string — ``"cpu"``, ``"0"`` (GPU index), etc.

    Returns
    -------
    dict
        ``path``, ``resolution`` (input side length),
        ``input_shape`` (full shape tuple), ``provider``,
        ``loaded`` (bool).
    """
    p = Path(dfm_path)
    if not p.exists():
        raise FileNotFoundError(f"DFM model not found: {dfm_path}")

    providers = _make_ort_providers(device, verbose=True)

    # Allow SessionOptions for graph optimisation
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(str(p), opts, providers=providers)

    inp = sess.get_inputs()[0]
    input_shape = tuple(inp.shape)

    out_shapes = [tuple(o.shape) for o in sess.get_outputs()]

    h, w = _session_input_size(sess)

    info = {
        "path": str(p.resolve()),
        "resolution": h,  # input side length (square)
        "input_shape": input_shape,
        "output_shapes": out_shapes,
        "provider": sess.get_providers()[0],
        "loaded": True,
    }

    _sessions[dfm_path] = sess
    _input_shape_cache[dfm_path] = (h, w)

    return info


# ---------------------------------------------------------------------------
#  2.  Session query / cleanup
# ---------------------------------------------------------------------------
def is_loaded(dfm_path: str) -> bool:
    """Return True if *dfm_path* has an active session."""
    return dfm_path in _sessions


def unload_all() -> None:
    """Drop all cached ONNX sessions."""
    _sessions.clear()
    _input_shape_cache.clear()


# ---------------------------------------------------------------------------
#  3.  Face detection
# ---------------------------------------------------------------------------
def detect_faces(image: np.ndarray,
                 detector: str = "s3fd",
                 max_faces: int = 1) -> List[dict]:
    """Detect faces and extract 68-point landmarks.

    Parameters
    ----------
    image : np.ndarray
        BGR ``uint8`` frame.
    detector : str
        Detector name — ``"s3fd"``, ``"blazeface"``, etc.
    max_faces : int
        Maximum number of faces to return (sorted by area, largest first).

    Returns
    -------
    list[dict]
        Each item has ``landmarks`` (np.ndarray (68, 2)) and
        ``bbox`` (tuple ``(left, top, right, bottom)``).
    """
    fd = FaceDetector(detector_name=detector)
    return fd.detect(image, max_faces=max_faces)


# ---------------------------------------------------------------------------
#  4.  Single-face swap  (simple alpha-blend compositing)
# ---------------------------------------------------------------------------



def _real_xseg(bgr):
    '''Real XSeg mask extract using XSegLite ONNX model.'''
    _load_xseglite()
    return _xseg_extract(bgr)

def _noop_xseg(bgr):
    """No-op XSeg: returns ones mask (no XSeg clipping)."""
    return np.ones((bgr.shape[0], bgr.shape[1], 1), dtype=np.float32)

def _noop_enhancer(bgr, is_tanh=False, preserve_size=False):
    """No-op face enhancer. Returns input unchanged or upscaled 4x if preserve_size=False."""
    if not preserve_size:
        import cv2 as _cv2
        h, w = bgr.shape[:2]
        return _cv2.resize(bgr, (w * 4, h * 4), interpolation=_cv2.INTER_CUBIC)
    return bgr

def _load_xseglite():
    global _xseg_session
    if _xseg_session is not None:
        return
    xseglite_path = Path(__file__).parent.parent / 'workspace' / 'model' / 'XSegLite' / 'XSegLite_256.onnx'
    if xseglite_path.exists():
        _xseg_session = ort.InferenceSession(str(xseglite_path), providers=['CPUExecutionProvider'])
        print(f"[xseg] XSegLite loaded from {xseglite_path}")
    else:
        print(f"[xseg] XSegLite model not found at {xseglite_path}")

def _xseg_extract(face_bgr):
    '''Extract XSeg mask from a face crop (BGR uint8, any size). Returns mask (H,W,1) float32 [0,1].'''
    if _xseg_session is None:
        return np.ones((face_bgr.shape[0], face_bgr.shape[1], 1), dtype=np.float32)
    import cv2 as _cv2
    inp = _cv2.resize(face_bgr, (256, 256), interpolation=_cv2.INTER_CUBIC)
    inp = inp.astype(np.float32)[np.newaxis, ...]  # NHWC
    in_name = _xseg_session.get_inputs()[0].name
    out = _xseg_session.run(None, {in_name: inp})[0]
    mask = np.clip(out, 0.0, 1.0).squeeze(0)  # NHWC->HWC (256,256,1)
    if mask.ndim == 2: mask = mask[..., None]
    mask[mask < 0.1] = 0.0
    mask = _cv2.resize(mask, (face_bgr.shape[1], face_bgr.shape[0]), interpolation=_cv2.INTER_CUBIC)
    if mask.ndim == 2: mask = mask[..., None]
    mask = np.clip(mask, 0.0, 1.0)
    return mask

def swap_face(frame, landmarks, dfm_path, settings=None):
    """Swap a single face using MergeMaskedFace for full synthesis support."""
    sess = _sessions.get(dfm_path)
    if sess is None:
        raise RuntimeError(f"Model {dfm_path} is not loaded. Call load_dfm() first.")

    input_h, input_w = _input_shape_cache.get(dfm_path, (256, 256))
    predictor_input_shape = (input_h, input_w, 3)

    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    is_nchw = len(inp_shape) == 4 and inp_shape[1] in (1, 3)

    def predictor_func(face_bgr):
        """ONNX predictor, caches in _last_inference for recomposite."""
        if is_nchw:
            inp = face_bgr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        else:
            inp = face_bgr[np.newaxis, ...].astype(np.float32)
        outs = sess.run(None, {inp_name: inp})
        def _h(arr):
            arr = np.clip(arr, 0.0, 1.0)
            if arr.ndim == 4: arr = arr.squeeze(0)
            if arr.ndim == 3 and arr.shape[0] in (1, 3): arr = arr.transpose(1, 2, 0)
            if arr.ndim == 2: arr = arr[..., None]
            return arr
        bgr = _h(outs[1]) if len(outs) > 1 else _h(outs[0])
        msk = _h(outs[0])
        msk_dst = _h(outs[2]) if len(outs) > 2 else msk.copy()
        _last_inference[dfm_path] = (bgr, msk, msk_dst)
        print(f"[swap_face] ONNX infer OK: bgr={bgr.shape} mask={msk.shape}")
        return bgr, msk, msk_dst

    from merger.MergerConfig import MergerConfigMasked, ctm_str_dict
    from facelib import FaceType
    from merger.MergeMasked import MergeMaskedFace
    from merger.FrameInfo import FrameInfo

    face_type_str = (settings or {}).get("face_type", "full_face")
    ft_map = {"half_face": FaceType.HALF, "mid_full": FaceType.MID_FULL, "full_face": FaceType.FULL,
              "whole_face": FaceType.WHOLE_FACE, "head": FaceType.HEAD}
    cfg = MergerConfigMasked(face_type=ft_map.get(face_type_str, FaceType.FULL))
    if settings:
        cfg.mode = settings.get("mode", "overlay")
        cfg.masked_hist_match = settings.get("masked_hist_match", True)
        cfg.mask_mode = int(settings.get("mask_mode", 2))
        cfg.erode_mask_modifier = int(settings.get("erode_mask_modifier", 0))
        cfg.blur_mask_modifier = int(settings.get("blur_mask_modifier", 0))
        cfg.motion_blur_power = int(settings.get("motion_blur_power", 0))
        ct = settings.get("color_transfer_mode", "rct")
        cfg.color_transfer_mode = ctm_str_dict.get(ct, ctm_str_dict["rct"])
        cfg.output_face_scale = int(settings.get("output_face_scale", 0))
        cfg.super_resolution_power = int(settings.get("super_resolution_power", 0))
        cfg.image_denoise_power = int(settings.get("image_denoise_power", 0))
        cfg.bicubic_degrade_power = int(settings.get("bicubic_degrade_power", 0))
        cfg.color_degrade_power = int(settings.get("color_degrade_power", 0))
        cfg.sharpen_mode = int(settings.get("sharpen_mode", 0))
        cfg.blursharpen_amount = int(settings.get("blursharpen_amount", 0))

    frame_f32 = frame.astype(np.float32) / 255.0
    frame_info = FrameInfo(landmarks_list=[landmarks])

    try:
        mmf_result = MergeMaskedFace(
            predictor_func, predictor_input_shape,
            _noop_enhancer, _real_xseg if cfg.mask_mode >= 6 else _noop_xseg, cfg, frame_info,
            frame, frame_f32, landmarks)
        # MergeMaskedFace returns (img, mask) tuple; take just the image
        if isinstance(mmf_result, tuple):
            mmf_result = mmf_result[0]
        return mmf_result
    except Exception as e:
        import traceback
        print(f"[swap_face] MergeMaskedFace failed: {e}")
        traceback.print_exc()
        # Fallback: simple alpha blend
        output_size = input_h
        face_mat = LandmarksProcessor.get_transform_mat(landmarks, output_size, FaceType.FULL)
        dst_face = cv2.warpAffine(frame_f32, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)
        dst_face = np.clip(dst_face, 0.0, 1.0)
        def _h(arr):
            arr = np.clip(arr, 0.0, 1.0)
            if arr.ndim == 4: arr = arr.squeeze(0)
            if arr.ndim == 3 and arr.shape[0] in (1, 3): arr = arr.transpose(1, 2, 0)
            if arr.ndim == 2: arr = arr[..., None]
            return arr
        if is_nchw:
            inp = dst_face.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        else:
            inp = dst_face[np.newaxis, ...].astype(np.float32)
        outs = sess.run(None, {inp_name: inp})
        prd_bgr = _h(outs[0])
        prd_mask = _h(outs[1])
        if prd_bgr.shape[2] != 3: prd_bgr = np.stack([prd_bgr[..., 0]] * 3, axis=-1)
        inv_mat = cv2.invertAffineTransform(face_mat)
        h, w = frame.shape[:2]
        bf = cv2.warpAffine(prd_bgr, inv_mat, (w, h), flags=cv2.INTER_CUBIC)
        bm = cv2.warpAffine(prd_mask, inv_mat, (w, h), flags=cv2.INTER_CUBIC)
        bm = np.clip(bm, 0.0, 1.0)
        res = frame_f32 * (1.0 - bm) + bf * bm
        return (np.clip(res, 0.0, 1.0) * 255).astype(np.uint8)


def recomposite_face(frame, landmarks, dfm_path, settings, cached_bgr, cached_mask, cached_dst_mask):
    """Re-composite using cached model outputs (no ONNX inference)."""
    from merger.MergeMasked import MergeMaskedFace
    from merger.MergerConfig import MergerConfigMasked, ctm_str_dict
    from merger.FrameInfo import FrameInfo
    from facelib import FaceType, LandmarksProcessor
    ft_map = {"half_face": FaceType.HALF, "mid_full": FaceType.MID_FULL, "full_face": FaceType.FULL,
              "whole_face": FaceType.WHOLE_FACE, "head": FaceType.HEAD}
    cfg = MergerConfigMasked(face_type=ft_map.get(settings.get("face_type", "full_face"), FaceType.FULL))
    cfg.mode = settings.get("mode", "overlay")
    cfg.masked_hist_match = settings.get("masked_hist_match", True)
    cfg.mask_mode = int(settings.get("mask_mode", 4))
    cfg.erode_mask_modifier = int(settings.get("erode_mask_modifier", 0))
    cfg.blur_mask_modifier = int(settings.get("blur_mask_modifier", 0))
    cfg.motion_blur_power = int(settings.get("motion_blur_power", 0))
    ct = settings.get("color_transfer_mode", "rct")
    cfg.color_transfer_mode = ctm_str_dict.get(ct, ctm_str_dict["rct"])
    cfg.output_face_scale = int(settings.get("output_face_scale", 0))
    cfg.super_resolution_power = int(settings.get("super_resolution_power", 0))
    cfg.image_denoise_power = int(settings.get("image_denoise_power", 0))
    cfg.bicubic_degrade_power = int(settings.get("bicubic_degrade_power", 0))
    cfg.color_degrade_power = int(settings.get("color_degrade_power", 0))
    cfg.sharpen_mode = int(settings.get("sharpen_mode", 0))
    cfg.blursharpen_amount = int(settings.get("blursharpen_amount", 0))
    frame_f32 = frame.astype(np.float32) / 255.0
    fi = FrameInfo(landmarks_list=[landmarks])
    def pred(face_bgr):
        return cached_bgr, cached_mask, cached_dst_mask
    try:
        r = MergeMaskedFace(pred, (cached_bgr.shape[1], cached_bgr.shape[0], 3),
                           _noop_enhancer, _real_xseg if cfg.mask_mode >= 6 else _noop_xseg, cfg, fi, frame, frame_f32, landmarks)
        if isinstance(r, tuple): r = r[0]
        return r
    except Exception as e:
        import traceback
        print(f"[recomposite] MergeMaskedFace failed: {e}")
        traceback.print_exc()
        return frame
def analyze_frame(frame: np.ndarray,
                  dfm_path: str,
                  settings: dict) -> dict:
    """Run the complete preview pipeline on a single frame.

    Pipeline
    --------
    1. Scale the frame down if ``settings["preview_scale"] > 1``.
    2. Detect faces (using detector from settings).
    3. Draw landmarks + bounding box on a copy for the **detection** view.
    4. Run ``swap_face`` for the **swapped** view.
    5. Encode all views as base64 JPEG.

    Parameters
    ----------
    frame : np.ndarray
        BGR ``uint8`` image.
    dfm_path : str
        Loaded DFM model path.
    settings : dict
        Merged user settings (as returned by ``studio_settings.load()``).
        Relevant keys:
            ``preview_scale``, ``detector``, ``max_faces``.

    Returns
    -------
    dict
        ``original``  — base64 JPEG of the (possibly scaled) original.
        ``detection`` — base64 JPEG with face landmarks + bbox drawn.
        ``swapped``   — base64 JPEG after face swap.
        ``has_face``  — ``bool``.
        ``faces``     — list of detection dicts (landmarks/bbox).
    """
    # ---- 5a. Scale ----
    scale = int(settings.get("preview_scale", 2))
    h, w = frame.shape[:2]
    if scale > 1:
        new_w = w // scale
        new_h = h // scale
        working = cv2.resize(frame, (new_w, new_h), cv2.INTER_AREA)
    else:
        working = frame.copy()

    # ---- 5b. Detect faces ----
    detector_name = settings.get("detector", "s3fd")
    max_faces = int(settings.get("max_faces", 1))

    faces = detect_faces(working, detector=detector_name, max_faces=max_faces)

    # ---- 5c. Draw detection overlay ----
    detection_img = working.copy()
    has_face = len(faces) > 0
    for face_info in faces:
        bbox = face_info["bbox"]  # (l, t, r, b)
        lmks = face_info["landmarks"]

        # Bounding box
        cv2.rectangle(detection_img, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Landmarks
        for x, y in lmks.astype(np.int32):
            cv2.circle(detection_img, (x, y), 2, (0, 0, 255), -1)

        # Jaw line for visual reference
        jaw = lmks[0:17].astype(np.int32)
        for i in range(len(jaw) - 1):
            cv2.line(detection_img, tuple(jaw[i]), tuple(jaw[i + 1]),
                     (255, 255, 0), 1)

    # ---- 5d. Swap face ----
    swapped = working.copy()
    if has_face:
        if is_loaded(dfm_path):
            try:
                first_face = faces[0]
                # Check if we can skip inference (recomposite mode)
                skip_infer = settings.get("_skip_inference", False) if settings else False
                if skip_infer and dfm_path in _last_inference:
                    _cache = _last_inference[dfm_path]
                    swapped = recomposite_face(working, first_face["landmarks"], dfm_path, settings,
                                              _cache[0], _cache[1], _cache[2])
                else:
                    swapped = swap_face(working, first_face["landmarks"], dfm_path, settings)

            except Exception as e:
                print(f"[studio_pipeline] swap_face failed: {e}")
                # swapped stays as original on failure
        # else: no DFM model loaded — swapped stays as original, warning displayed in JS

    # ---- 5e. Encode ----
    original_b64 = _img_to_jpeg_base64(working)
    detection_b64 = _img_to_jpeg_base64(detection_img)
    swapped_b64 = _img_to_jpeg_base64(swapped)

    return {
        "original": original_b64,
        "detection": detection_b64,
        "swapped": swapped_b64,
        "has_face": has_face,
        "model_loaded": is_loaded(dfm_path),
        "faces": [
            {
                "bbox": list(f["bbox"]),
                "landmarks": f["landmarks"].tolist(),
            }
            for f in faces
        ],
    }
