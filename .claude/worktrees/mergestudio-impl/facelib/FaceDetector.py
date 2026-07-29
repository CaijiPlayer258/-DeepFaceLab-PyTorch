"""
FaceDetector wrapper — combines a face detection model (S3FD, BlazeFace, etc.)
with a landmark extractor (InsightFace2D106) into a single unified interface.

Import via:  from facelib import FaceDetector
"""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from xlib.onnxruntime import (
    ORTDeviceInfo,
    get_available_devices_info,
    get_cpu_device_info,
)

# ---------------------------------------------------------------------------
# Lazy model imports (heavy — only imported when a detector is first created)
# ---------------------------------------------------------------------------
_DETECTOR_CLASSES = {}
_LANDMARKER_CLASS = None


def _import_detectors():
    global _DETECTOR_CLASSES
    if _DETECTOR_CLASSES:
        return
    from modelhub.onnx import BlazeFace, CenterFace, S3FD, YoloV5Face
    from modelhub.onnx.YoloV8Face import YoloV8Face

    _DETECTOR_CLASSES = {
        "blazeface": BlazeFace,
        "centerface": CenterFace,
        "s3fd": S3FD,
        "yolov5face": YoloV5Face,
        "yolov8face": YoloV8Face,
    }


def _import_landmarker():
    global _LANDMARKER_CLASS
    if _LANDMARKER_CLASS is not None:
        return
    from modelhub.onnx.InsightFace2d106.InsightFace2D106 import InsightFace2D106

    _LANDMARKER_CLASS = InsightFace2D106


# ---------------------------------------------------------------------------
#  106pt -> 68pt conversion  (copied from Extractor/Extractor.py)
# ---------------------------------------------------------------------------
_LANDMARK106_TO_68_IDX = [
    1, 10, 12, 14, 16, 3, 5, 7, 0,  # chin 9
    23, 21, 19, 32, 30, 28, 26, 17,  # brows start   (actually 8 pts)
    43, 48, 49, 51, 50,              # left eyebrow  5
    102, 103, 104, 105, 101,         # right eyebrow 5
    72, 73, 74, 86, 78, 79, 80, 85, 84,  # nose 9
    35, 41, 42, 39, 37, 36,          # left eye      6
    89, 95, 96, 93, 91, 90,          # right eye     6
    52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55, 65, 66, 62, 70, 69, 57,
    60, 54,  # mouth 20
]


def _landmark106to68(pts106: np.ndarray) -> np.ndarray:
    """Convert (106,2) → (68,2) using the canonical index map."""
    if pts106.shape[0] != 106:
        return pts106[:68] if pts106.shape[0] >= 68 else pts106
    return np.array([pts106[i] for i in _LANDMARK106_TO_68_IDX], dtype=pts106.dtype)


# ---------------------------------------------------------------------------
#  Device resolution helpers
# ---------------------------------------------------------------------------
def _resolve_device(device: str) -> ORTDeviceInfo:
    """Convert a user-facing device string ('0', '1', 'cpu') → ORTDeviceInfo."""
    if device is None or device.lower() in ("cpu", "-1"):
        return get_cpu_device_info()
    try:
        idx = int(device)
    except (ValueError, TypeError):
        return get_cpu_device_info()
    devices = get_available_devices_info(include_cpu=False)
    for dev in devices:
        if dev.get_index() == idx:
            return dev
    return get_cpu_device_info()


# ---------------------------------------------------------------------------
#  FaceDetector
# ---------------------------------------------------------------------------
class FaceDetector:
    """Unified face detector + landmark extractor.

    Parameters
    ----------
    detector_name : str
        One of ``"s3fd"``, ``"blazeface"``, ``"centerface"``,
        ``"yolov5face"``, ``"yolov8face"`` (case-insensitive).
    device : str or ORTDeviceInfo, optional
        ``"0"``, ``"1"``, ``"cpu"``, or an ``ORTDeviceInfo`` instance.
        Defaults to CPU.
    """

    def __init__(self, detector_name: str = "s3fd",
                 device: Optional[str] = None):
        _import_detectors()
        _import_landmarker()

        if isinstance(device, ORTDeviceInfo):
            ort_dev = device
        else:
            ort_dev = _resolve_device(device) if device else get_cpu_device_info()

        key = detector_name.lower().replace("-", "").replace("_", "")
        cls = _DETECTOR_CLASSES.get(key)
        if cls is None:
            raise ValueError(
                f"Unknown detector {detector_name!r}. "
                f"Available: {list(_DETECTOR_CLASSES.keys())}"
            )

        self._detector = cls(ort_dev)
        self._landmarker = _LANDMARKER_CLASS(ort_dev)
        self._detector_name = key

    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray,
               max_faces: int = 1,
               threshold: float = 0.5,
               min_face_size: int = 40
               ) -> List[dict]:
        """Run detection + landmark extraction.

        Parameters
        ----------
        image : np.ndarray
            BGR image, ``uint8``, shape ``(H, W, 3)``.
        max_faces : int
            Maximum number of faces to return (sorted by confidence/area).
        threshold : float
            Detection confidence threshold.
        min_face_size : int
            Minimum face side length in pixels.

        Returns
        -------
        list[dict]
            Each dict has:
                ``landmarks`` : np.ndarray (68, 2) — 68-point landmarks.
                ``bbox``      : tuple (left, top, right, bottom).
        """
        h, w = image.shape[:2]

        # ---- 1. Detect bounding boxes ----
        raw = self._detector.extract(image,
                                     threshold=threshold,
                                     fixed_window=0,
                                     min_face_size=min_face_size)
        if not raw or len(raw) == 0:
            return []

        # The detector returns list-of-lists: one list per batch item
        faces = raw[0] if isinstance(raw[0], list) else raw
        if len(faces) == 0:
            return []

        # Sort by area descending so the largest face comes first
        def _area(f):
            return (f[2] - f[0]) * (f[3] - f[1])

        faces.sort(key=_area, reverse=True)
        if max_faces > 0:
            faces = faces[:max_faces]

        # ---- 2. Extract landmarks per bbox ----
        results: List[dict] = []
        for face_rect in faces:
            l, t, r, b = map(int, face_rect[:4])
            # Clamp to image bounds
            l = max(0, l)
            t = max(0, t)
            r = min(w, r)
            b = min(h, b)
            if r <= l or b <= t:
                continue

            # Add margin
            margin = int((r - l) * 0.2)
            l_c = max(0, l - margin)
            t_c = max(0, t - margin)
            r_c = min(w, r + margin)
            b_c = min(h, b + margin)

            face_crop = image[t_c:b_c, l_c:r_c]
            if face_crop.size == 0:
                continue

            try:
                lmks_raw = self._landmarker.extract(face_crop)
            except Exception:
                continue
            if lmks_raw is None or len(lmks_raw) == 0:
                continue

            # First face's landmarks, convert back to full-image coords
            pts106 = lmks_raw[0].copy()  # (106, 2)
            pts106[:, 0] += l_c
            pts106[:, 1] += t_c

            pts68 = _landmark106to68(pts106)

            results.append({
                "landmarks": pts68.astype(np.float32),
                "bbox": (l, t, r, b),
            })

        return results
