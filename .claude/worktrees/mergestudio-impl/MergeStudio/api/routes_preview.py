import base64
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from MergeStudio.api.schemas import AnalyzeRequest, ReCompositeRequest
from MergeStudio.core.merger import MergeMaskedFace
from MergeStudio.core.model_loader import model_loader
from MergeStudio.core.config import MergerConfigMasked
from MergeStudio.core.detector.pipeline import detect_and_align
from MergeStudio.core.detector.factory import DetectorFactory, LandmarkFactory, get_device_info

router = APIRouter()

_onnx_cache = {}
_current_video = None
_current_model_path = None
_current_predictor = None


@router.get("/preview/frame/{idx}")
async def get_frame(idx: int):
    if _current_video is None:
        raise HTTPException(400, "No video loaded")

    cap = cv2.VideoCapture(_current_video)
    if not cap.isOpened():
        raise HTTPException(500, "Cannot open video")

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(404, "Frame " + str(idx) + " not found")

    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.post("/preview/analyze")
async def analyze_frame(req: AnalyzeRequest):
    if _current_video is None:
        raise HTTPException(400, "No video loaded")

    cap = cv2.VideoCapture(_current_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, req.frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(404, "Frame " + str(req.frame_idx) + " not found")

    cfg = _config_from_dict(req.config)

    # Detect faces
    device = get_device_info()
    detector = DetectorFactory.create(req.detector, device)
    landmarker = LandmarkFactory.create(req.landmarker, device)
    face_list = detect_and_align(detector, landmarker, frame, cfg.face_type)

    # Prepare face detection overlay (draw boxes)
    detection_img = frame.copy()
    faces_json = []
    for face in face_list:
        x, y, r, b = face['face_rect']
        faces_json.append({'x': x, 'y': y, 'w': r - x, 'h': b - y})
        cv2.rectangle(detection_img, (x, y), (r, b), (91, 91, 214), 2)

    # Run merge if predictor available
    swapped_img = frame.copy()
    if _current_predictor is not None and face_list:
        for face_data in face_list:
            landmarks = face_data['landmarks']
            face_out, _ = MergeMaskedFace(
                frame, landmarks, cfg, _current_predictor)
            swapped_img = face_out

    # Encode to JPEG
    _, orig_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _, det_jpeg = cv2.imencode('.jpg', detection_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _, swap_jpeg = cv2.imencode('.jpg', swapped_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return {
        "original": base64.b64encode(orig_jpeg.tobytes()).decode(),
        "detection": base64.b64encode(det_jpeg.tobytes()).decode(),
        "swapped": base64.b64encode(swap_jpeg.tobytes()).decode(),
        "face_count": len(face_list),
        "faces": faces_json,
    }


@router.post("/preview/recomposite")
async def recomposite(req: ReCompositeRequest):
    if _current_video is None:
        raise HTTPException(400, "No video loaded")
    if _current_predictor is None:
        raise HTTPException(400, "No model loaded")

    cap = cv2.VideoCapture(_current_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, req.frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(404, "Frame " + str(req.frame_idx) + " not found")

    cfg = _config_from_dict(req.config)

    # Use cached ONNX output if available
    onnx_tensor = _onnx_cache.get(req.frame_idx)
    if onnx_tensor is not None:
        # Run MergeMaskedFace with cached tensor
        pass

    # Fallback to full analysis
    device = get_device_info()
    detector = DetectorFactory.create('YOLOv8', device)
    landmarker = LandmarkFactory.create('insightface-2d106det', device)
    face_list = detect_and_align(detector, landmarker, frame, cfg.face_type)

    swapped_img = frame.copy()
    for face_data in face_list:
        landmarks = face_data['landmarks']
        face_out, _ = MergeMaskedFace(
            frame, landmarks, cfg, _current_predictor)
        swapped_img = face_out

    _, swap_jpeg = cv2.imencode('.jpg', swapped_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {
        "status": "ok",
        "swapped": base64.b64encode(swap_jpeg.tobytes()).decode(),
    }


def _config_from_dict(d: dict) -> MergerConfigMasked:
    valid_keys = [k for k in MergerConfigMasked.__init__.__code__.co_varnames
                  if k != 'self' and k != 'kwargs']
    kwargs = {k: v for k, v in d.items() if k in valid_keys}
    return MergerConfigMasked(**kwargs)


def set_current_video(video_path: str):
    global _current_video
    _current_video = video_path


def set_current_model(model_path: str):
    global _current_model_path, _current_predictor
    _current_model_path = model_path
    try:
        session = model_loader.load_model(model_path)
        def predictor(face_img):
            input_name = session.get_inputs()[0].name
            # Normalize and convert to CHW
            img = face_img.astype(np.float32) / 255.0
            input_tensor = np.transpose(img, (2, 0, 1))[None, :, :, :]
            return session.run(None, {input_name: input_tensor})[0]
        _current_predictor = predictor
    except Exception as e:
        print("Failed to load model:", e)
        _current_predictor = None
