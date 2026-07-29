from pathlib import Path
import cv2
from fastapi import APIRouter, HTTPException
from MergeStudio.api.schemas import OpenProjectRequest, OpenProjectResponse, ModelLoadRequest

router = APIRouter()


@router.post("/project/open", response_model=OpenProjectResponse)
async def open_project(req: OpenProjectRequest):
    workspace = Path(req.path)
    if not workspace.exists():
        return OpenProjectResponse(status="error", message="Path not found: " + req.path)

    # Scan for video files
    video_path = None
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        candidates = list(workspace.glob('*' + ext))
        if candidates:
            video_path = str(candidates[0])
            break

    total_frames = 0
    fps = 0.0
    width = 0
    height = 0
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

    # Scan for models
    models = []
    model_dir = workspace / "model"
    if model_dir.exists():
        for f in sorted(model_dir.glob("*")):
            if f.suffix.lower() in ('.dfm', '.pth', '.npy'):
                models.append({
                    "name": f.name,
                    "path": str(f),
                    "format": f.suffix.lower().lstrip('.'),
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                })

    # Scan aligned directory
    aligned_dir = workspace / "data_dst" / "aligned"
    aligned_count = 0
    has_aligned = aligned_dir.exists()
    if has_aligned:
        aligned_count = len(list(aligned_dir.glob("*.jpg")))

    # List video files
    videos = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        for f in workspace.glob('*' + ext):
            videos.append(f.name)

    return OpenProjectResponse(
        status="ok",
        video_path=video_path,
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        models=models,
        videos=videos,
        has_aligned=has_aligned,
        aligned_count=aligned_count,
    )


@router.get("/models")
async def list_models():
    return {"models": []}


@router.post("/models/load")
async def load_model(req: ModelLoadRequest):
    model_path = Path(req.name)
    if not model_path.exists():
        raise HTTPException(404, "Model not found: " + req.name)
    if model_path.suffix.lower() != '.dfm':
        raise HTTPException(400, "Only .dfm (ONNX) models can be loaded directly")
    return {"status": "loading", "name": req.name, "format": "dfm"}


@router.post("/models/export-to-dfm")
async def export_to_dfm(req: ModelLoadRequest):
    """Convert .pth or .npy model to .dfm."""
    model_path = Path(req.name)
    if not model_path.exists():
        raise HTTPException(404, "Model not found: " + req.name)
    suffix = model_path.suffix.lower()
    if suffix not in ('.pth', '.npy'):
        raise HTTPException(400, "Unsupported model format: " + suffix + " (expected .pth or .npy)")
    return {"status": "converting", "source": req.name, "format": suffix}
