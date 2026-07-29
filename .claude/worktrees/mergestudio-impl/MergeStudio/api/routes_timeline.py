from fastapi import APIRouter
from MergeStudio.api.schemas import CutSegment

router = APIRouter()

_cut_segments = []


@router.get("/timeline/faces")
async def get_faces():
    """Get face distribution data for timeline display."""
    return {"faces": []}


@router.post("/timeline/cut")
async def update_cut(req: CutSegment):
    global _cut_segments

    if req.action == "add":
        _cut_segments.append({
            "start_frame": req.start_frame,
            "end_frame": req.end_frame,
        })
    elif req.action == "remove":
        _cut_segments = [
            s for s in _cut_segments
            if not (s['start_frame'] == req.start_frame and s['end_frame'] == req.end_frame)
        ]
    elif req.action == "update":
        for s in _cut_segments:
            if s['start_frame'] == req.start_frame:
                s['end_frame'] = req.end_frame

    return {"status": "ok", "segments": _cut_segments}
