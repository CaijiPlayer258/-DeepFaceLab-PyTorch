import asyncio
import uuid
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from MergeStudio.api.schemas import ExportStartRequest
from MergeStudio.core.export_pipeline import ExportPipeline

router = APIRouter()

_active_exports = {}


@router.post("/export/start")
async def start_export(req: ExportStartRequest):
    export_id = str(uuid.uuid4())[:8]

    pipeline = ExportPipeline(
        workspace=".",
        video_path=".",
        output_path="output_" + export_id + ".mp4",
        config=req.config,
        cut_segments=req.cut_segments,
        output_format=req.format,
        quality=req.quality,
    )

    _active_exports[export_id] = pipeline
    return {"export_id": export_id, "status": "started"}


@router.get("/export/progress/{export_id}")
async def get_progress(export_id: str):
    pipeline = _active_exports.get(export_id)
    if pipeline is None:
        raise HTTPException(404, "Export not found")

    async def event_stream():
        while True:
            status = pipeline.status
            yield "data: " + json.dumps(status) + "\n\n"
            if status['stage'] in ('complete', 'error'):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/export/cancel/{export_id}")
async def cancel_export(export_id: str):
    pipeline = _active_exports.get(export_id)
    if pipeline is None:
        raise HTTPException(404, "Export not found")
    pipeline.cancel()
    return {"status": "cancelled"}
