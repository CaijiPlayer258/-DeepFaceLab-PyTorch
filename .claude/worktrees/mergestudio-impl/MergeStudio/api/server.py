"""
FastAPI app factory with CORS, static file mounting, and router registration.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


def create_app() -> FastAPI:
    app = FastAPI(title="MergeStudio", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ui_dir = Path(__file__).parent.parent / "ui"
    if ui_dir.exists():
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    from MergeStudio.api.routes_project import router as project_router
    from MergeStudio.api.routes_preview import router as preview_router
    from MergeStudio.api.routes_timeline import router as timeline_router
    from MergeStudio.api.routes_export import router as export_router

    app.include_router(project_router, prefix="/api")
    app.include_router(preview_router, prefix="/api")
    app.include_router(timeline_router, prefix="/api")
    app.include_router(export_router, prefix="/api")

    return app
