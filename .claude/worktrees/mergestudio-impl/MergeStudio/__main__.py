"""
MergeStudio - DeepFaceLab Torch Merge Studio
Standalone web server for face swap merging.
"""
import uvicorn
from MergeStudio.api.server import create_app


def main():
    app = create_app()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
