"""python -m MaskProcessor — launch the FastAPI server."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("MaskProcessor.api.server:app", host="127.0.0.1", port=8000)
