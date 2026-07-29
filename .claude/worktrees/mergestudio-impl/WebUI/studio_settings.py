"""Studio settings — persisted as JSON in workspace/studio_settings.json."""
import json
import os

DEFAULTS = {
    "device": "0",
    "preview_scale": 2,
    "detector": "S3FD",
    "landmark": "insightface-2d106det",
    "max_faces": 1,
    # MergerConfigMasked parameters
    "mode": "overlay",
    "face_type": "full_face",
    "masked_hist_match": True,
    "mask_mode": 2,
    "erode_mask_modifier": 0,
    "blur_mask_modifier": 0,
    "motion_blur_power": 0,
    "color_transfer_mode": "rct",
    "output_face_scale": 0,
    "super_resolution_power": 0,
    "image_denoise_power": 0,
    "bicubic_degrade_power": 0,
    "color_degrade_power": 0,
    "sharpen_mode": 0,
    "blursharpen_amount": 0,
    # Export
    "output_format": "video",
    "image_format": "png",
    "jpeg_quality": 90,
    "encoder": "libx264",
    "crf": 20,
    "output_fps": 0,
}

SETTINGS_PATH = None


def init(workspace_dir: str):
    global SETTINGS_PATH
    SETTINGS_PATH = os.path.join(workspace_dir, "studio_settings.json")
    if not os.path.exists(SETTINGS_PATH):
        _write(DEFAULTS)


def _write(data: dict):
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load() -> dict:
    if not SETTINGS_PATH or not os.path.exists(SETTINGS_PATH):
        return dict(DEFAULTS)
    try:
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
        merged = dict(DEFAULTS)
        merged.update(data)
        return merged
    except (json.JSONDecodeError, IOError):
        return dict(DEFAULTS)


def save(data: dict):
    current = load()
    current.update(data)
    _write(current)


def get(key: str):
    return load().get(key, DEFAULTS.get(key))
