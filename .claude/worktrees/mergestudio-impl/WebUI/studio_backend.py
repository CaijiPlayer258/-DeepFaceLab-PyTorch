"""FFmpeg-based frame extraction backend for Merger Studio."""
import os
import subprocess
import tempfile
import uuid
import threading
from fractions import Fraction
from pathlib import Path

import threading as _threading
import uuid as _uuid
import math as _math
import time as _time
import cv2 as _cv2
import numpy as _np
from pathlib import Path as _Path
from mainscripts import Extractor as _Extractor

_batch_tasks = {}
_batch_lock = _threading.Lock()

class BatchExportTask:
    def __init__(self, sid, dfm_path, output_dir, start_sec, end_sec, settings):
        self.sid = sid
        self.dfm_path = dfm_path
        self.output_dir = output_dir
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.settings = settings
        self.cancelled = False
        self.status = "pending"
        self.progress = 0.0
        self.current_frame = 0
        self.total_frames = 0
        self.error = None
        self.elapsed_sec = 0.0

class BatchExportV2Task:
    def __init__(self, sid, options):
        self.sid = sid
        self.options = options
        self.cancelled = False
        self.status = "pending"
        self.progress = 0.0
        self.current_stage = 0
        self.stage_name = ""
        self.stages = [
            {"name": "Extract frames", "progress": 0.0, "status": "pending"},
            {"name": "Extract faces", "progress": 0.0, "status": "pending"},
            {"name": "Merge frames", "progress": 0.0, "status": "pending"},
        ]
        self.current_frame = 0
        self.total_frames = 0
        self.error = None
        self.elapsed_sec = 0.0
        self._resume_event = _threading.Event()
        self._resume_event.set()
        self._ffmpeg_proc = None  # track ffmpeg subprocess for kill
        self._run_thread = None   # track the background thread

class StudioBackend:
    def __init__(self):
        self._sessions = {}  # session_id -> video_path
        self._thumbs = {}    # session_id -> {'data': bytes, 'width': int, 'height': int}
        self._lock = threading.Lock()
        self._ffmpeg = self._find_ffmpeg()
        self._hwaccel = self._detect_hwaccel()
        self._loaded_model = None  # path to loaded DFM model
        self._workspace_dir = None

    def _find_ffmpeg(self):
        """Locate ffmpeg binary."""
        _root = Path(__file__).parent.parent
        paths = [
            str(_root / 'ffmpeg' / 'ffmpeg.exe'),
            str(_root / '_internal' / 'ffmpeg' / 'ffmpeg.exe'),
            str(_root / '_internal' / 'ffmpeg.exe'),
            str(_root / 'ffmpeg.exe'),
            str(Path.home() / '.dfl' / 'ffmpeg.exe'),
            str(Path.home() / 'ffmpeg' / 'ffmpeg.exe'),
            'C:\\ffmpeg\\bin\\ffmpeg.exe',
            'ffmpeg', 'ffmpeg.exe',
        ]
        for p in paths:
            try:
                subprocess.run([p, '-version'], capture_output=True, timeout=5)
                print(f'[StudioBackend] ffmpeg found: {p}')
                return p
            except Exception:
                continue
        print('[StudioBackend] WARNING: ffmpeg not found, frame extraction disabled')
        return None

    def _detect_hwaccel(self):
        """Detect available GPU for hardware-accelerated decoding."""
        try:
            result = subprocess.run(
                [self._ffmpeg, '-hide_banner', '-hwaccels'],
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout
            if 'cuda' in output:
                return 'cuda'
            elif 'd3d11va' in output:
                return 'd3d11va'
            elif 'qsv' in output:
                return 'qsv'
        except Exception:
            pass
        return None

    def upload_video(self, file_data, filename):
        """Store uploaded video and return session_id."""
        sid = uuid.uuid4().hex[:12]
        ext = Path(filename).suffix or '.mp4'
        tmp_path = Path(tempfile.gettempdir()) / f'dfl_studio_{sid}{ext}'
        tmp_path.write_bytes(file_data)
        with self._lock:
            self._sessions[sid] = {'path': str(tmp_path), 'original_name': Path(filename).stem}
        return sid

    def get_video_path(self, sid):
        with self._lock:
            info = self._sessions.get(sid)
            return info['path'] if isinstance(info, dict) else info

    def get_original_name(self, sid):
        with self._lock:
            info = self._sessions.get(sid)
            if isinstance(info, dict):
                return info.get('original_name', f'video_{sid}')
            return f'video_{sid}'

    def extract_frame(self, sid, time_sec, quality=2):
        """Extract a single frame at time_sec (JPEG bytes). Key: -ss BEFORE -i."""
        if not self._ffmpeg:
            return None
        video_path = self.get_video_path(sid)
        if not video_path or not Path(video_path).exists():
            return None

        cmd = [self._ffmpeg, '-hide_banner', '-loglevel', 'error']
        # GPU acceleration (no hwaccel_output_format cuda — pipe:1 needs system memory)
        if self._hwaccel == 'cuda':
            cmd.extend(['-hwaccel', 'cuda'])
        elif self._hwaccel == 'd3d11va':
            cmd.extend(['-hwaccel', 'd3d11va'])

        # -ss BEFORE -i for fast seeking (10-50x faster)
        cmd.extend([
            '-ss', str(time_sec),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', str(quality),
            '-f', 'image2',
            'pipe:1'
        ])

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except Exception:
            pass
        return None

    def probe_duration(self, sid):
        """Get video duration in seconds."""
        video_path = self.get_video_path(sid)
        if not video_path:
            return None
        cmd = [
            self._ffmpeg, '-hide_banner', '-loglevel', 'error',
            '-i', video_path,
            '-f', 'null', '-'
        ]
        try:
            # Parse duration from stderr
            result = subprocess.run(
                [self._ffmpeg, '-i', video_path],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stderr.split('\n'):
                if 'Duration' in line:
                    # Duration: 00:00:10.05
                    parts = line.split('Duration: ')[1].split(',')[0]
                    h, m, s = parts.split(':')
                    return float(h) * 3600 + float(m) * 60 + float(s)
        except Exception:
            pass
        return None

    def start_stream(self, sid, start_sec=0, fps=30, quality=3):
        """Start ffmpeg MJPEG stream subprocess. Returns the Popen object."""
        video_path = self.get_video_path(sid)
        if not video_path or not Path(video_path).exists():
            return None
        cmd = [self._ffmpeg, '-hide_banner', '-loglevel', 'error']
        if self._hwaccel == 'cuda':
            cmd.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
        cmd.extend([
            '-ss', str(start_sec),
            '-i', video_path,
            '-vf', f'fps={fps},scale=iw:ih',  # maintain resolution, enforce fps
            '-q:v', str(quality),
            '-f', 'mjpeg',
            'pipe:1'
        ])
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            with self._lock:
                key = f'stream_{sid}'
                if key in self._sessions:
                    old = self._sessions.get(key)
                    if old and hasattr(old, 'poll') and old.poll() is None:
                        old.kill()
                self._sessions[key] = proc
            return proc
        except Exception:
            return None

    def stop_stream(self, sid):
        """Kill streaming ffmpeg process."""
        with self._lock:
            key = f'stream_{sid}'
            proc = self._sessions.pop(key, None)
        if proc and hasattr(proc, 'poll') and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass

    def seek_stream(self, sid, new_time):
        """Restart stream at new position."""
        return self.start_stream(sid, new_time)

    # ── Workspace & Model management ─────────────────────────────────
    def set_workspace(self, path: str):
        self._workspace_dir = path

    def load_model(self, dfm_path: str) -> dict:
        """Validate and record loaded DFM model path. Also registers with studio_pipeline."""
        from pathlib import Path
        if not Path(dfm_path).exists():
            raise FileNotFoundError(f"DFM model not found: {dfm_path}")
        try:
            from . import studio_pipeline as sp
            info = sp.load_dfm(dfm_path)
            self._loaded_model = dfm_path
            return {
                "path": dfm_path,
                "resolution": info.get("resolution", 0),
                "provider": info.get("provider", "?"),
                "loaded": True,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load DFM model: {e}")

    def get_loaded_model(self) -> dict:
        if not self._loaded_model:
            return {"loaded": False}
        return {"path": self._loaded_model, "loaded": True}

    def unload_model(self):
        self._loaded_model = None

    # ── Settings ─────────────────────────────────────────────────────
    def get_settings(self) -> dict:
        from . import studio_settings
        return studio_settings.load()

    def update_settings(self, data: dict) -> dict:
        from . import studio_settings
        studio_settings.save(data)
        return self.get_settings()

    # ── Cache key & check ────────────────────────────────────────────
    def get_cache_key(self, video_path: str) -> str:
        """Cache key: filename_size from a file path."""
        from pathlib import Path
        p = Path(video_path)
        try:
            size = p.stat().st_size
            return f"{p.stem}_{size}"
        except IOError:
            return f"{p.stem}_0"

    def get_cache_key_for_sid(self, sid: str) -> str:
        """Cache key using the ORIGINAL uploaded filename (not temp path)."""
        video_path = self.get_video_path(sid)
        if not video_path:
            return None
        from pathlib import Path
        orig_name = self.get_original_name(sid)
        try:
            size = Path(video_path).stat().st_size
            return f"{orig_name}_{size}"
        except IOError:
            return f"{orig_name}_0"

    def check_cache(self, sid: str) -> dict:
        video_path = self.get_video_path(sid)
        if not video_path or not self._workspace_dir:
            return {"hit": False}
        from pathlib import Path
        ws_cache = Path(self._workspace_dir) / "studio_cache"
        cache_key = self.get_cache_key_for_sid(sid)
        if not cache_key:
            return {"hit": False}
        cache_dir = ws_cache / cache_key
        meta_path = cache_dir / "meta.json"
        if cache_dir.exists() and meta_path.exists():
            try:
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                return {"hit": True, "cache_dir": str(cache_dir), "meta": meta}
            except Exception:
                pass
        return {"hit": False}

    # ── Thumb cache extraction ───────────────────────────────────────
    def extract_thumb_cache(self, sid: str, cache_dir: str) -> str:
        """Extract video to 2fps 360p JPEG frames. Returns cache_dir on success."""
        import subprocess, math, json, os
        from pathlib import Path

        video_path = self.get_video_path(sid)
        if not video_path or not Path(video_path).exists():
            raise RuntimeError("Video not found")

        duration = self.probe_duration(sid)
        if not duration or duration <= 0:
            raise RuntimeError("Could not determine video duration")

        os.makedirs(cache_dir, exist_ok=True)

        cmd = [self._ffmpeg, "-hide_banner", "-loglevel", "error"]
        if self._hwaccel in ("cuda", "d3d11va", "qsv"):
            cmd.extend(["-hwaccel", self._hwaccel])
        cmd.extend([
            "-i", video_path,
            "-vf", "fps=2,scale=640:360:force_original_aspect_ratio=decrease",
            "-q:v", "5",
            "-start_number", "0",
            "-y",
            os.path.join(cache_dir, "frame_%06d.jpg"),
        ])

        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            raise RuntimeError(f"FFmpeg extraction failed: {stderr}")

        total_frames = max(1, math.ceil(duration * 2))
        orig_name = self.get_original_name(sid)
        meta = {
            "original_filename": orig_name,
            "fps": 2,
            "total_frames": total_frames,
            "width": 640,
            "height": 360,
            "quality": 5,
            "version": 1,
        }
        with open(os.path.join(cache_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        return cache_dir

    # ── Single frame export ──────────────────────────────────────────
    def export_frame(self, sid: str, time_sec: float) -> bytes:
        """Extract a single frame at full quality. Returns JPEG bytes."""
        import subprocess
        from pathlib import Path

        video_path = self.get_video_path(sid)
        if not video_path or not Path(video_path).exists():
            raise RuntimeError("Video not found")

        cmd = [self._ffmpeg, "-hide_banner", "-loglevel", "error"]
        if self._hwaccel == "cuda":
            cmd.extend(["-hwaccel", "cuda"])
        elif self._hwaccel == "d3d11va":
            cmd.extend(["-hwaccel", "d3d11va"])
        cmd.extend([
            "-ss", str(time_sec),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            "-f", "image2", "pipe:1",
        ])
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0 or not result.stdout:
            raise RuntimeError("Frame export failed")
        return result.stdout

    def analyze_frame(self, sid: str, dfm_path: str, settings: dict) -> dict:
        """Extract current video frame, run pipeline, return base64 results."""
        import numpy as np, cv2
        video_path = self.get_video_path(sid)
        if not video_path:
            raise RuntimeError("No video loaded")
        time_sec = float(settings.get("time_sec", 0))
        jpg_bytes = self.export_frame(sid, time_sec)
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        from . import studio_pipeline as sp
        return sp.analyze_frame(frame, dfm_path, settings)

    def recomposite(self, sid: str, dfm_path: str, settings: dict) -> dict:
        """Re-composite with new settings using cached model outputs (no ONNX inference)."""
        import numpy as np, cv2
        video_path = self.get_video_path(sid)
        if not video_path:
            raise RuntimeError("No video loaded")
        time_sec = float(settings.get("time_sec", 0))
        jpg_bytes = self.export_frame(sid, time_sec)
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        settings['_skip_inference'] = True
        from . import studio_pipeline as sp
        return sp.analyze_frame(frame, dfm_path, settings)

    def cleanup_session(self, sid):
        """Remove session video file, cached thumbs, and stop any stream."""
        self.stop_stream(sid)
        video_path = self.get_video_path(sid)
        if video_path:
            try:
                Path(video_path).unlink(missing_ok=True)
            except Exception:
                pass
        with self._lock:
            self._sessions.pop(sid, None)
            self._thumbs.pop(sid, None)

    def start_batch_export(self, sid, dfm_path, output_dir, start_sec, end_sec, settings):
        task_id = _uuid.uuid4().hex[:12]
        task = BatchExportTask(sid, dfm_path, output_dir, start_sec, end_sec, settings)
        duration = end_sec - start_sec
        fps = 30
        task.total_frames = max(1, _math.ceil(duration * fps))
        with _batch_lock:
            _batch_tasks[task_id] = task
        def _run():
            os.makedirs(output_dir, exist_ok=True)
            from . import studio_pipeline as sp
            task.status = "running"
            t0 = _time.perf_counter()
            for i in range(task.total_frames):
                if task.cancelled:
                    task.status = "cancelled"
                    return
                t = start_sec + i / fps
                try:
                    jpg_bytes = self.export_frame(sid, t)
                    nparr = _np.frombuffer(jpg_bytes, _np.uint8)
                    frame = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
                    faces = sp.detect_faces(frame, settings.get("detector", "s3fd"), settings.get("max_faces", 1))
                    if faces and sp.is_loaded(dfm_path):
                        swapped = sp.swap_face(frame, faces[0]["landmarks"], dfm_path)
                    else:
                        swapped = frame
                    out_path = os.path.join(output_dir, f"frame_{i:06d}.png")
                    _cv2.imwrite(out_path, swapped)
                except Exception as e:
                    print(f"[BatchExport] frame {i} failed: {e}")
                task.current_frame = i + 1
                task.progress = (i + 1) / task.total_frames
                task.elapsed_sec = _time.perf_counter() - t0
            task.status = "done"
        t = _threading.Thread(target=_run, daemon=True)
        t.start()
        return task_id

    def get_batch_status(self, task_id):
        with _batch_lock:
            task = _batch_tasks.get(task_id)
        if not task:
            return {"status": "not_found"}
        result = {
            "status": task.status,
            "progress": task.progress,
            "current_frame": task.current_frame,
            "total_frames": task.total_frames,
            "elapsed_sec": task.elapsed_sec,
            "error": task.error,
        }
        if hasattr(task, 'current_stage'):
            result["current_stage"] = task.current_stage
            result["stage_name"] = task.stage_name
            result["stages"] = task.stages
        return result

    def cancel_batch_export(self, task_id):
        with _batch_lock:
            task = _batch_tasks.get(task_id)
        if task:
            task.cancelled = True
            # Kill ffmpeg process if running
            if hasattr(task, '_ffmpeg_proc') and task._ffmpeg_proc:
                try:
                    task._ffmpeg_proc.kill()
                    task._ffmpeg_proc.wait(timeout=3)
                    print(f"[cancel] ffmpeg process killed for task {task_id}")
                except Exception:
                    pass
            # Resume if waiting for user
            if hasattr(task, '_resume_event'):
                task._resume_event.set()
            print(f"[cancel] Task {task_id} cancelled")

    # ── Batch export v2 ─────────────────────────────────────────────
    def batch_extract_frames(self, sid, output_dir, format="jpg", jpeg_quality=100, fps=30,
                             progress_callback=None):
        """Extract all frames from session video to output_dir using ffmpeg.
        Updates progress via callback(current_frame, total_frames).
        """
        import re as _re
        video_path = self.get_video_path(sid)
        if not video_path or not _Path(video_path).exists():
            raise RuntimeError("Video not found")

        _Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Probe duration and total frames first
        duration = self.probe_duration(sid) or 0
        total_frames = max(1, int(duration * fps))
        print(f"[batch_extract_frames] duration={duration:.1f}s fps={fps} total_frames={total_frames} output={output_dir}")

        qv = max(1, min(31, int(round(31 - (jpeg_quality - 1) * 30 / 99))))
        ext = "jpg" if format.lower() == "jpg" else "png"

        cmd = [self._ffmpeg, "-hide_banner", "-loglevel", "error",
               "-progress", "pipe:1", "-nostats"]
        if self._hwaccel in ("cuda", "d3d11va", "qsv"):
            cmd.extend(["-hwaccel", self._hwaccel])
        cmd.extend([
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", str(qv),
            "-start_number", "0",
            "-y",
            os.path.join(output_dir, f"frame_%06d.{ext}"),
        ])

        print(f"[batch_extract_frames] ffmpeg cmd: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # Store process handle for cancel/kill (find task via progress_callback closure)
        # This is set on the BatchExportV2Task from the _make_cb closure
        if progress_callback and hasattr(progress_callback, '_task_ref'):
            progress_callback._task_ref._ffmpeg_proc = proc
        last_frame = 0
        for line_raw in iter(proc.stdout.readline, b''):
            line = line_raw.decode(errors='replace').strip()
            if '=' in line:
                key, val = line.split('=', 1)
                if key == 'frame':
                    last_frame = int(val)
                    if progress_callback and total_frames > 0:
                        progress_callback(min(last_frame, total_frames), total_frames)
                elif key == 'out_time_us' or key == 'out_time_ms':
                    t_us = int(val)
                    current_time = t_us / 1000000.0
                    pct = min(100, int(current_time / duration * 100)) if duration > 0 else 0
                    print(f"  [ffmpeg] frame={last_frame} time={current_time:.1f}s/{duration:.1f}s {pct}%", flush=True)
                elif key == 'progress' and val == 'end':
                    break

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"FFmpeg frame extraction failed (return code {rc})")

        if progress_callback:
            progress_callback(total_frames, total_frames)
        print(f"[batch_extract_frames] done: {total_frames} frames -> {output_dir}")
        return output_dir

    def batch_extract_faces(self, frame_dir, aligned_dir, detector="s3fd", max_faces=1, device="0"):
        """Extract faces from frames using mainscripts.Extractor.main().

        Args:
            frame_dir: Directory containing extracted frame images.
            aligned_dir: Output directory for aligned DFL face images.
            detector: Face detector name ("s3fd", "blazeface", etc.).
            max_faces: Maximum faces to extract per frame.
            device: Device string ("0" for GPU 0, "cpu", etc.).

        Returns:
            aligned_dir path on success.
        """
        input_path = _Path(frame_dir)
        output_path = _Path(aligned_dir)

        if not input_path.exists():
            raise RuntimeError(f"Frame directory not found: {frame_dir}")

        output_path.mkdir(parents=True, exist_ok=True)

        cpu_only = device.lower() == "cpu"
        force_gpu_idxs = None
        if not cpu_only:
            try:
                force_gpu_idxs = [int(device)]
            except (ValueError, TypeError):
                force_gpu_idxs = [0]

        _Extractor.main(
            detector=detector,
            input_path=input_path,
            output_path=output_path,
            output_debug=None,
            manual_fix=False,
            manual_output_debug_fix=False,
            face_type='full_face',
            max_faces_from_image=max_faces,
            image_size=512,
            jpeg_quality=100,
            cpu_only=cpu_only,
            force_gpu_idxs=force_gpu_idxs,
        )

        return aligned_dir

    def batch_merge_frames(self, frame_dir, aligned_dir, dfm_path, settings,
                           progress_callback=None):
        """Merge faces back onto frames using MergeMaskedFace pipeline.

        Iterates over frames in frame_dir, loads the corresponding aligned
        DFL face (landmarks), runs MergeMaskedFace with the given settings,
        and overwrites the original frame in place.

        Args:
            frame_dir: Directory containing the original frame images.
            aligned_dir: Directory containing aligned DFL face images.
            dfm_path: Path to the loaded DFM model.
            settings: Merged UI settings dict.
            progress_callback: Optional func(current, total).

        Returns:
            dict with keys: total, succeeded, failed.
        """
        from . import studio_pipeline as sp
        from merger.MergeMasked import MergeMaskedFace
        from merger.MergerConfig import MergerConfigMasked, ctm_str_dict
        from merger.FrameInfo import FrameInfo
        from facelib import FaceType
        from DFLIMG import DFLJPG

        # Validate model is loaded
        sess = sp._sessions.get(dfm_path)
        if sess is None:
            raise RuntimeError(f"Model {dfm_path} is not loaded. Call load_model() first.")

        # Predictor setup (same pattern as studio_pipeline.swap_face)
        input_h, input_w = sp._input_shape_cache.get(dfm_path, (256, 256))
        predictor_input_shape = (input_h, input_w, 3)

        inp_name = sess.get_inputs()[0].name
        inp_shape = sess.get_inputs()[0].shape
        is_nchw = len(inp_shape) == 4 and inp_shape[1] in (1, 3)

        def predictor_func(face_bgr):
            if is_nchw:
                inp = face_bgr.transpose(2, 0, 1)[_np.newaxis, ...].astype(_np.float32)
            else:
                inp = face_bgr[_np.newaxis, ...].astype(_np.float32)
            outs = sess.run(None, {inp_name: inp})

            def _h(arr):
                arr = _np.clip(arr, 0.0, 1.0)
                if arr.ndim == 4:
                    arr = arr.squeeze(0)
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = arr.transpose(1, 2, 0)
                if arr.ndim == 2:
                    arr = arr[..., None]
                return arr

            bgr = _h(outs[1]) if len(outs) > 1 else _h(outs[0])
            msk = _h(outs[0])
            msk_dst = _h(outs[2]) if len(outs) > 2 else msk.copy()
            return bgr, msk, msk_dst

        # Build MergerConfig from UI settings
        ft_map = {
            "half_face": FaceType.HALF, "mid_full": FaceType.MID_FULL,
            "full_face": FaceType.FULL, "whole_face": FaceType.WHOLE_FACE,
            "head": FaceType.HEAD,
        }
        cfg = MergerConfigMasked(
            face_type=ft_map.get(settings.get("face_type", "full_face"), FaceType.FULL)
        )
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

        # Gather frame files
        frame_extensions = ("*.jpg", "*.jpeg", "*.png")
        frame_paths = []
        for ext in frame_extensions:
            frame_paths.extend(sorted(_Path(frame_dir).glob(ext)))

        total = len(frame_paths)
        succeeded = 0
        failed = 0

        for i, frame_path in enumerate(frame_paths):
            if progress_callback:
                progress_callback(i + 1, total)
            frame_stem = frame_path.stem
            aligned_path = _Path(aligned_dir) / f"{frame_stem}_0.jpg"

            if not aligned_path.exists():
                print(f"[batch_merge] No aligned face for {frame_stem}, skipping")
                failed += 1
                continue

            try:
                frame = _cv2.imread(str(frame_path), _cv2.IMREAD_COLOR)
                if frame is None:
                    failed += 1
                    continue

                dfl_img = DFLJPG.load(str(aligned_path))
                if dfl_img is None or not dfl_img.has_data():
                    failed += 1
                    continue

                landmarks = dfl_img.get_landmarks()

                frame_f32 = frame.astype(_np.float32) / 255.0
                frame_info = FrameInfo(landmarks_list=[landmarks])

                mmf_result = MergeMaskedFace(
                    predictor_func, predictor_input_shape,
                    sp._noop_enhancer,
                    sp._real_xseg if cfg.mask_mode >= 6 else sp._noop_xseg,
                    cfg, frame_info, frame, frame_f32, landmarks,
                )

                if isinstance(mmf_result, tuple):
                    mmf_result = mmf_result[0]

                # Overwrite the original frame in place
                _cv2.imwrite(str(frame_path), mmf_result)
                succeeded += 1

            except Exception as e:
                import traceback
                print(f"[batch_merge] Frame {frame_stem} failed: {e}")
                traceback.print_exc()
                failed += 1

        return {"total": total, "succeeded": succeeded, "failed": failed}

    def resume_batch_export(self, task_id):
        """Resume a v2 batch export task paused for user filter approval.

        Args:
            task_id: The task ID returned by start_export_v2.

        Returns:
            True if the task was found and resumed, False otherwise.
        """
        with _batch_lock:
            task = _batch_tasks.get(task_id)
        if task and task.status == "waiting_for_user":
            task._resume_event.set()
            return True
        return False

    def start_export_v2(self, sid, options):
        """Multi-stage batch export: extract frames -> extract faces -> merge frames.

        Stages:
            1. batch_extract_frames  (ffmpeg)
            2. batch_extract_faces   (mainscripts.Extractor)
            2.5. pause if need_pause_for_filter=True (status=waiting_for_user)
            3. batch_merge_frames    (MergeMaskedFace)

        Args:
            sid: Session ID for the uploaded video.
            options: dict with keys:
                output_dir (str)          — root output directory.
                format (str)              — "jpg" or "png".
                jpeg_quality (int)        — 1-100.
                fps (int)                 — output frame rate.
                detector (str)            — face detector name.
                max_faces (int)           — max faces per frame.
                device (str)              — "0", "cpu", etc.
                need_pause_for_filter (bool) — pause after face extraction.
                dfm_path (str)            — path to DFM model.
                settings (dict)           — merge UI settings.

        Returns:
            task_id (str) for status polling via get_batch_status().
        """
        task_id = _uuid.uuid4().hex[:12]
        task = BatchExportV2Task(sid, options)

        with _batch_lock:
            _batch_tasks[task_id] = task

        def _run_v2():
            try:
                task.status = "running"
                t0 = _time.perf_counter()

                output_dir = options.get("output_dir",
                    os.path.join(self._workspace_dir or ".", "workspace", "export",
                                 f"export_{_uuid.uuid4().hex[:8]}"))
                frame_dir = os.path.join(output_dir, "frames")
                aligned_dir = os.path.join(output_dir, "aligned")
                dfm_path = options.get("dfm_path")
                settings = options.get("settings", {})

                os.makedirs(frame_dir, exist_ok=True)
                os.makedirs(aligned_dir, exist_ok=True)

                # Stage 1: extract frames
                task.current_stage = 1
                task.stage_name = "Extracting frames"
                task.stages[0]["status"] = "running"
                task.stages[0]["progress"] = 0.0

                if task.cancelled:
                    task.status = "cancelled"
                    return

                # Progress callback: updates task.stages and prints log
                def _make_cb(stage_idx, label):
                    def _cb(curr, total):
                        p = min(1.0, curr / max(1, total)) if total > 0 else 0
                        task.stages[stage_idx]["current"] = curr
                        task.stages[stage_idx]["total"] = total
                        task.stages[stage_idx]["progress"] = p
                        task.progress = (stage_idx * 0.33 + p * 0.33)
                        task.elapsed_sec = _time.perf_counter() - t0
                        print(f"[export] {label}: {curr}/{total} ({int(p*100)}%)", flush=True)
                    _cb._task_ref = task  # link to task for ffmpeg process tracking
                    return _cb

                self.batch_extract_frames(
                    sid, frame_dir,
                    format=options.get("format", "jpg"),
                    jpeg_quality=options.get("jpeg_quality", 100),
                    fps=options.get("fps", 30),
                    progress_callback=_make_cb(0, "extract frames"),
                )

                task.stages[0]["status"] = "done"
                print(f"[export] Stage 1 done. elapsed={_time.perf_counter()-t0:.1f}s")
                task.elapsed_sec = _time.perf_counter() - t0

                if task.cancelled:
                    task.status = "cancelled"
                    return

                # Stage 2: extract faces
                task.current_stage = 2
                task.stage_name = "Extracting faces"
                task.stages[1]["status"] = "running"
                task.stages[1]["progress"] = 0.0

                # Count frames for progress
                frame_files = sorted(_Path(frame_dir).glob("frame_*.*"))
                task.stages[1]["total"] = len(frame_files)
                task.stages[1]["current"] = 0

                self.batch_extract_faces(
                    frame_dir, aligned_dir,
                    detector=options.get("detector", "s3fd"),
                    max_faces=options.get("max_faces", 1),
                    device=options.get("device", "0"),
                )

                # After extractor done, check results
                aligned_files = sorted(_Path(aligned_dir).glob("*.*"))
                print(f"[export] Stage 2 done: {len(aligned_files)} faces extracted from {len(frame_files)} frames")
                task.stages[1]["status"] = "done"
                task.stages[1]["progress"] = 1.0
                task.stages[1]["current"] = len(aligned_files)
                task.elapsed_sec = _time.perf_counter() - t0

                if task.cancelled:
                    task.status = "cancelled"
                    return

                # Stage 2.5: pause for user to filter aligned faces
                if options.get("need_pause_for_filter"):
                    task.status = "waiting_for_user"
                    task.stage_name = "Waiting for user filter approval"
                    task.aligned_dir = str(aligned_dir)
                    task._resume_event.clear()
                    print(f"[export] Stage 2.5: waiting for user at {aligned_dir}")
                    task._resume_event.wait()
                    task.status = "running"
                    print(f"[export] Stage 2.5: user resumed")

                if task.cancelled:
                    task.status = "cancelled"
                    return

                # Stage 3: merge frames
                task.current_stage = 3
                task.stage_name = "Merging frames"
                task.stages[2]["status"] = "running"
                task.stages[2]["progress"] = 0.0

                # Count frames for merge progress
                frame_files_st3 = sorted(_Path(frame_dir).glob("frame_*.*"))
                task.stages[2]["total"] = len(frame_files_st3)
                task.stages[2]["current"] = 0

                result = self.batch_merge_frames(frame_dir, aligned_dir, dfm_path, settings,
                                                  progress_callback=_make_cb(2, "merge"))

                task.current_frame = result.get("succeeded", 0)
                task.total_frames = result.get("total", 0)
                task.stages[2]["status"] = "done"
                task.stages[2]["progress"] = 1.0
                task.progress = 1.0
                task.status = "done"
                task.elapsed_sec = _time.perf_counter() - t0

            except Exception as e:
                import traceback
                traceback.print_exc()
                task.status = "error"
                task.error = str(e)

        task._run_thread = _threading.Thread(target=_run_v2, daemon=True)
        task._run_thread.start()
        return task_id

    def open_directory(self, path):
        """Open the given path in the system file manager.

        On Windows uses os.startfile (explorer).
        Returns True if successful, False otherwise.
        """
        try:
            os.startfile(path)
            return True
        except Exception as e:
            print(f"[open_directory] Failed to open {path}: {e}")
            return False

    # ── Cache management ────────────────────────────────────────────
    def list_cache(self) -> list:
        if not self._workspace_dir:
            return []
        cache_root = os.path.join(self._workspace_dir, "studio_cache")
        if not os.path.isdir(cache_root):
            return []
        entries = []
        for name in sorted(os.listdir(cache_root)):
            meta_path = os.path.join(cache_root, name, "meta.json")
            if os.path.isfile(meta_path):
                try:
                    import json
                    with open(meta_path) as f:
                        meta = json.load(f)
                    total_size = 0
                    frame_dir = os.path.join(cache_root, name)
                    for fname in os.listdir(frame_dir):
                        if fname.endswith('.jpg'):
                            total_size += os.path.getsize(os.path.join(frame_dir, fname))
                    entries.append({
                        "key": name,
                        "filename": meta.get("original_filename", "unknown"),
                        "total_frames": meta.get("total_frames", 0),
                        "size_bytes": total_size,
                        "size_mb": round(total_size / 1048576, 1),
                    })
                except:
                    pass
        return entries

    def delete_cache(self, cache_key: str) -> bool:
        import shutil
        cache_dir = os.path.join(self._workspace_dir, "studio_cache", cache_key)
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
            return True
        return False

    def get_frames_strip(self, cache_key, start, count, width, height):
        from PIL import Image
        import io, os
        cache_dir = os.path.join(self._workspace_dir, "studio_cache", cache_key)
        if not os.path.isdir(cache_dir):
            return None
        images = []
        for i in range(start, start + count):
            fpath = os.path.join(cache_dir, f"frame_{i:06d}.jpg")
            if os.path.isfile(fpath):
                img = Image.open(fpath)
                img = img.resize((width, height), Image.LANCZOS)
                images.append(img)
        if not images:
            return None
        total_w = width * len(images)
        strip = Image.new('RGB', (total_w, height))
        for idx, img in enumerate(images):
            strip.paste(img, (idx * width, 0))
        buf = io.BytesIO()
        strip.save(buf, format='JPEG', quality=80)
        return buf.getvalue()

    def extract_thumbstrip(self, sid, n, width, height, quality=4):
        """Extract n evenly-spaced thumbnails as a single horizontal JPEG strip.

        Runs ffmpeg ONCE (not n times) using fps + tile filters.
        Returns raw JPEG bytes of a strip image (width*n × height).
        """
        video_path = self.get_video_path(sid)
        if not video_path or not Path(video_path).exists():
            return None
        duration = self.probe_duration(sid)
        if not duration or duration <= 0 or n < 1:
            return None

        cmd = [self._ffmpeg, '-hide_banner', '-loglevel', 'error']
        # HW acceleration for decoding (auto-copies back to system for filter)
        if self._hwaccel in ('cuda', 'd3d11va', 'qsv'):
            cmd.extend(['-hwaccel', self._hwaccel])
        # Exact rational fps avoids rounding errors with the fps filter
        fps_frac = Fraction(n, duration).limit_denominator(1000000)
        cmd.extend([
            '-i', video_path,
            '-vf', f'fps={fps_frac.numerator}/{fps_frac.denominator},scale={width}:{height},tile={n}x1',
            '-q:v', str(quality),
            '-f', 'image2',
            'pipe:1'
        ])
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except Exception:
            pass
        return None

    def get_pregen_strip(self, sid, width, height, quality=4):
        """Get (or generate) a pre-generated thumbnail strip at ~1 fps.

        The strip is cached in memory after first generation, so subsequent
        requests (e.g. after zoom/scroll) are instant — no ffmpeg call.
        For long videos (>1800 frames) the interval is adjusted to cap at 1800.
        """
        with self._lock:
            cached = self._thumbs.get(sid)
            if cached and cached.get('width') == width and cached.get('height') == height:
                return cached['data']

        video_path = self.get_video_path(sid)
        if not video_path or not Path(video_path).exists():
            return None
        duration = self.probe_duration(sid)
        if not duration or duration <= 0:
            return None

        n = max(1, int(duration) + 1)  # one per second, rounded up
        interval = 1
        if n > 1800:
            interval = (n + 1799) // 1800
            n = (n + interval - 1) // interval

        cmd = [self._ffmpeg, '-hide_banner', '-loglevel', 'error']
        if self._hwaccel in ('cuda', 'd3d11va', 'qsv'):
            cmd.extend(['-hwaccel', self._hwaccel])
        fps_val = Fraction(1, interval)
        cmd.extend([
            '-i', video_path,
            '-vf', f'fps={fps_val.numerator}/{fps_val.denominator},scale={width}:{height},tile={n}x1',
            '-q:v', str(quality),
            '-f', 'image2',
            'pipe:1'
        ])
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=180)
            if result.returncode == 0 and result.stdout:
                with self._lock:
                    self._thumbs[sid] = {'data': result.stdout, 'width': width, 'height': height, 'count': n}
                return result.stdout
        except Exception:
            pass
        return None
