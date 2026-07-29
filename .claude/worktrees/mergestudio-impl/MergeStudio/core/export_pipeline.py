"""
Multiprocess batch export pipeline.
Stages: extract_frames -> extract_faces -> merge_frames -> encode_video
"""
import subprocess
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
import traceback
import json
from typing import List, Dict, Optional, Callable, Tuple
from MergeStudio.core.detector.pipeline import detect_and_align
from MergeStudio.core.detector.factory import DetectorFactory, LandmarkFactory, get_device_info
from MergeStudio.core.merger import MergeMaskedFace
from MergeStudio.core.config import MergerConfigMasked


class ExportPipeline:
    """Manages the 4-stage batch export process."""

    def __init__(self, workspace: str, video_path: str, output_path: str,
                 config: dict, cut_segments: List[dict] = None,
                 output_format: str = 'mp4', quality: str = 'high',
                 detector_name: str = 'YOLOv8',
                 landmarker_name: str = 'insightface-2d106det',
                 model_path: str = None):
        self.workspace = Path(workspace)
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.config = config
        self.cfg_obj = MergerConfigMasked(**config) if config else MergerConfigMasked()
        self.cut_segments = cut_segments or []
        self.output_format = output_format
        self.quality = quality
        self.detector_name = detector_name
        self.landmarker_name = landmarker_name
        self.model_path = model_path
        self.temp_dir = Path(tempfile.mkdtemp(prefix='mergestudio_export_'))
        self.status = {
            'stage': 'idle',
            'progress': 0,
            'fps': 0,
            'eta': 0,
            'total_frames': 0,
            'processed_frames': 0,
        }
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self, progress_callback: Optional[Callable] = None):
        """Run the full export pipeline."""
        try:
            self._update_status('extract_frames', progress_callback)
            frames_dir = self.temp_dir / 'frames'
            frames_dir.mkdir(parents=True, exist_ok=True)
            total = self._extract_frames(frames_dir)
            self.status['total_frames'] = total
            if self._cancelled:
                return

            self._update_status('extract_faces', progress_callback)
            faces_dir = self.temp_dir / 'faces'
            faces_dir.mkdir(exist_ok=True)
            faces_json = self.temp_dir / 'faces_data.json'
            self._extract_faces(frames_dir, faces_dir, faces_json)
            if self._cancelled:
                return

            self._update_status('merge_frames', progress_callback)
            merged_dir = self.temp_dir / 'merged'
            merged_dir.mkdir(exist_ok=True)
            self._merge_frames(frames_dir, merged_dir, faces_json)
            if self._cancelled:
                return

            self._update_status('encode_video', progress_callback)
            self._encode_video(merged_dir)
            self._update_status('complete', progress_callback)

        except Exception as e:
            self.status['error'] = str(e)
            self._update_status('error', progress_callback)
            traceback.print_exc()
        finally:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _extract_frames(self, frames_dir: Path) -> int:
        cmd = [
            'ffmpeg', '-i', str(self.video_path),
            '-q:v', '2',
            str(frames_dir / '%05d.jpg')
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return len(list(frames_dir.glob('*.jpg')))

    def _extract_faces(self, frames_dir: Path, faces_dir: Path, faces_json: Path):
        """Detect and align faces from all extracted frames."""
        device = get_device_info()
        detector = DetectorFactory.create(self.detector_name, device)
        landmarker = LandmarkFactory.create(self.landmarker_name, device)

        faces_data = {}
        frame_files = sorted(frames_dir.glob('*.jpg'))
        total = len(frame_files)

        for idx, f in enumerate(frame_files):
            if self._cancelled:
                break

            frame = cv2.imread(str(f))
            if frame is None:
                continue

            try:
                face_list = detect_and_align(detector, landmarker, frame)
                if face_list:
                    faces_data[f.name] = []
                    for face in face_list:
                        faces_data[f.name].append({
                            'rect': face['face_rect'],
                            'landmarks': face['landmarks'].tolist(),
                            'out_size': face['out_size'],
                        })
                    # Save aligned face thumbnails for ArcFace
                    for fi, face in enumerate(face_list):
                        aligned = face.get('aligned_face')
                        if aligned is not None:
                            thumb_path = faces_dir / f"{f.stem}_face{fi}.jpg"
                            cv2.imwrite(str(thumb_path), aligned)
            except Exception:
                pass

            self.status['processed_frames'] = idx + 1
            self.status['progress'] = int((idx + 1) / total * 100)

        # Save faces data for merge stage
        with open(faces_json, 'w') as fp:
            json.dump(faces_data, fp)

    def _merge_frames(self, frames_dir: Path, merged_dir: Path, faces_json: Path):
        """Merge faces back into frames."""
        # Load faces data
        with open(faces_json) as fp:
            faces_data = json.load(fp)

        # Build cfg
        cfg = self.cfg_obj

        # Set predictor_func based on model_path
        predictor_func = None
        if self.model_path:
            try:
                from MergeStudio.core.model_loader import model_loader
                session = model_loader.load_model(self.model_path)
                def predictor(face_img):
                    input_name = session.get_inputs()[0].name
                    input_data = np.transpose(face_img.astype(np.float32) / 255.0,
                                              (2, 0, 1))[None, :, :, :]
                    return session.run(None, {input_name: input_data})[0]
            except Exception:
                pass

        frame_files = sorted(frames_dir.glob('*.jpg'))
        total = len(frame_files)

        for idx, f in enumerate(frame_files):
            if self._cancelled:
                break

            frame = cv2.imread(str(f))
            if frame is None:
                shutil.copy(f, merged_dir / f.name)
                continue

            # Check if this frame has no face
            face_list = faces_data.get(f.name, [])
            if not face_list:
                shutil.copy(f, merged_dir / f.name)
                self.status['processed_frames'] = idx + 1
                continue

            # Merge each face
            out = frame.copy()
            for face_data in face_list:
                landmarks = np.array(face_data['landmarks'], dtype=np.float32)
                face_out, _ = MergeMaskedFace(
                    out, landmarks, cfg, predictor_func)
                out = face_out

            cv2.imwrite(str(merged_dir / f.name), out)
            self.status['processed_frames'] = idx + 1
            self.status['progress'] = int((idx + 1) / total * 100)

    def _encode_video(self, frames_dir: Path):
        if self._cancelled:
            return
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-i', str(frames_dir / '%05d.jpg'),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18' if self.quality == 'high' else '23',
            str(self.output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _update_status(self, stage: str, callback: Optional[Callable] = None):
        self.status['stage'] = stage
        if callback:
            callback(self.status.copy())
