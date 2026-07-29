"""
Face detector and landmarker factory.
Reference: Extractor/Extractor.py
"""
from modelhub.onnx import BlazeFace, CenterFace, S3FD, YoloV5Face
from modelhub.onnx.YoloV8Face import YoloV8Face
from modelhub.onnx import InsightFace2D106


class DetectorFactory:
    DETECTORS = {
        'BlazeFace': BlazeFace,
        'CenterFace': CenterFace,
        'S3FD': S3FD,
        'YoloV5Face': YoloV5Face,
        'YoloV8Face': YoloV8Face,
    }

    @classmethod
    def create(cls, name: str, device_info):
        detector_class = cls.DETECTORS.get(name)
        if detector_class is None:
            raise ValueError("Unsupported detector: " + name)
        return detector_class(device_info)


class LandmarkFactory:
    LANDMARKS = {
        'insightface-2d106det': InsightFace2D106,
    }

    @classmethod
    def create(cls, name: str, device_info):
        landmark_class = cls.LANDMARKS.get(name)
        if landmark_class is None:
            raise ValueError("Unsupported landmarker: " + name)
        return landmark_class(device_info)


def get_device_info():
    import onnxruntime
    providers = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        return ('CUDAExecutionProvider', {})
    elif 'DmlExecutionProvider' in providers:
        return ('DmlExecutionProvider', {})
    else:
        return ('CPUExecutionProvider', {})
