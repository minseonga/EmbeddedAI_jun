# Try ONNX version first (no mmcv dependency), fallback to MMPose version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HAND_ONNX = ROOT / "assets/models/hand_mobilenetv2_256x256.onnx"
FACE_ONNX = ROOT / "assets/models/face_rtmpose_t_256x256.onnx"

if HAND_ONNX.exists() and FACE_ONNX.exists():
    # Use ONNX version
    from .pipeline_onnx import HandTrackingPipeline, draw_landmarks, draw_detections
    print("[hand_tracking] Using ONNX pipeline")
else:
    # Use MMPose version (requires mmcv)
    from .pipeline import HandTrackingPipeline, draw_landmarks, draw_detections
    print("[hand_tracking] Using MMPose pipeline")

__all__ = ["HandTrackingPipeline", "draw_landmarks", "draw_detections"]
