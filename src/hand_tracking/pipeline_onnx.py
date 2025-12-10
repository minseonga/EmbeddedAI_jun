"""
Hand & Face Tracking Pipeline using ONNX models
(Use after export_onnx.py)

- Hand: MobileNetV2 + Heatmap (21 keypoints)
- Face: RTMPose-T + SimCC (106 keypoints)
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Tuple, Optional

ROOT = Path(__file__).resolve().parents[2]

# ONNX model paths
HAND_ONNX_PATH = ROOT / "assets/models/hand_mobilenetv2_256x256.onnx"
FACE_ONNX_PATH = ROOT / "assets/models/face_rtmpose_t_256x256.onnx"

# Hand connections (21 keypoints)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# Face106 mouth indices
MOUTH_UPPER_OUTER = 87
MOUTH_LOWER_OUTER = 93
MOUTH_LEFT = 84
MOUTH_RIGHT = 90


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray, color=(0, 255, 0)):
    """Draw hand landmarks on frame."""
    for pt in landmarks:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), 3, color, -1)
    for i, j in HAND_CONNECTIONS:
        if i < len(landmarks) and j < len(landmarks):
            pt1 = (int(landmarks[i, 0]), int(landmarks[i, 1]))
            pt2 = (int(landmarks[j, 0]), int(landmarks[j, 1]))
            cv2.line(frame, pt1, pt2, color, 2)


def draw_detections(frame: np.ndarray, detections: np.ndarray, color=(255, 0, 0)):
    """Draw detection boxes on frame."""
    for det in detections:
        x1, y1, x2, y2 = det[:4].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


class HandTrackingPipeline:
    """Hand & Face tracking using ONNX models."""

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.input_size = (256, 256)

        # Normalization (ImageNet)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        print("[Pipeline] Loading ONNX models...")

        # Check if models exist
        if not HAND_ONNX_PATH.exists():
            raise FileNotFoundError(f"Hand ONNX model not found: {HAND_ONNX_PATH}\nRun export_onnx.py first!")
        if not FACE_ONNX_PATH.exists():
            raise FileNotFoundError(f"Face ONNX model not found: {FACE_ONNX_PATH}\nRun export_onnx.py first!")

        # Load ONNX models
        self.hand_session = ort.InferenceSession(str(HAND_ONNX_PATH))
        self.hand_input_name = self.hand_session.get_inputs()[0].name
        print(f"[Pipeline] Hand model loaded: {HAND_ONNX_PATH.name}")

        self.face_session = ort.InferenceSession(str(FACE_ONNX_PATH))
        self.face_input_name = self.face_session.get_inputs()[0].name
        print(f"[Pipeline] Face model loaded: {FACE_ONNX_PATH.name}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for ONNX model."""
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std

        # HWC -> NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32)

    def _decode_heatmap(self, heatmaps: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        """Decode heatmaps to keypoints."""
        B, K, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.reshape(B, K, -1)

        idx = np.argmax(heatmaps_flat, axis=2)
        conf = 1.0 / (1.0 + np.exp(-np.max(heatmaps_flat, axis=2)))  # sigmoid

        x = (idx % W).astype(np.float32) / W * orig_w
        y = (idx // W).astype(np.float32) / H * orig_h

        keypoints = np.stack([x, y, conf], axis=2)
        return keypoints[0]

    def _decode_simcc(self, pred_x: np.ndarray, pred_y: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        """Decode SimCC predictions to keypoints."""
        # pred_x: [1, 106, 512], pred_y: [1, 106, 512]
        x_locs = np.argmax(pred_x, axis=2).astype(np.float32) / 2.0  # simcc_split_ratio=2.0
        y_locs = np.argmax(pred_y, axis=2).astype(np.float32) / 2.0

        # Scale to original size
        x_locs = x_locs / self.input_size[0] * orig_w
        y_locs = y_locs / self.input_size[1] * orig_h

        # Confidence (softmax max)
        x_exp = np.exp(pred_x - np.max(pred_x, axis=2, keepdims=True))
        x_prob = x_exp / np.sum(x_exp, axis=2, keepdims=True)
        y_exp = np.exp(pred_y - np.max(pred_y, axis=2, keepdims=True))
        y_prob = y_exp / np.sum(y_exp, axis=2, keepdims=True)

        conf = (np.max(x_prob, axis=2) + np.max(y_prob, axis=2)) / 2

        keypoints = np.stack([x_locs, y_locs, conf], axis=2)
        return keypoints[0]

    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, float, Optional[np.ndarray]]:
        """Process frame and return hand landmarks + face info."""
        h, w = frame.shape[:2]

        landmarks_list = []
        detections = []
        mar = 0.0
        mouth_center = None

        # Preprocess
        input_tensor = self._preprocess(frame)

        # === Hand Processing ===
        try:
            outputs = self.hand_session.run(None, {self.hand_input_name: input_tensor})
            heatmaps = outputs[0]

            keypoints = self._decode_heatmap(heatmaps, h, w)
            mean_conf = keypoints[:, 2].mean()

            if mean_conf > 0.3:
                landmarks_list.append(keypoints)
                x_min, x_max = keypoints[:, 0].min(), keypoints[:, 0].max()
                y_min, y_max = keypoints[:, 1].min(), keypoints[:, 1].max()
                detections.append(np.array([x_min, y_min, x_max, y_max, mean_conf]))

        except Exception as e:
            print(f"[Hand] Error: {e}")

        # === Face Processing ===
        try:
            outputs = self.face_session.run(None, {self.face_input_name: input_tensor})
            pred_x, pred_y = outputs[0], outputs[1]

            keypoints = self._decode_simcc(pred_x, pred_y, h, w)
            mean_conf = keypoints[:, 2].mean()

            if mean_conf > 0.1:
                # Calculate MAR
                upper = keypoints[MOUTH_UPPER_OUTER]
                lower = keypoints[MOUTH_LOWER_OUTER]
                left = keypoints[MOUTH_LEFT]
                right = keypoints[MOUTH_RIGHT]

                mouth_height = np.linalg.norm(upper[:2] - lower[:2])
                mouth_width = np.linalg.norm(left[:2] - right[:2])

                if mouth_width > 1:
                    mar = mouth_height / mouth_width

                mouth_center = (upper[:2] + lower[:2] + left[:2] + right[:2]) / 4

        except Exception as e:
            print(f"[Face] Error: {e}")

        return landmarks_list, np.array(detections) if detections else np.array([]), mar, mouth_center

    def print_stats(self):
        print("\n" + "=" * 50)
        print("Hand & Face Tracking - ONNX")
        print("=" * 50)
        print(f"Hand: {HAND_ONNX_PATH.name}")
        print(f"Face: {FACE_ONNX_PATH.name}")
        print("=" * 50 + "\n")
