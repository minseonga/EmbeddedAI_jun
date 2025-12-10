"""
Hand & Face Tracking Pipeline using MMPose
- Hand: MobileNetV2 + Heatmap (21 keypoints)
- Face: RTMPose-T + SimCC (106 keypoints)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional

ROOT = Path(__file__).resolve().parents[2]
MMPOSE_ROOT = ROOT / "mmpose"

# Model paths
HAND_MODEL_PATH = ROOT / "assets/models/mobilenetv2_coco_wholebody_hand_256x256-06b8c877_20210909.pth"
FACE_MODEL_PATH = ROOT / "assets/models/rtmpose-t_simcc-face6_pt-in1k_120e-256x256-df79d9a5_20230529.pth"

# Config paths
HAND_CONFIG = MMPOSE_ROOT / "configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_mobilenetv2_8xb32-210e_coco-wholebody-hand-256x256.py"
FACE_CONFIG = MMPOSE_ROOT / "configs/face_2d_keypoint/rtmpose/face6/rtmpose-t_8xb256-120e_face6-256x256.py"

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
    """Hand & Face tracking using MMPose."""

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.input_size = (256, 256)

        # Normalization
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        print("[Pipeline] Loading MMPose models...")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Import mmpose
        from mmpose.apis import init_model

        # Load hand model
        self.hand_model = init_model(
            str(HAND_CONFIG),
            str(HAND_MODEL_PATH),
            device=str(self.device)
        )
        self.hand_model.eval()
        print("[Pipeline] Hand model loaded (21 keypoints)")

        # Load face model
        self.face_model = init_model(
            str(FACE_CONFIG),
            str(FACE_MODEL_PATH),
            device=str(self.device)
        )
        self.face_model.eval()
        print("[Pipeline] Face model loaded (106 keypoints)")

        print(f"[Pipeline] Device: {self.device}")

    def _preprocess(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, dict]:
        """Preprocess frame for model input."""
        h, w = frame.shape[:2]

        if bbox is not None:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            # Expand bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            size = max(bw, bh) * 1.25
            x1 = int(max(0, cx - size / 2))
            y1 = int(max(0, cy - size / 2))
            x2 = int(min(w, cx + size / 2))
            y2 = int(min(h, cy + size / 2))
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame
            x1, y1, x2, y2 = 0, 0, w, h

        # Resize
        img = cv2.resize(crop, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std

        # To tensor
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

        meta = {
            'bbox': np.array([x1, y1, x2, y2]),
            'scale_x': (x2 - x1) / self.input_size[0],
            'scale_y': (y2 - y1) / self.input_size[1],
        }

        return tensor.to(self.device), meta

    def _decode_heatmap(self, heatmaps: torch.Tensor, meta: dict) -> np.ndarray:
        """Decode heatmaps to keypoints."""
        B, K, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.view(B, K, -1)

        idx = heatmaps_flat.argmax(dim=2)
        conf = torch.sigmoid(heatmaps_flat.max(dim=2)[0])

        x = (idx % W).float() / W * self.input_size[0]
        y = (idx // W).float() / H * self.input_size[1]

        # Scale back to original frame
        x = x * meta['scale_x'] + meta['bbox'][0]
        y = y * meta['scale_y'] + meta['bbox'][1]

        keypoints = torch.stack([x, y, conf], dim=2)
        return keypoints[0].cpu().numpy()

    def _decode_simcc(self, pred_x: torch.Tensor, pred_y: torch.Tensor, meta: dict) -> np.ndarray:
        """Decode SimCC predictions to keypoints."""
        # pred_x: [1, 106, 512], pred_y: [1, 106, 512]
        x_locs = pred_x.argmax(dim=2).float() / 2.0  # simcc_split_ratio=2.0
        y_locs = pred_y.argmax(dim=2).float() / 2.0

        # Confidence
        x_prob = torch.softmax(pred_x, dim=2)
        y_prob = torch.softmax(pred_y, dim=2)
        conf = (x_prob.max(dim=2)[0] + y_prob.max(dim=2)[0]) / 2

        # Scale back
        x_locs = x_locs * meta['scale_x'] + meta['bbox'][0]
        y_locs = y_locs * meta['scale_y'] + meta['bbox'][1]

        keypoints = torch.stack([x_locs, y_locs, conf], dim=2)
        return keypoints[0].cpu().numpy()

    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, float, Optional[np.ndarray]]:
        """Process frame and return hand landmarks + face info."""
        h, w = frame.shape[:2]

        landmarks_list = []
        detections = []
        mar = 0.0
        mouth_center = None

        # Full frame bbox
        full_bbox = np.array([0, 0, w, h, 1.0])

        # === Hand Processing ===
        try:
            tensor, meta = self._preprocess(frame, full_bbox)
            with torch.no_grad():
                # Use model's forward
                feats = self.hand_model.extract_feat(tensor)
                heatmaps = self.hand_model.head.forward(feats)

                keypoints = self._decode_heatmap(heatmaps, meta)
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
            tensor, meta = self._preprocess(frame, full_bbox)
            with torch.no_grad():
                feats = self.face_model.extract_feat(tensor)
                pred_x, pred_y = self.face_model.head.forward(feats)

                keypoints = self._decode_simcc(pred_x, pred_y, meta)
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

    def export_onnx(self, output_dir: Path = None):
        """Export models to ONNX format."""
        if output_dir is None:
            output_dir = ROOT / "assets/models"

        output_dir.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(1, 3, 256, 256).to(self.device)

        # Export hand model
        hand_onnx_path = output_dir / "hand_mobilenetv2_256x256.onnx"
        print(f"[Export] Exporting hand model to {hand_onnx_path}")
        torch.onnx.export(
            self.hand_model,
            dummy_input,
            str(hand_onnx_path),
            input_names=['input'],
            output_names=['heatmaps'],
            dynamic_axes={'input': {0: 'batch'}, 'heatmaps': {0: 'batch'}},
            opset_version=11
        )
        print(f"[Export] Hand model exported")

        # Export face model
        face_onnx_path = output_dir / "face_rtmpose_t_256x256.onnx"
        print(f"[Export] Exporting face model to {face_onnx_path}")
        torch.onnx.export(
            self.face_model,
            dummy_input,
            str(face_onnx_path),
            input_names=['input'],
            output_names=['pred_x', 'pred_y'],
            dynamic_axes={'input': {0: 'batch'}, 'pred_x': {0: 'batch'}, 'pred_y': {0: 'batch'}},
            opset_version=11
        )
        print(f"[Export] Face model exported")

        return hand_onnx_path, face_onnx_path

    def print_stats(self):
        print("\n" + "=" * 50)
        print("Hand & Face Tracking - MMPose")
        print("=" * 50)
        print(f"Hand: MobileNetV2 + Heatmap (21 pts)")
        print(f"Face: RTMPose-T + SimCC (106 pts)")
        print(f"Device: {self.device}")
        print("=" * 50 + "\n")
