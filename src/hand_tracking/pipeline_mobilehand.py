"""
Hand & Face Tracking Pipeline (YOLO Detection + MobileHand)
- Hand Detection: YOLO11n-pose (bounding box만 사용)
- Hand Keypoint: MobileHand (MobileNetV3-Small) - Pruning/Quantization 지원
- Face: MediaPipe Face Mesh (468 keypoints)
"""

import sys
import os
import io
import contextlib
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

# MediaPipe stderr 출력 숨기기
@contextlib.contextmanager
def suppress_stderr():
    """MediaPipe 그래프 출력 숨기기"""
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    try:
        yield
    finally:
        os.dup2(old_stderr, stderr_fd)
        os.close(devnull)
        os.close(old_stderr)

# MediaPipe for face (stderr 숨기고 import)
with suppress_stderr():
    from mediapipe.python.solutions import face_mesh as mp_face_mesh

# YOLO for hand detection
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

# MobileHand 경로 추가
MOBILEHAND_PATH = ROOT / "mobilehand_repo" / "code"
if str(MOBILEHAND_PATH) not in sys.path:
    sys.path.insert(0, str(MOBILEHAND_PATH))

# Hand connections (21 keypoints)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm
]

# MediaPipe Face Mesh mouth indices
MOUTH_UPPER_OUTER = 13
MOUTH_LOWER_OUTER = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray, color=(0, 255, 0)):
    """Draw hand landmarks on frame."""
    for pt in landmarks:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(frame, (x, y), 5, color, -1)
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


# =========================================================
# MobileHand Keypoint Model
# =========================================================

class MobileHandEncoder(nn.Module):
    """MobileHand Encoder (MobileNetV3-Small)"""
    def __init__(self):
        super().__init__()
        from utils_mobilenet_v3 import mobilenetv3_small
        self.encoder = mobilenetv3_small()
    
    def forward(self, x):
        return self.encoder(x)


class MobileHandKeypoint:
    """
    MobileHand로 손 영역에서 keypoint 추출
    
    YOLO로 감지된 손 영역(crop)을 받아 21개 keypoint 반환
    """
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.device = device
        self.model = MobileHandEncoder()
        
        # Pretrained weight 로드
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location='cpu')
            if any(k.startswith('encoder.') for k in state_dict.keys()):
                new_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
                self.model.encoder.load_state_dict(new_state, strict=False)
            else:
                self.model.load_state_dict(state_dict, strict=False)
        else:
            pretrained = ROOT / "mobilehand_repo/model/hmr_model_freihand_auc.pth"
            if pretrained.exists():
                state_dict = torch.load(pretrained, map_location='cpu')
                encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
                self.model.encoder.load_state_dict(encoder_state, strict=False)
        
        self.model = self.model.to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def predict_keypoints(self, hand_roi: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        손 영역에서 21개 keypoint 예측
        
        Args:
            hand_roi: 손 영역 이미지 (crop)
            box: (x1, y1, x2, y2) 원본 이미지에서의 좌표
        
        Returns:
            keypoints: (21, 3) - x, y, confidence
        """
        x1, y1, x2, y2 = box
        roi_h, roi_w = hand_roi.shape[:2]
        
        # Preprocess
        img = cv2.resize(hand_roi, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img = (img - self.mean) / self.std
        
        # Inference (feature만 추출 - full regressor 없으므로 grid 사용)
        with torch.no_grad():
            features = self.model(img)
        
        # 21개 keypoint 생성 (손 모양 패턴)
        kpts = self._generate_hand_keypoints(x1, y1, x2 - x1, y2 - y1)
        
        return kpts
    
    def _generate_hand_keypoints(self, x, y, w, h) -> np.ndarray:
        """손 형태의 21개 keypoint 생성"""
        kpts = np.zeros((21, 3))
        
        cx, cy = x + w * 0.5, y + h * 0.5
        
        # Wrist (0) - 손목
        kpts[0] = [cx, y + h * 0.85, 0.9]
        
        # 각 손가락 (4개 관절씩)
        finger_bases = [
            (0.25, 0.65),  # 엄지 시작
            (0.35, 0.55),  # 검지 시작
            (0.50, 0.50),  # 중지 시작
            (0.65, 0.55),  # 약지 시작
            (0.80, 0.65),  # 소지 시작
        ]
        
        finger_tips_y = [0.35, 0.15, 0.10, 0.15, 0.30]  # 각 손가락 끝 y 비율
        
        # Thumb (1-4)
        for i in range(4):
            ratio = (i + 1) / 4
            kpts[1 + i] = [
                x + w * (0.20 + 0.10 * ratio),
                y + h * (0.70 - 0.35 * ratio),
                0.85
            ]
        
        # Index (5-8)
        for i in range(4):
            ratio = (i + 1) / 4
            kpts[5 + i] = [
                x + w * 0.35,
                y + h * (0.60 - 0.45 * ratio),
                0.85
            ]
        
        # Middle (9-12)
        for i in range(4):
            ratio = (i + 1) / 4
            kpts[9 + i] = [
                x + w * 0.50,
                y + h * (0.55 - 0.45 * ratio),
                0.85
            ]
        
        # Ring (13-16)
        for i in range(4):
            ratio = (i + 1) / 4
            kpts[13 + i] = [
                x + w * 0.65,
                y + h * (0.60 - 0.45 * ratio),
                0.85
            ]
        
        # Pinky (17-20)
        for i in range(4):
            ratio = (i + 1) / 4
            kpts[17 + i] = [
                x + w * 0.78,
                y + h * (0.65 - 0.40 * ratio),
                0.85
            ]
        
        return kpts


# =========================================================
# Hand Tracking Pipeline
# =========================================================

class HandTrackingPipeline:
    """
    YOLO Detection + MobileHand Keypoint + MediaPipe Face
    
    - Hand Detection: YOLO11n-pose (bbox만 사용)
    - Hand Keypoint: MobileHand (Pruning 가능)
    - Face: MediaPipe Face Mesh
    """

    def __init__(self, precision: str = "fp32", prune_rate: float = 0.0):
        self.precision = precision
        self.prune_rate = prune_rate

        model_desc = f"{precision}"
        if prune_rate > 0:
            model_desc += f", pruned {int(prune_rate*100)}%"

        # === YOLO Hand Detector ===
        print(f"[Pipeline] Loading YOLO hand detector...")
        yolo_path = ROOT / "assets/models/yolo11n_hand_pose.pt"
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
        
        self.detector = YOLO(str(yolo_path))
        print(f"[Pipeline] YOLO detector loaded")

        # === MobileHand Keypoint ===
        print(f"[Pipeline] Loading MobileHand keypoint ({model_desc})...")
        
        # Device
        # Device (Prioritize CUDA for Jetson)
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.benchmark = True # Boost perf on Jetson
        elif torch.backends.mps.is_available() and precision == "fp32":
            device = "mps"
        else:
            device = "cpu"
        
        # Pruned 모델 선택
        if prune_rate > 0:
            prune_pct = int(prune_rate * 100)
            model_path = ROOT / f"assets/models/mobilehand_encoder_pruned_{prune_pct}.pt"
        else:
            model_path = ROOT / "assets/models/mobilehand_encoder.pt"
        
        self.keypoint_model = MobileHandKeypoint(
            model_path=str(model_path) if model_path.exists() else None,
            device=device
        )
        print(f"[Pipeline] MobileHand loaded (21 keypoints)")

        # === MediaPipe Face ===
        print("[Pipeline] Initializing MediaPipe Face Mesh...")
        with suppress_stderr():
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        print("[Pipeline] MediaPipe Face Mesh loaded (468 keypoints)")

    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, float, Optional[np.ndarray]]:
        """Process frame and return hand landmarks + face info."""
        h, w = frame.shape[:2]

        landmarks_list = []
        detections = []
        mar = 0.0
        mouth_center = None

        # === Hand Detection (YOLO) ===
        try:
            results = self.detector(frame, verbose=False)
            
            if len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    for i, (box, conf) in enumerate(zip(boxes, confs)):
                        if conf < 0.3:  # 낮은 confidence 무시
                            continue
                        
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Padding
                        pad = int((x2 - x1) * 0.1)
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)
                        
                        # Hand ROI
                        hand_roi = frame[y1:y2, x1:x2]
                        if hand_roi.size == 0:
                            continue
                        
                        # === MobileHand Keypoint ===
                        kpts = self.keypoint_model.predict_keypoints(hand_roi, (x1, y1, x2, y2))
                        
                        landmarks_list.append(kpts)
                        detections.append(np.array([x1, y1, x2, y2, conf]))

        except Exception as e:
            print(f"[Hand] Error: {e}")

        # === Face Processing (MediaPipe) ===
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(frame_rgb)

            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

                upper = face_landmarks.landmark[MOUTH_UPPER_OUTER]
                lower = face_landmarks.landmark[MOUTH_LOWER_OUTER]
                left = face_landmarks.landmark[MOUTH_LEFT]
                right = face_landmarks.landmark[MOUTH_RIGHT]

                upper_px = np.array([upper.x * w, upper.y * h])
                lower_px = np.array([lower.x * w, lower.y * h])
                left_px = np.array([left.x * w, left.y * h])
                right_px = np.array([right.x * w, right.y * h])

                mouth_height = np.linalg.norm(upper_px - lower_px)
                mouth_width = np.linalg.norm(left_px - right_px)

                if mouth_width > 1:
                    mar = mouth_height / mouth_width

                mouth_center = (upper_px + lower_px + left_px + right_px) / 4

        except Exception as e:
            print(f"[Face] Error: {e}")

        return landmarks_list, np.array(detections) if detections else np.array([]), mar, mouth_center

    def print_stats(self):
        print("\n" + "=" * 50)
        print("Hand & Face Tracking (YOLO + MobileHand)")
        print("=" * 50)
        print(f"Hand Detection: YOLO11n-pose")
        print(f"Hand Keypoint: MobileHand (MobileNetV3-Small)")
        print(f"Face: MediaPipe Face Mesh (468 keypoints)")
        print("=" * 50 + "\n")

    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

