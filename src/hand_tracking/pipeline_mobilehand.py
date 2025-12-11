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
# MobileHand Full Model (Encoder + Regressor + MANO)
# =========================================================

class MobileHandHMR:
    """
    MobileHand Full Pipeline (HMR)
    
    - Encoder: MobileNetV3-Small → 576 features
    - Regressor: 576 → MANO params (39)
    - MANO: params → 21 keypoints
    """
    def __init__(self, model_path: str = None, device: str = 'cpu', dataset: str = 'freihand'):
        self.device = device
        
        # Fix import path (Ensure local repo code is in path)
        import sys
        import os
        repo_path = str(ROOT / "mobilehand_repo/code")
        if repo_path not in sys.path:
            sys.path.append(repo_path)
            
        # HMR 모델 import
        try:
            from utils_neural_network import HMR
        except ImportError as e:
            print(f"[Error] Failed to import HMR from utils_neural_network")
            raise e
        
        # Argparse 모사 (HMR init에 필요)
        class Args:
            def __init__(self, data):
                self.data = data
        
        arg = Args(dataset)
        self.model = HMR(arg)
        
        # Pretrained weight 로드
        pretrained = ROOT / f"mobilehand_repo/model/hmr_model_{dataset}_auc.pth"
        if pretrained.exists():
            state_dict = torch.load(pretrained, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[MobileHand] Loaded HMR model: {pretrained.name}")
        else:
            print(f"[MobileHand] Warning: Pretrained not found at {pretrained}")
        
        # Pruned encoder 로드 (있으면 Encoder만 교체)
        if model_path and Path(model_path).exists():
            print(f"[MobileHand] Loading pruned encoder from {Path(model_path).name}...")
            pruned_state = torch.load(model_path, map_location='cpu')
            
            # Encoder state extraction
            encoder_state = {}
            for k, v in pruned_state.items():
                if k.startswith('encoder.'):
                    encoder_state[k.replace('encoder.', '')] = v
                elif not k.startswith('regressor.') and not k.startswith('mano.'):
                     # If model was saved as just encoder
                     encoder_state[k] = v
            
            if encoder_state:
                self.model.encoder.load_state_dict(encoder_state, strict=False)
                print(f"[MobileHand] Pruned encoder loaded successfully")
        
        self.model = self.model.to(device).eval()
    
    def predict_keypoints(self, hand_roi: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        손 영역에서 21개 keypoint 예측 (실제 손가락 움직임 반영)
        """
        x1, y1, x2, y2 = box
        box_w, box_h = x2 - x1, y2 - y1
        roi_h, roi_w = hand_roi.shape[:2]
        
        # === Aspect ratio 유지하면서 224x224 정사각형으로 변환 ===
        max_side = max(roi_h, roi_w)
        square_img = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        
        pad_x = (max_side - roi_w) // 2
        pad_y = (max_side - roi_h) // 2
        square_img[pad_y:pad_y + roi_h, pad_x:pad_x + roi_w] = hand_roi
        
        img = cv2.resize(square_img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.as_tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1) / 255.0
        
        # Inference
        with torch.no_grad():
            img_input = img.to(self.device).unsqueeze(0)
            # HMR returns: keypt, joint, vert, ang, params
            keypt, joint, vert, ang, params = self.model(img_input)
        
        # keypt: [1, 21, 2] - 224x224 기준 좌표
        keypt_np = keypt[0].cpu().numpy()
        
        # === 좌표 역변환 (224x224 -> 원본) ===
        scale = max_side / 224.0
        kpts = np.zeros((21, 3))
        
        for i in range(21):
            kx_scaled = keypt_np[i, 0] * scale
            ky_scaled = keypt_np[i, 1] * scale
            
            kx_unpadded = kx_scaled - pad_x
            ky_unpadded = ky_scaled - pad_y
            
            # ROI 내 비율 -> 원본 좌표
            # (Note: roi_w, roi_h could be 0 safely handled by detection check outside)
            kx = (kx_unpadded / roi_w) * box_w + x1
            ky = (ky_unpadded / roi_h) * box_h + y1
            
            # Clipping
            kx = np.clip(kx, x1, x2)
            ky = np.clip(ky, y1, y2)
            
            kpts[i] = [kx, ky, 0.9] # Confidence dummy 0.9
        
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

        # === YOLO Hand Detector (Disabled) ===
        self.detector = None
        print(f"[Pipeline] YOLO detector disabled (using Center Crop)")

        # === MobileHand Keypoint (Full HMR + MANO) ===
        print(f"[Pipeline] Loading MobileHand HMR ({model_desc})...")
        
        # Device
        # Device (Prioritize CUDA for Jetson)
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.benchmark = True # Boost perf on Jetson
        elif torch.backends.mps.is_available() and precision == "fp32":
            device = "mps"
        else:
            device = "cpu"
        
        # Pruned 모델 선택 (Encoder Weight만 교체)
        if prune_rate > 0:
            prune_pct = int(prune_rate * 100)
            model_path = ROOT / f"assets/models/mobilehand_encoder_pruned_{prune_pct}.pt"
        else:
            # Full HMR uses internal default if None
            model_path = None 
        
        self.keypoint_model = MobileHandHMR(
            model_path=str(model_path) if model_path and model_path.exists() else None,
            device=device
        )
        print(f"[Pipeline] MobileHand HMR loaded (21 keypoints from MANO)")

        # === MediaPipe Face (Disabled for Performance) ===
        self.face_mesh = None
        print("[Pipeline] MediaPipe Face Mesh disabled for performance.")

    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, float, Optional[np.ndarray]]:
        """Process frame and return hand landmarks + face info."""
        h, w = frame.shape[:2]

        landmarks_list = []
        detections = []
        mar = 0.0
        mouth_center = None

        # === Hand Detection (Center Crop Fallback) ===
        # YOLO 제거됨 -> 화면 중앙을 ROI로 가정
        
        # Center Crop (60% of screen)
        crop_size = min(h, w) * 0.6
        cx, cy = w // 2, h // 2
        
        x1 = int(cx - crop_size / 2)
        y1 = int(cy - crop_size / 2)
        x2 = int(cx + crop_size / 2)
        y2 = int(cy + crop_size / 2)
        
        # Clipping
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        
        hand_roi = frame[y1:y2, x1:x2]
        
        # [Debug] Show Hand ROI
        cv2.imshow("Hand ROI Input", hand_roi)
        
        if hand_roi.size > 0:
            # === MobileHand Keypoint ===
            kpts = self.keypoint_model.predict_keypoints(hand_roi, (x1, y1, x2, y2))
            
            # [Debug] Print min/max of keypoints relative to ROI
            print(f"[Debug] Kpts Range: X({kpts[:,0].min():.1f}~{kpts[:,0].max():.1f}), Y({kpts[:,1].min():.1f}~{kpts[:,1].max():.1f})")
            
            landmarks_list.append(kpts)
            # Fake detection box with high confidence
            detections.append(np.array([x1, y1, x2, y2, 0.99]))
            
            # [Debug] Draw ROI on main frame (Blue box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Center ROI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # === Face Processing (Disabled) ===
        return landmarks_list, np.array(detections) if detections else np.array([]), 0.0, None

    def print_stats(self):
        print("\n" + "=" * 50)
        print("Hand & Face Tracking (YOLO + MobileHand)")
        print("=" * 50)
        print(f"Hand Detection: YOLO11n-pose")
        print(f"Hand Keypoint: MobileHand (MobileNetV3-Small)")
        print(f"Face: MediaPipe Face Mesh (468 keypoints)")
        print("=" * 50 + "\n")

    def __del__(self):
        if getattr(self, 'face_mesh', None):
            self.face_mesh.close()

