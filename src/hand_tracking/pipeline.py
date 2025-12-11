"""
Hand & Face Tracking Pipeline
- Hand: YOLO11n-Pose (21 keypoints) from chrismuntean/YOLO11n-pose-hands
- Face: MediaPipe Face Mesh (468 keypoints, focus on mouth for MAR)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO

# Import MediaPipe solutions directly to avoid TensorFlow dependency issues
from mediapipe.python.solutions import face_mesh as mp_face_mesh

ROOT = Path(__file__).resolve().parents[2]

# Hand connections (21 keypoints) - MediaPipe format
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm
]

# MediaPipe Face Mesh mouth indices (outer lips)
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


class HandTrackingPipeline:
    """Hand & Face tracking using YOLOv8 + MediaPipe."""

    def __init__(self, precision: str = "fp32", prune_rate: float = 0.0):
        self.precision = precision
        self.prune_rate = prune_rate

        model_desc = f"{precision}"
        if prune_rate > 0:
            model_desc += f", pruned {int(prune_rate*100)}%"

        print(f"[Pipeline] Loading YOLO11n-Pose hand model ({model_desc})...")

        # Select model based on precision and pruning
        if prune_rate > 0:
            # Use pruned model
            prune_pct = int(prune_rate * 100)
            model_path = ROOT / f"assets/models/yolo11n_hand_pose_pruned_{prune_pct}.pt"
            if not model_path.exists():
                print(f"[Warning] Pruned model not found, using base model")
                model_path = ROOT / "assets/models/yolo11n_hand_pose.pt"
        elif precision == "int8":
            model_path = ROOT / "assets/models/yolo11n_hand_pose_int8.engine"
            if not model_path.exists():
                print(f"[Warning] INT8 TensorRT engine not found, using FP32")
                model_path = ROOT / "assets/models/yolo11n_hand_pose.pt"
        elif precision == "fp16":
            model_path = ROOT / "assets/models/yolo11n_hand_pose_fp16.engine"
            if not model_path.exists():
                print(f"[Warning] FP16 TensorRT engine not found, using FP32")
                model_path = ROOT / "assets/models/yolo11n_hand_pose.pt"
        else:
            model_path = ROOT / "assets/models/yolo11n_hand_pose.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.hand_model = YOLO(str(model_path))
        print(f"[Pipeline] YOLO11n-Pose loaded (21 keypoints per hand, {model_desc})")

        print("[Pipeline] Initializing MediaPipe Face Mesh...")
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

        # === Hand Processing (YOLOv8) ===
        try:
            # Run YOLOv8 pose detection
            results = self.hand_model(frame, verbose=False)

            if len(results) > 0:
                result = results[0]

                # Check if keypoints are detected
                if result.keypoints is not None and len(result.keypoints) > 0:
                    # Extract keypoints for each detected hand
                    keypoints_data = result.keypoints.data  # Shape: [num_hands, 21, 3] (x, y, conf)
                    boxes = result.boxes.xyxy  # Bounding boxes

                    for i, kpts in enumerate(keypoints_data):
                        # Convert from tensor to numpy
                        kpts_np = kpts.cpu().numpy()  # Shape: [21, 3]

                        # Filter out low confidence keypoints (optional)
                        # For now, keep all keypoints
                        landmarks_list.append(kpts_np)

                        # Get bounding box
                        if i < len(boxes):
                            box = boxes[i].cpu().numpy()  # [x1, y1, x2, y2]
                            mean_conf = kpts_np[:, 2].mean()
                            detections.append(np.array([box[0], box[1], box[2], box[3], mean_conf]))

        except Exception as e:
            print(f"[Hand] Error: {e}")

        # === Face Processing (MediaPipe) ===
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(frame_rgb)

            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

                # Extract mouth landmarks for MAR calculation
                upper = face_landmarks.landmark[MOUTH_UPPER_OUTER]
                lower = face_landmarks.landmark[MOUTH_LOWER_OUTER]
                left = face_landmarks.landmark[MOUTH_LEFT]
                right = face_landmarks.landmark[MOUTH_RIGHT]

                # Convert to pixel coordinates
                upper_px = np.array([upper.x * w, upper.y * h])
                lower_px = np.array([lower.x * w, lower.y * h])
                left_px = np.array([left.x * w, left.y * h])
                right_px = np.array([right.x * w, right.y * h])

                # Calculate MAR
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
        print("Hand & Face Tracking")
        print("=" * 50)
        print(f"Hand: YOLO11n-Pose (21 keypoints)")
        print(f"Face: MediaPipe Face Mesh (468 keypoints)")
        print("=" * 50 + "\n")

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
