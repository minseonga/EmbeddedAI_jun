"""
Emoji Reactor - RTMPose (Pure PyTorch, no ONNX)

States:
- HANDS_UP      : hand above --raise-thresh
- SMILING       : mouth aspect ratio > --smile-thresh
- AHA           : index finger only up
- CONFUSED      : index finger near mouth
- FRUSTRATED    : two hands up high
- STRAIGHT_FACE : default

Run:
  python app.py --no-gstreamer --camera 0   # PC/Mac
  python app.py                              # Jetson Nano (GStreamer)
"""

import argparse
import os
import sys
import time
import threading
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ASSETS = ROOT / "assets"
EMOJI_DIR = ASSETS / "emojis"
AUDIO_DIR = ASSETS / "audio"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_tracking import HandTrackingPipeline, draw_landmarks

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480


def load_emojis():
    """Load emoji images."""
    file_map = {
        "SMILING": "smile.jpg",
        "STRAIGHT_FACE": "plain.png",
        "HANDS_UP": "air.jpg",
        "AHA": "aha.png",
        "CONFUSED": "confused.png",
        "FRUSTRATED": "frustrated.png",
    }

    def load_set(prefix):
        loaded = {}
        for state, filename in file_map.items():
            name = f"{prefix}{filename}" if prefix else filename
            path = EMOJI_DIR / name
            img = cv2.imread(str(path))
            if img is not None:
                loaded[state] = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
        return loaded

    return {
        "default": load_set(""),
        "monkey": load_set("monkey_"),
        "blank": np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    }


def is_hand_up(landmarks, frame_h, thresh):
    """Check if wrist is above threshold."""
    return landmarks[0, 1] / frame_h < thresh


def is_index_up_only(lm, frame_h):
    """Index finger up, others down."""
    idx_tip = lm[8, 1]
    idx_pip = lm[6, 1]
    if idx_tip >= idx_pip - 0.003 * frame_h:
        return False
    other_tips = [lm[4, 1], lm[12, 1], lm[16, 1], lm[20, 1]]
    return idx_tip < min(other_tips) - 0.006 * frame_h


def is_confused(lm, mouth_center, frame_h, thresh):
    """Index finger near mouth."""
    if mouth_center is None:
        return False
    dist = np.linalg.norm(lm[8, :2] - mouth_center)
    return dist / frame_h < thresh


def count_hands_up(landmarks_list, frame_h, thresh):
    """Count hands with wrist above threshold."""
    return sum(1 for lm in landmarks_list if lm[0, 1] / frame_h < thresh)


class BackgroundMusic(threading.Thread):
    """Background music player."""
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self._running = True
        self._proc = None

    def stop(self):
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
            except:
                pass

    def run(self):
        if not os.path.isfile(self.path):
            return
        cmd = None
        if sys.platform == "darwin" and shutil.which("afplay"):
            cmd = ["afplay", self.path]
        elif shutil.which("ffplay"):
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.path]
        if not cmd:
            return

        while self._running:
            try:
                self._proc = subprocess.Popen(cmd)
                self._proc.wait()
            except:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--raise-thresh', type=float, default=0.25)
    parser.add_argument('--smile-thresh', type=float, default=0.35)
    parser.add_argument('--confused-thresh', type=float, default=0.08)
    parser.add_argument('--frustrated-thresh', type=float, default=0.45)
    parser.add_argument('--no-mirror', action='store_true')
    parser.add_argument('--no-music', action='store_true')
    parser.add_argument('--no-gstreamer', action='store_true')
    args = parser.parse_args()

    emoji_sets = load_emojis()

    # Music
    music = None
    if not args.no_music:
        music = BackgroundMusic(str(AUDIO_DIR / "yessir.mp3"))
        music.start()

    # Camera (GStreamer for Jetson Nano)
    if not args.no_gstreamer:
        pipeline = (
            "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        print("Opening camera (GStreamer)...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        print(f"Opening camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow('Reactor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Reactor', WINDOW_WIDTH * 2, WINDOW_HEIGHT)

    # RTMPose pipeline (handles both hands and face)
    print("[Init] RTMPose (hand + face)...")
    pipeline = HandTrackingPipeline(precision=args.precision)
    pipeline.print_stats()

    fps_hist = []
    mode = "default"

    print("\n[Ready] m=monkey, e=emoji, q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not args.no_mirror:
            frame = frame[:, ::-1].copy()
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        h, w = frame.shape[:2]

        # RTMPose inference (returns hands + face info)
        t0 = time.time()
        landmarks, detections, mar, mouth_center = pipeline.process_frame(frame)
        fps = 1.0 / (time.time() - t0 + 1e-6)
        fps_hist = (fps_hist + [fps])[-30:]

        # State decision
        if mode == "monkey":
            state = "STRAIGHT_FACE"
            if count_hands_up(landmarks, h, args.frustrated_thresh) >= 2:
                state = "FRUSTRATED"
            elif any(is_confused(lm, mouth_center, h, args.confused_thresh) for lm in landmarks):
                state = "CONFUSED"
            elif any(is_index_up_only(lm, h) for lm in landmarks):
                state = "AHA"
        else:
            state = "STRAIGHT_FACE"
            if any(is_hand_up(lm, h, args.raise_thresh) for lm in landmarks):
                state = "HANDS_UP"
            elif mar > args.smile_thresh:
                state = "SMILING"

        emoji = emoji_sets.get(mode, {}).get(state, emoji_sets["blank"])
        emoji_char = {"HANDS_UP": "🙌", "SMILING": "😊", "STRAIGHT_FACE": "😐",
                      "AHA": "💡", "CONFUSED": "🤔", "FRUSTRATED": "😤"}.get(state, "❓")

        # Draw
        vis = frame.copy()
        for lm in landmarks:
            draw_landmarks(vis, lm)

        cv2.putText(vis, f"{state} {emoji_char}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"FPS {np.mean(fps_hist):.0f} | {mode}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Reactor', np.hstack((vis, emoji)))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('m'):
            mode = "monkey"
        elif key == ord('e'):
            mode = "default"

    cap.release()
    cv2.destroyAllWindows()
    if music:
        music.stop()


if __name__ == "__main__":
    main()
