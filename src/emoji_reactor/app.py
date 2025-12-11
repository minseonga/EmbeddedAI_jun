"""
Emoji Reactor - Hand & Face Tracking

States:
- HANDS_UP      : hand above --raise-thresh
- SMILING       : mouth aspect ratio > --smile-thresh
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
    }

    loaded = {}
    for state, filename in file_map.items():
        path = EMOJI_DIR / filename
        img = cv2.imread(str(path))
        if img is not None:
            loaded[state] = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
        else:
            print(f"[Warning] Could not load {path}")

    return loaded


def is_hand_up(landmarks, frame_h, thresh):
    """Check if wrist is above threshold."""
    return landmarks[0, 1] / frame_h < thresh




class BackgroundMusic(threading.Thread):
    """Background music player that loops."""
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
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self._proc.wait()
            except:
                break


def play_sound(sound_name):
    """Play sound effect for emoji state change."""
    sound_path = AUDIO_DIR / f"{sound_name}.mp3"
    if not os.path.isfile(sound_path):
        return

    cmd = None
    if sys.platform == "darwin" and shutil.which("afplay"):
        cmd = ["afplay", str(sound_path)]
    elif shutil.which("ffplay"):
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(sound_path)]

    if cmd:
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp32')
    parser.add_argument('--prune', type=float, default=0.0, help='Pruning rate (0.0-0.7)')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--raise-thresh', type=float, default=0.25)
    parser.add_argument('--smile-thresh', type=float, default=0.35)
    parser.add_argument('--no-mirror', action='store_true')
    parser.add_argument('--no-gstreamer', action='store_true')
    args = parser.parse_args()

    emojis = load_emojis()
    blank_emoji = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

    # Background music
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
    pipeline = HandTrackingPipeline(precision=args.precision, prune_rate=args.prune)
    pipeline.print_stats()

    fps_hist = []
    prev_state = None

    print("\n[Ready] Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not args.no_mirror:
            frame = frame[:, ::-1].copy()
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        h, w = frame.shape[:2]

        # Hand & Face inference
        t0 = time.time()
        landmarks, detections, mar, mouth_center = pipeline.process_frame(frame)
        fps = 1.0 / (time.time() - t0 + 1e-6)
        fps_hist = (fps_hist + [fps])[-30:]

        # State decision
        state = "STRAIGHT_FACE"
        if any(is_hand_up(lm, h, args.raise_thresh) for lm in landmarks):
            state = "HANDS_UP"
        elif mar > args.smile_thresh:
            state = "SMILING"

        # Play sound when state changes
        if state != prev_state and prev_state is not None:
            # Sound files should be named: HANDS_UP.mp3, SMILING.mp3, STRAIGHT_FACE.mp3
            play_sound(state)
        prev_state = state

        # Get emoji image
        emoji = emojis.get(state, blank_emoji)
        emoji_char = {"HANDS_UP": "🙌", "SMILING": "😊", "STRAIGHT_FACE": "😐"}.get(state, "❓")

        # Draw
        vis = frame.copy()
        for lm in landmarks:
            draw_landmarks(vis, lm)

        cv2.putText(vis, f"{state} {emoji_char}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"FPS {np.mean(fps_hist):.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Reactor', np.hstack((vis, emoji)))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    music.stop()


if __name__ == "__main__":
    main()
