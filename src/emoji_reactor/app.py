"""
Gesture Camera - Hand Tracking
 
States:
- CAMERA MODE:
  - FIST: Capture photo
  - SWIPE: Enter Gallery
- GALLERY MODE:
  - SWIPE LEFT/RIGHT: Navigate photos
  - FIST: Back to Camera
 
Run:
  python app.py --camera 0
"""
 
import argparse
import os
import sys
import time
import cv2
import numpy as np
import datetime
from pathlib import Path
 
# Setup Paths
ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ASSETS = ROOT / "assets"
CAPTURE_DIR = ASSETS / "captures"
 
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
 
# from hand_tracking import HandTrackingPipeline, draw_landmarks # Loaded dynamically
from hand_tracking.gesture import GestureDetector
 
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
 
def open_default_camera(cam_id=0):
    """Open a plain OpenCV camera."""
    backend = getattr(cv2, "CAP_AVFOUNDATION", None) if sys.platform == "darwin" else getattr(cv2, "CAP_V4L2", None)
    cap = cv2.VideoCapture(cam_id, backend) if backend else cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    return cap
 
def get_camera_pipeline(cam_id=0, no_gstreamer=False):
    cap = None
    if not no_gstreamer:
        pipeline = (
            "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        print("Opening camera (GStreamer)...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("GStreamer failed. Fallback to default.")
            cap.release()
            cap = None
 
    if cap is None:
        print(f"Opening camera {cam_id} (OpenCV)...")
        cap = open_default_camera(cam_id)
    
    return cap
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp32')
    parser.add_argument('--prune', type=float, default=0.0)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-mirror', action='store_true')
    parser.add_argument('--no-gstreamer', action='store_true')
    parser.add_argument("--method", type=str, default="mediapipe", choices=["mediapipe", "mobilehand"], help="Tracking method")
    args = parser.parse_args()
 
    if not CAPTURE_DIR.exists():
        CAPTURE_DIR.mkdir(parents=True)
 
    # Init Pipeline
    print(f"[Init] Hand Tracking Pipeline via {args.method}...")
    
    if args.method == "mobilehand":
        from hand_tracking.pipeline_mobilehand import HandTrackingPipeline, draw_landmarks
    else:
        from hand_tracking.pipeline_mediapipe import HandTrackingPipeline, draw_landmarks

    pipeline = HandTrackingPipeline(precision=args.precision, prune_rate=args.prune)
    detector = GestureDetector()
 
    cap = get_camera_pipeline(args.camera, args.no_gstreamer)
    if not cap.isOpened():
        print("Failed to open camera.")
        return
 
    cv2.namedWindow('Gesture Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gesture Camera', WINDOW_WIDTH, WINDOW_HEIGHT)
 
    # State
    MODE = "CAMERA" # CAMERA / GALLERY
    gallery_files = []
    gallery_idx = 0
    flash_timer = 0
    
    # Message overlay
    msg_timer = 0
    current_msg = ""
 
    print("\n[Ready] Press 'q' to quit\n")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        if not args.no_mirror:
            frame = frame[:, ::-1].copy()
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # 1. Inference
        landmarks_list, detections, _, _ = pipeline.process_frame(frame)
        
        # 2. Gesture Detection
        gesture = detector.update(landmarks_list)
        
        # === MODE LOGIC ===
        
        if MODE == "CAMERA":
            # Draw UI
            cv2.putText(frame, "CAMERA MODE (Fist: Capture, Swipe: Gallery)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Action
            if gesture == "FIST":
                # Capture
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = CAPTURE_DIR / f"cap_{timestamp}.jpg"
                cv2.imwrite(str(save_path), frame)
                print(f"[Captured] {save_path.name}")
                
                flash_timer = 5 # Flash effect frames
                current_msg = "Captured!"
                msg_timer = 30
                
            elif gesture == "SWIPE_RIGHT" or gesture == "SWIPE_LEFT":
                # Enter Gallery
                MODE = "GALLERY"
                gallery_files = sorted(list(CAPTURE_DIR.glob("*.jpg")), key=os.path.getmtime)
                if gallery_files:
                    gallery_idx = len(gallery_files) - 1 # Show latest
                    current_msg = "Entering Gallery"
                    msg_timer = 30
                else:
                    MODE = "CAMERA" # Back to camera if no files
                    current_msg = "No photos yet!"
                    msg_timer = 30
            
            # Flash effect
            if flash_timer > 0:
                overlay = np.ones_like(frame) * 255
                alpha = 0.5
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                flash_timer -= 1
                
        elif MODE == "GALLERY":
            # Refresh files just in case
            if not gallery_files:
                gallery_files = sorted(list(CAPTURE_DIR.glob("*.jpg")), key=os.path.getmtime)
                
            if not gallery_files:
                MODE = "CAMERA"
                continue
                
            # Show Image
            try:
                img_path = str(gallery_files[gallery_idx])
                gallery_img = cv2.imread(img_path)
                gallery_img = cv2.resize(gallery_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
                
                # Overlay on frame (or replace frame)
                # Let's replace frame but keep gesture visibility (optional)
                # Actually user wants to browse, so show the image.
                frame = gallery_img
            except:
                print(f"Error loading {gallery_files[gallery_idx]}")
                del gallery_files[gallery_idx]
                gallery_idx = 0
            
            # UI
            cv2.putText(frame, f"GALLERY {gallery_idx+1}/{len(gallery_files)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Swipe: Nav, Fist: Camera", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Action
            if gesture == "SWIPE_LEFT": # Previous
                gallery_idx = (gallery_idx - 1) % len(gallery_files)
            elif gesture == "SWIPE_RIGHT": # Next
                gallery_idx = (gallery_idx + 1) % len(gallery_files)
            elif gesture == "FIST":
                MODE = "CAMERA"
                current_msg = "Back to Camera"
                msg_timer = 30
        
        # === COMMON UI ===
        
        # Draw Landmarks (for visual feedback even in Gallery mode if we overlay?)
        # For Gallery mode, maybe logic of gesture detection still runs on CAMERA frame, but we show GALLERY image.
        # But wait! 'frame' variable is overwritten in Gallery Mode above.
        # Detection ran on 'frame' BEFORE overwrite. Correct.
        # But visualization `draw_landmarks` needs to draw on the FINAL frame?
        # If in Gallery Mode, drawing landmarks on top of the photo might be confusing but helpful to enable gestures.
        # Let's draw landmarks on whatever is shown.
        if landmarks_list:
            for lm in landmarks_list:
                draw_landmarks(frame, lm)
        
        # Message Overlay
        if msg_timer > 0:
            cv2.putText(frame, current_msg, (WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            msg_timer -= 1
 
        cv2.imshow('Gesture Camera', frame)
 
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()
