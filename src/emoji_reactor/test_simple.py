#!/usr/bin/env python
"""
간단한 테스트 스크립트 - app.py 대신 사용
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import numpy as np

print("[Test] Importing pipeline...")
from hand_tracking import HandTrackingPipeline, draw_landmarks

print("[Test] Creating pipeline...")
pipeline = HandTrackingPipeline(precision='fp32', prune_rate=0.0)
pipeline.print_stats()

print("[Test] Opening camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[ERROR] Cannot open camera!")
    exit(1)

print("[Test] Starting loop (Press 'q' to quit)...")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = frame[:, ::-1].copy()  # Mirror
    frame = cv2.resize(frame, (640, 480))
    
    # Process
    landmarks, dets, mar, mouth_center = pipeline.process_frame(frame)
    
    # Draw
    for lm in landmarks:
        draw_landmarks(frame, lm)
    
    # Info
    state = "HANDS_UP" if len(landmarks) > 0 else "NORMAL"
    cv2.putText(frame, f"{state} | Hands: {len(landmarks)} | MAR: {mar:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[Test] Done!")
