#!/usr/bin/env python
"""
MobileHand HMR 테스트 스크립트
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "mobilehand_repo/code"))

import cv2
import numpy as np
import time

print("=" * 50)
print("MobileHand HMR 테스트")
print("=" * 50)

print("\n1. Importing HandTrackingPipeline...")
from hand_tracking import HandTrackingPipeline, draw_landmarks

print("\n2. Creating pipeline...")
try:
    pipeline = HandTrackingPipeline()
    print("✅ Pipeline created!")
except Exception as e:
    print(f"❌ Pipeline creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Opening camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera failed!")
    sys.exit(1)
print("✅ Camera opened!")

print("\n4. Running inference loop...")
print("Press 'q' to quit\n")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = frame[:, ::-1].copy()  # Mirror
    
    # Process
    try:
        landmarks, dets, mar, mc = pipeline.process_frame(frame)
        frame_count += 1
        
        # Draw
        for lm in landmarks:
            draw_landmarks(frame, lm)
        
        # FPS
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f} | Hands: {len(landmarks)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    cv2.imshow('MobileHand HMR Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal frames: {frame_count}, Avg FPS: {fps:.1f}")
print("Done!")
