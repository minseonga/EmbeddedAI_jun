#!/usr/bin/env python
"""
환경 디버깅 스크립트
- 왜 터미널에서 직접 실행하면 MediaPipe 출력이 나오는지 확인
"""
import os
import sys

# 환경 출력
print("=" * 50)
print("환경 정보")
print("=" * 50)
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"CWD: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print()

# MediaPipe 환경변수 설정 확인
print("MediaPipe 관련 환경변수:")
for key in ['TF_CPP_MIN_LOG_LEVEL', 'GLOG_minloglevel', 'MEDIAPIPE_DISABLE_GPU']:
    print(f"  {key}: {os.environ.get(key, 'Not set')}")
print()

# MediaPipe import 전에 환경변수 설정
print("환경변수 설정 중...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

# stderr 리다이렉트 (MediaPipe 출력 숨기기)
import io
import contextlib

print("MediaPipe 테스트...")
print("(이 다음에 indexes_mapping 출력이 나오면 환경변수가 안 먹히는 것)")
print()

# MediaPipe import
try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    print("✓ MediaPipe import 성공")
except Exception as e:
    print(f"✗ MediaPipe import 실패: {e}")
    sys.exit(1)

# FaceMesh 생성
print("FaceMesh 초기화...")
try:
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,  # attention 비활성화 (출력 줄임)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ FaceMesh 초기화 성공")
except Exception as e:
    print(f"✗ FaceMesh 초기화 실패: {e}")
    sys.exit(1)

print()
print("=" * 50)
print("테스트 완료!")
print("=" * 50)
