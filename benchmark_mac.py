import os
import sys
from ultralytics import YOLO
import time

# =========================================================
# 설정: 경로 및 파일명
# =========================================================
MODEL_DIR = "assets/models"
SOURCE_PT = "yolo11n_hand_pose.pt"
DATA_YAML = "coco8.yaml" # 가지고 있는 yaml 파일이 없으면 자동 다운로드됨

# 경로 합치기
pt_path = os.path.join(MODEL_DIR, SOURCE_PT)

print(f"🚀 Mac 검증 시작: {pt_path}")

# =========================================================
# 1. 모델 변환 (Export) - 에러 방지 처리 포함
# =========================================================
if not os.path.exists(pt_path):
    print(f"❌ 원본 모델이 없습니다: {pt_path}")
    sys.exit(1)

model = YOLO(pt_path)

print("\n[1/3] CoreML (INT8) 변환 중... (Mac 속도 최적화)")
try:
    # nms=True를 켜면 CoreML 내부에서 NMS처리를 해서 더 빠름
    model.export(format='coreml', int8=True, nms=True) 
except Exception as e:
    print(f"⚠️ CoreML 변환 실패: {e}")

print("\n[2/3] TFLite (INT8) 변환 중... (용량 최적화)")
try:
    model.export(format='tflite', int8=True)
except Exception as e:
    print(f"⚠️ TFLite 변환 실패: {e}")


# =========================================================
# 2. 성능 벤치마크 (Benchmark)
# =========================================================
print("\n[3/3] 성능 측정 시작...")

# 측정할 모델 목록 자동 탐색
targets = [
    ("PyTorch (Original)", pt_path),
    ("CoreML (INT8)", pt_path.replace(".pt", ".mlpackage")),
    ("TFLite (INT8)", pt_path.replace(".pt", "_int8.tflite")) # tflite 이름 규칙 확인 필요
]

results = []

for name, path in targets:
    # 파일 존재 확인
    if not os.path.exists(path):
        # TFLite의 경우 이름이 다를 수 있어 한 번 더 확인
        if "tflite" in path:
             # 보통 _saved_model 폴더나 다른 이름일 수 있음, 여기선 단순화
             pass
        results.append((name, "Not Found", "N/A"))
        continue

    try:
        # 모델 로드 및 검증
        print(f" -> 측정 중: {name}...")
        test_model = YOLO(path, task='pose') # pose 모델이므로 task 명시
        
        # 벤치마크 (Validation) - 10장만 빠르게 테스트하고 싶지만 옵션 제한적
        # imgsz=640
        metrics = test_model.val(data=DATA_YAML, imgsz=640, verbose=False)
        
        # 속도 계산 (Inference time)
        inf_time = metrics.speed['inference']
        fps = 1000 / inf_time if inf_time > 0 else 0
        
        # 용량 계산
        if os.path.isdir(path): # CoreML은 폴더
            size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(path) for f in fn) / (1024**2)
        else:
            size_mb = os.path.getsize(path) / (1024**2)
            
        results.append((name, f"{fps:.1f} FPS", f"{size_mb:.2f} MB"))
        
    except Exception as e:
        print(f"   ❌ {name} 에러: {e}")
        results.append((name, "Error", "Error"))

# =========================================================
# 3. 최종 결과표 출력
# =========================================================
print("\n" + "="*50)
print(f"{'Model':<20} | {'Speed':<12} | {'Size':<10}")
print("-" * 50)
for name, speed, size in results:
    print(f"{name:<20} | {speed:<12} | {size:<10}")
print("="*50)
print("📌 CoreML은 Mac에서 빠르고, TFLite는 용량이 작아야 성공입니다.")