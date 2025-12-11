import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from ultralytics import YOLO

# torch_pruning 확인
try:
    import torch_pruning as tp
    HAS_TP = True
except ImportError:
    HAS_TP = False
    print("⚠️ 'torch_pruning' 없음. (설치: pip install torch-pruning)")

# =========================================================
# ⚙️ 설정
# =========================================================
BASE_DIR = "assets/models"
ORIGINAL_MODEL = "yolo11n_hand_pose.pt"
IMG_SIZE = 640

# 경로 설정
original_path = os.path.join(BASE_DIR, ORIGINAL_MODEL)

print("="*70)
print("🚀 [Absolute Final] 데이터셋 에러 없는 속도 측정 스크립트")
print("="*70)

results = []

# ---------------------------------------------------------
# 🛠️ 헬퍼 함수: 속도 및 용량 측정 (Predict 모드)
# ---------------------------------------------------------
def benchmark_speed(model_obj, name, file_path=None, is_yolo=True):
    print(f"   👉 측정 중: {name}...")
    
    # 1. 용량 측정
    size_mb = 0
    if file_path and os.path.exists(file_path):
        if os.path.isdir(file_path): # CoreML 등
            size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(file_path) for f in fn) / (1024**2)
        else:
            size_mb = os.path.getsize(file_path) / (1024**2)
    
    # 2. 속도 측정 (Warmup + Test)
    try:
        # 더미 이미지 생성 (검은 화면)
        dummy_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(5):
            if is_yolo:
                # verbose=False로 로그 끄기
                model_obj.predict(dummy_img, imgsz=IMG_SIZE, verbose=False, device='cpu')
            else:
                # Raw PyTorch Model
                dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
                model_obj(dummy_input)

        # 진짜 측정 (10회 평균)
        t_start = time.time()
        for _ in range(10):
            if is_yolo:
                model_obj.predict(dummy_img, imgsz=IMG_SIZE, verbose=False, device='cpu')
            else:
                dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
                with torch.no_grad():
                    model_obj(dummy_input)
        t_end = time.time()
        
        avg_time = (t_end - t_start) / 10
        fps = 1.0 / avg_time
        
        print(f"      ✅ 결과: {fps:.1f} FPS | {size_mb:.2f} MB")
        results.append((name, f"{fps:.1f} FPS", f"{size_mb:.2f} MB"))
        
    except Exception as e:
        print(f"      ❌ 측정 실패: {str(e)[:50]}")
        results.append((name, "Error", f"{size_mb:.2f} MB"))

# =========================================================
# 1️⃣ Original Model
# =========================================================
if os.path.exists(original_path):
    model = YOLO(original_path)
    benchmark_speed(model, "Original (FP32)", original_path)

# =========================================================
# 2️⃣ Structured Pruning (In-Memory 측정)
# =========================================================
if HAS_TP and os.path.exists(original_path):
    print("\n[Step 2] Structured Pruning (30%) 생성 및 측정...")
    try:
        # 모델 로드 (Raw PyTorch)
        yolo_tmp = YOLO(original_path)
        model_raw = yolo_tmp.model
        
        # Pruning
        example_inputs = torch.randn(1, 3, 640, 640)
        imp = tp.importance.MagnitudeImportance(p=1)
        ignored_layers = []
        for m in model_raw.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == model_raw.head.nc:
                ignored_layers.append(m)
        
        pruner = tp.pruner.MagnitudePruner(
            model_raw, example_inputs, importance=imp, iterative_steps=1, pruning_ratio=0.3, ignored_layers=ignored_layers
        )
        pruner.step()
        
        # 저장 (Jetson용)
        save_path = original_path.replace(".pt", "_structured.pt")
        torch.save(model_raw, save_path)
        
        # 측정 (Raw Model로 측정)
        benchmark_speed(model_raw, "Structured Pruned (30%)", save_path, is_yolo=False)
        
    except Exception as e:
        print(f"   ❌ Pruning 실패: {e}")

# =========================================================
# 3️⃣ Unstructured Pruning (Fix & Measure)
# =========================================================
print("\n[Step 3] Unstructured Pruning 모델 정리...")
pruned_files = [
    os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) 
    if f.endswith('.pt') and 'pruned' in f and 'fixed' not in f and 'structured' not in f
]

for f_path in pruned_files:
    try:
        model_wrap = YOLO(f_path)
        # 마스크 제거
        for name, m in model_wrap.model.named_modules():
            if hasattr(m, "weight_mask"):
                prune.remove(m, "weight")
        
        # _fixed.pt로 저장
        fixed_path = f_path.replace(".pt", "_fixed.pt")
        model_wrap.save(fixed_path)
        
        # 측정
        name = os.path.basename(f_path).replace("yolo11n_hand_pose_", "").replace(".pt", "")
        benchmark_speed(YOLO(fixed_path), f"Unstructured ({name})", fixed_path)
        
    except Exception as e:
        print(f"   Skip {os.path.basename(f_path)}: {e}")

# =========================================================
# 4️⃣ Quantization (CoreML, TFLite)
# =========================================================
print("\n[Step 4] Quantization 모델 측정...")

# CoreML
coreml_path = original_path.replace(".pt", ".mlpackage")
if not os.path.exists(coreml_path):
    try:
        YOLO(original_path).export(format='coreml', int8=True, nms=True)
    except: pass

if os.path.exists(coreml_path):
    # CoreML 로드는 YOLO('file.mlpackage')로 가능
    benchmark_speed(YOLO(coreml_path, task='pose'), "CoreML (INT8)", coreml_path)

# TFLite
tflite_path = None
potential_path = os.path.join(BASE_DIR, "yolo11n_hand_pose_saved_model", "yolo11n_hand_pose_int8.tflite")
if os.path.exists(potential_path):
    tflite_path = potential_path
else:
    # 없으면 만들어서 찾기
    try:
        YOLO(original_path).export(format='tflite', int8=True)
        if os.path.exists(potential_path): tflite_path = potential_path
    except: pass

if tflite_path:
    benchmark_speed(YOLO(tflite_path, task='pose'), "TFLite (INT8)", tflite_path)


# =========================================================
# 📊 최종 결과
# =========================================================
print("\n" + "="*75)
print(f"{'Model':<35} | {'Speed (Mac)':<15} | {'Size':<10}")
print("-" * 75)
for name, speed, size in results:
    print(f"{name:<35} | {speed:<15} | {size:<10}")
print("="*75)