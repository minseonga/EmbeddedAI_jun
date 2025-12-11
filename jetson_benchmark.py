import os
import time
import numpy as np
from ultralytics import YOLO

# =========================================================
# ⚙️ 설정 (Jetson Nano 환경)
# =========================================================
MODEL_DIR = "assets/models"  # 모델 파일들을 이 폴더에 넣으세요
IMG_SIZE = 640

print("="*60)
print("🚀 Jetson Nano - TensorRT 변환 및 벤치마크 시작")
print("="*60)

# 1. 대상 파일 찾기 (.pt 및 .onnx)
# _fixed.pt (Pruning), .pt (Original), .onnx
candidates = [
    f for f in os.listdir(MODEL_DIR) 
    if (f.endswith('.pt') or f.endswith('.onnx')) 
    and 'coreml' not in f and 'structured' not in f
]
candidates.sort()

results = []

for f_name in candidates:
    file_path = os.path.join(MODEL_DIR, f_name)
    model_name = f_name.replace(".pt", "").replace(".onnx", "")
    
    print(f"\n👉 처리 중: {model_name}")
    
    try:
        # -----------------------------------------------------
        # [A] TensorRT 엔진 변환 (Export)
        # -----------------------------------------------------
        # 이미 변환된 엔진이 있으면 스킵
        engine_path = file_path.replace(".pt", ".engine").replace(".onnx", ".engine")
        
        if os.path.exists(engine_path):
            print("   ✅ TensorRT 엔진이 이미 존재합니다. (Skip Export)")
        else:
            print("   ⚙️ TensorRT(FP16) 변환 시작... (시간이 좀 걸립니다)")
            # .pt 파일이면 YOLO로 로드해서 변환
            if f_name.endswith('.pt'):
                model = YOLO(file_path)
                # FP16 (Half) 적용 -> Jetson 속도 핵심!
                model.export(format='engine', half=True, imgsz=IMG_SIZE, device=0) 
            
            # .onnx 파일이면 바로 엔진 변환 (yolo 커맨드라인 처럼 동작시키기 위해 로드)
            elif f_name.endswith('.onnx'):
                # ONNX는 YOLO 클래스로 바로 로드가 안될 수 있어, subprocess 권장하나 
                # 여기선 Ultralytics 기능을 믿고 시도
                model = YOLO(file_path, task='pose')
                # ONNX는 이미 구조가 고정이라 export 옵션이 제한적일 수 있음
                # 보통 pt -> engine이 정석임. ONNX는 패스하거나 수동 변환 필요할 수 있음.
                pass 

        # -----------------------------------------------------
        # [B] 속도 측정 (Benchmark)
        # -----------------------------------------------------
        # 변환된 엔진이 있으면 엔진을 로드, 없으면 원본 로드
        load_path = engine_path if os.path.exists(engine_path) else file_path
        current_type = "TensorRT (FP16)" if load_path.endswith(".engine") else "PyTorch/ONNX"
        
        print(f"   🔥 벤치마킹 시작 ({current_type})...")
        
        # 모델 로드 (task='pose' 필수)
        test_model = YOLO(load_path, task='pose')
        
        # 더미 데이터로 Warmup
        dummy_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        for _ in range(5):
            test_model.predict(dummy_img, imgsz=IMG_SIZE, verbose=False, device=0)
            
        # 속도 측정 (20회 평균)
        t_start = time.time()
        for _ in range(20):
            test_model.predict(dummy_img, imgsz=IMG_SIZE, verbose=False, device=0)
        t_end = time.time()
        
        avg_time = (t_end - t_start) / 20
        fps = 1.0 / avg_time
        
        # 용량 확인 (엔진 파일 우선)
        size_mb = os.path.getsize(load_path) / (1024**2)
        
        print(f"      ✅ 결과: {fps:.1f} FPS | {size_mb:.2f} MB")
        results.append((model_name, current_type, f"{fps:.1f} FPS", f"{size_mb:.2f} MB"))

    except Exception as e:
        print(f"      ❌ 실패: {e}")
        results.append((model_name, "Error", "Error", "Error"))

# =========================================================
# 📊 최종 결과표 (Jetson Nano)
# =========================================================
print("\n" + "="*80)
print(f"{'Model':<35} | {'Format':<15} | {'Speed':<10} | {'Size':<10}")
print("-" * 80)
for name, fmt, speed, size in results:
    print(f"{name:<35} | {fmt:<15} | {speed:<10} | {size:<10}")
print("="*80)