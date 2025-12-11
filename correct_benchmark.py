"""
📊 모델 최적화 벤치마크 스크립트 (수정본)

이 스크립트는 다음 3가지를 정확하게 측정합니다:
1. 모델 사이즈 (파라미터 수)
2. FLOPs (연산량)
3. 실행 속도 (FPS/Latency)

⚠️ Jetson Nano에서 실행하세요!
"""

import os
import sys
import time
import csv
import copy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# =========================================================
# 필수 라이브러리 확인
# =========================================================
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ 'ultralytics' 설치 필요: pip install ultralytics")
    sys.exit(1)

# torch_pruning 확인 (진짜 structured pruning용)
try:
    import torch_pruning as tp
    HAS_TP = True
except ImportError:
    HAS_TP = False
    print("⚠️ 'torch_pruning' 없음. Structured pruning 건너뜀")
    print("   설치: pip install torch-pruning")

# FLOPs 계산도구
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("⚠️ 'thop' 없음. FLOPs 계산 건너뜀")
    print("   설치: pip install thop")

# =========================================================
# ⚙️ 설정
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
ORIGINAL_MODEL_PATH = MODELS_DIR / "yolo11n_hand_pose.pt"

IMG_SIZE = 640
RESULTS_FILE = ROOT_DIR / "benchmark_results.csv"

# Jetson Nano 환경 감지
IS_JETSON = os.path.exists("/etc/nv_tegra_release")
if IS_JETSON:
    print("✅ Jetson Nano 환경 감지됨")
    DEVICE = 0  # GPU
else:
    print("ℹ️ 일반 PC 환경 (CPU 모드)")
    DEVICE = 'cpu'

# =========================================================
# 🛠️ 유틸리티 함수
# =========================================================

def count_parameters(model: nn.Module) -> int:
    """모델의 총 파라미터 수(학습가능+고정) 반환"""
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model: nn.Module) -> int:
    """0이 아닌 파라미터 수만 계산"""
    return sum((p != 0).sum().item() for p in model.parameters())


def get_flops(model: nn.Module, input_size=(1, 3, 640, 640)) -> float:
    """FLOPs(GFLOPs) 반환"""
    if not HAS_THOP:
        return 0.0
    
    model = model.to('cpu').eval()
    dummy_input = torch.randn(input_size).to('cpu')
    
    try:
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        # thop이 생성한 임시 속성 제거
        for module in model.modules():
            for attr in ['total_ops', 'total_params']:
                if hasattr(module, attr):
                    delattr(module, attr)
        return macs / 1e9  # GFLOPs
    except Exception as e:
        print(f"   ⚠️ FLOPs 측정 실패: {e}")
        return 0.0


def get_model_size_mb(file_path) -> float:
    """파일 크기(MB) 반환"""
    if not os.path.exists(file_path):
        return 0.0
    if os.path.isdir(file_path):
        return sum(
            os.path.getsize(os.path.join(dp, f)) 
            for dp, _, fn in os.walk(file_path) for f in fn
        ) / (1024**2)
    return os.path.getsize(file_path) / (1024**2)


def measure_speed(model, num_warmup=10, num_test=50) -> tuple:
    """
    추론 속도 측정
    Returns: (avg_fps, avg_latency_ms)
    """
    dummy_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(num_warmup):
        model.predict(dummy_img, imgsz=IMG_SIZE, verbose=False, device=DEVICE)
    
    # Measure
    times = []
    for _ in range(num_test):
        t_start = time.time()
        model.predict(dummy_img, imgsz=IMG_SIZE, verbose=False, device=DEVICE)
        t_end = time.time()
        times.append(t_end - t_start)
    
    avg_time = sum(times) / len(times)
    avg_fps = 1.0 / avg_time
    avg_latency_ms = avg_time * 1000
    
    return avg_fps, avg_latency_ms


def save_results(results: list):
    """결과를 CSV로 저장"""
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Model', 'Type', 'Params(M)', 'NonZero_Params(M)', 'FLOPs(G)', 
                      'FPS', 'Latency(ms)', 'Size(MB)', 'Prune_Rate', 'Precision']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\n💾 결과 저장됨: {RESULTS_FILE}")


# =========================================================
# 🎯 Structured Pruning 함수 (진짜 채널 제거)
# =========================================================

def apply_structured_pruning(model: nn.Module, prune_rate: float) -> nn.Module:
    """
    torch_pruning을 사용한 진짜 Structured Pruning
    실제로 채널 수를 줄여서 파라미터와 FLOPs 감소
    """
    if not HAS_TP:
        return model
    
    model = model.to('cpu').eval()
    example_inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to('cpu')
    
    # YOLO 모델의 마지막 레이어 감지 (pruning에서 제외)
    ignored_layers = []
    
    # Detect2D, Pose, Segment 등의 head 레이어 자동 감지
    for name, module in model.named_modules():
        # YOLO head의 마지막 Conv 레이어들은 출력 채널이 고정되어야 함
        if 'cv2' in name or 'cv3' in name:  # Detection head
            ignored_layers.append(module)
        if 'cv4' in name:  # Pose head
            ignored_layers.append(module)
        # DFL(Distribution Focal Loss) 레이어도 제외
        if hasattr(module, 'reg_max'):
            ignored_layers.append(module)
    
    # Importance 계산기 (L1 magnitude 기반)
    imp = tp.importance.MagnitudeImportance(p=1)
    
    # Pruner 생성 - DepGraph 기반으로 의존성 자동 처리
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=prune_rate,
        ignored_layers=ignored_layers,
        unwrapped_parameters=None,
        round_to=8,  # 채널 수를 8의 배수로 맞춤 (GPU 효율)
    )
    
    # Pruning 실행
    pruner.step()
    
    return model


# =========================================================
# 🎯 메인 벤치마크
# =========================================================

def run_benchmark():
    print("=" * 70)
    print("🚀 모델 최적화 벤치마크 시작")
    print("=" * 70)
    
    if not ORIGINAL_MODEL_PATH.exists():
        print(f"❌ 모델 파일 없음: {ORIGINAL_MODEL_PATH}")
        return
    
    results = []
    
    # =========================================================
    # 1️⃣ 원본 모델 (Baseline)
    # =========================================================
    print("\n[1] 원본 모델 (Baseline) 측정...")
    
    yolo_base = YOLO(ORIGINAL_MODEL_PATH)
    model_base = yolo_base.model
    
    base_params = count_parameters(model_base) / 1e6
    base_nonzero = count_nonzero_parameters(model_base) / 1e6
    base_flops = get_flops(model_base)
    base_fps, base_latency = measure_speed(yolo_base)
    base_size = get_model_size_mb(ORIGINAL_MODEL_PATH)
    
    print(f"   📊 파라미터: {base_params:.3f}M")
    print(f"   📊 FLOPs: {base_flops:.3f}G")
    print(f"   📊 속도: {base_fps:.1f} FPS ({base_latency:.1f}ms)")
    print(f"   📊 크기: {base_size:.2f}MB")
    
    results.append({
        'Model': 'Baseline',
        'Type': 'Original',
        'Params(M)': round(base_params, 3),
        'NonZero_Params(M)': round(base_nonzero, 3),
        'FLOPs(G)': round(base_flops, 3),
        'FPS': round(base_fps, 1),
        'Latency(ms)': round(base_latency, 1),
        'Size(MB)': round(base_size, 2),
        'Prune_Rate': 0.0,
        'Precision': 'FP32'
    })
    
    # =========================================================
    # 2️⃣ Structured Pruning (30%, 50%, 70%)
    # =========================================================
    if HAS_TP:
        print("\n[2] Structured Pruning 모델 생성 및 측정...")
        
        prune_rates = [0.3, 0.5, 0.7]
        
        for rate in prune_rates:
            pct = int(rate * 100)
            print(f"\n   🔹 Pruning Rate: {pct}%")
            
            try:
                # 새로운 모델 로드 (매번 fresh하게)
                yolo_tmp = YOLO(ORIGINAL_MODEL_PATH)
                model_pruned = copy.deepcopy(yolo_tmp.model)
                
                # Structured Pruning 적용
                model_pruned = apply_structured_pruning(model_pruned, rate)
                
                # 측정
                pruned_params = count_parameters(model_pruned) / 1e6
                pruned_nonzero = count_nonzero_parameters(model_pruned) / 1e6
                pruned_flops = get_flops(model_pruned)
                
                # 저장 후 YOLO로 다시 로드하여 속도 측정
                save_path = MODELS_DIR / f"yolo11n_hand_pose_real_pruned_{pct}.pt"
                
                # state_dict만 저장 (전체 모델 저장보다 안전)
                torch.save({
                    'model': model_pruned.state_dict(),
                    'yaml': yolo_tmp.model.yaml,  # 구조 정보
                    'stride': yolo_tmp.model.stride,
                    'names': yolo_tmp.model.names,
                }, save_path)
                
                pruned_size = get_model_size_mb(save_path)
                
                # 속도 측정 (raw PyTorch 모델로)
                model_pruned.to(DEVICE if DEVICE != 'cpu' else 'cpu').eval()
                
                dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
                if DEVICE != 'cpu':
                    dummy_input = dummy_input.cuda()
                
                # Warmup
                for _ in range(10):
                    with torch.no_grad():
                        model_pruned(dummy_input)
                
                # Measure
                times = []
                for _ in range(50):
                    t_start = time.time()
                    with torch.no_grad():
                        model_pruned(dummy_input)
                    times.append(time.time() - t_start)
                
                pruned_fps = 1.0 / (sum(times) / len(times))
                pruned_latency = (sum(times) / len(times)) * 1000
                
                # 실제 감소율 계산
                param_reduction = (1 - pruned_params / base_params) * 100
                flops_reduction = (1 - pruned_flops / base_flops) * 100 if base_flops > 0 else 0
                
                print(f"      ✅ 파라미터: {pruned_params:.3f}M ({param_reduction:.1f}% 감소)")
                print(f"      ✅ FLOPs: {pruned_flops:.3f}G ({flops_reduction:.1f}% 감소)")
                print(f"      ✅ 속도: {pruned_fps:.1f} FPS ({pruned_latency:.1f}ms)")
                print(f"      ✅ 크기: {pruned_size:.2f}MB")
                
                results.append({
                    'Model': f'Pruned_{pct}%',
                    'Type': 'Structured_Pruning',
                    'Params(M)': round(pruned_params, 3),
                    'NonZero_Params(M)': round(pruned_nonzero, 3),
                    'FLOPs(G)': round(pruned_flops, 3),
                    'FPS': round(pruned_fps, 1),
                    'Latency(ms)': round(pruned_latency, 1),
                    'Size(MB)': round(pruned_size, 2),
                    'Prune_Rate': rate,
                    'Precision': 'FP32'
                })
                
            except Exception as e:
                print(f"      ❌ 실패: {e}")
                import traceback
                traceback.print_exc()
    
    # =========================================================
    # 3️⃣ Quantization (FP16/INT8) - Jetson에서만
    # =========================================================
    if IS_JETSON:
        print("\n[3] Quantization 모델 생성 및 측정 (Jetson Only)...")
        
        # FP16 TensorRT
        print("\n   🔹 FP16 TensorRT 변환...")
        try:
            fp16_engine = MODELS_DIR / "yolo11n_hand_pose_fp16.engine"
            
            if not fp16_engine.exists():
                yolo_base.export(format='engine', half=True, imgsz=IMG_SIZE, device=0)
                # 생성된 .engine 파일 이름 변경
                default_engine = ORIGINAL_MODEL_PATH.with_suffix('.engine')
                if default_engine.exists():
                    default_engine.rename(fp16_engine)
            
            if fp16_engine.exists():
                yolo_fp16 = YOLO(fp16_engine, task='pose')
                fp16_fps, fp16_latency = measure_speed(yolo_fp16)
                fp16_size = get_model_size_mb(fp16_engine)
                
                print(f"      ✅ FP16 속도: {fp16_fps:.1f} FPS ({fp16_latency:.1f}ms)")
                
                results.append({
                    'Model': 'TensorRT_FP16',
                    'Type': 'Quantization',
                    'Params(M)': round(base_params, 3),
                    'NonZero_Params(M)': round(base_nonzero, 3),
                    'FLOPs(G)': round(base_flops, 3),
                    'FPS': round(fp16_fps, 1),
                    'Latency(ms)': round(fp16_latency, 1),
                    'Size(MB)': round(fp16_size, 2),
                    'Prune_Rate': 0.0,
                    'Precision': 'FP16'
                })
        except Exception as e:
            print(f"      ❌ FP16 실패: {e}")
        
        # INT8 TensorRT
        print("\n   🔹 INT8 TensorRT 변환...")
        try:
            int8_engine = MODELS_DIR / "yolo11n_hand_pose_int8.engine"
            
            if not int8_engine.exists():
                yolo_base.export(format='engine', int8=True, imgsz=IMG_SIZE, device=0)
                default_engine = ORIGINAL_MODEL_PATH.with_suffix('.engine')
                if default_engine.exists():
                    default_engine.rename(int8_engine)
            
            if int8_engine.exists():
                yolo_int8 = YOLO(int8_engine, task='pose')
                int8_fps, int8_latency = measure_speed(yolo_int8)
                int8_size = get_model_size_mb(int8_engine)
                
                print(f"      ✅ INT8 속도: {int8_fps:.1f} FPS ({int8_latency:.1f}ms)")
                
                results.append({
                    'Model': 'TensorRT_INT8',
                    'Type': 'Quantization',
                    'Params(M)': round(base_params, 3),
                    'NonZero_Params(M)': round(base_nonzero, 3),
                    'FLOPs(G)': round(base_flops, 3),
                    'FPS': round(int8_fps, 1),
                    'Latency(ms)': round(int8_latency, 1),
                    'Size(MB)': round(int8_size, 2),
                    'Prune_Rate': 0.0,
                    'Precision': 'INT8'
                })
        except Exception as e:
            print(f"      ❌ INT8 실패: {e}")
    else:
        print("\n[3] Quantization 스킵 (Jetson Nano에서만 TensorRT 지원)")
    
    # =========================================================
    # 📊 결과 출력 및 저장
    # =========================================================
    save_results(results)
    
    print("\n" + "=" * 90)
    print(f"{'Model':<20} | {'Params(M)':<12} | {'FLOPs(G)':<10} | {'FPS':<8} | {'Latency(ms)':<12} | {'Size(MB)':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['Model']:<20} | {r['Params(M)']:<12} | {r['FLOPs(G)']:<10} | {r['FPS']:<8} | {r['Latency(ms)']:<12} | {r['Size(MB)']:<10}")
    print("=" * 90)
    
    print("\n🎉 벤치마크 완료!")


if __name__ == "__main__":
    run_benchmark()
