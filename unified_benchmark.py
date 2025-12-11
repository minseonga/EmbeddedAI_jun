import os
import sys
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO
import numpy as np

# torch_pruning 확인 및 설치 안내
try:
    import torch_pruning as tp
    HAS_TP = True
except ImportError:
    HAS_TP = False
    print("⚠️ 'torch_pruning' 없음. 가지치기 측정은 건너뜁니다. (설치: pip install torch-pruning)")

# FLOPs 계산 도구
try:
    from thop import profile
except ImportError:
    print("⚠️ 'thop' 없음. FLOPs 계산을 건너뜁니다. (설치: pip install thop)")
    sys.exit(1) # 필수 라이브러리이므로 종료

# =========================================================
# ⚙️ 경로 및 설정
# =========================================================
# 현재 스크립트가 실행되는 위치를 기준으로 경로 설정
ROOT_DIR = Path(__file__).resolve().parent # /workspace/EmbeddedAI/ 가정
ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
ORIGINAL_MODEL_NAME = "yolo11n_hand_pose.pt"
ORIGINAL_MODEL_PATH = MODELS_DIR / ORIGINAL_MODEL_NAME

IMG_SIZE = 640
RESULTS_FILE = "final_experiment_results.csv"

# =========================================================
# 🛠️ 헬퍼 함수
# =========================================================

def cleanup_thop_attributes(model: nn.Module):
    """thop 측정 후 모델에 남아있는 임시 속성을 제거합니다."""
    for module in model.modules():
        for attr in ['total_ops', 'total_params', 'n_macs', 'n_params']:
            if hasattr(module, attr):
                delattr(module, attr)

def get_model_info(model: nn.Module, prune_rate: float):
    """Params와 FLOPs를 계산합니다."""
    try:
        model.to('cpu').eval()
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to('cpu')
        
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        
        flops_g = macs / 1e9       # Giga FLOPs
        params_m = params / 1e6    # Million Parameters
        
        # 측정 후 반드시 cleanup
        cleanup_thop_attributes(model)

        print(f"      [FLOPs] {params_m:.3f}M Params, {flops_g:.3f}G FLOPs")
        return params_m, flops_g
    except Exception as e:
        print(f"      [FLOPs] 측정 실패: {e}")
        return 0.0, 0.0

def save_result_to_csv(data: dict):
    """결과를 CSV 파일에 저장합니다."""
    file_exists = os.path.isfile(RESULTS_FILE)
    
    with open(RESULTS_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(data)
    
    print(f"      [저장] {data['Precision']}/{data['Prune Rate']} 결과 저장 완료.")


def benchmark_speed(model_wrap: YOLO, name: str, precision: str, prune_rate: float, file_path=None, base_flops=None, base_params=None):
    """모델의 용량, 속도(FPS/Latency)를 측정하고 저장합니다."""
    print(f"   👉 측정 중: {name} ({precision}, Prune:{prune_rate})")

    # 1. 용량 측정
    size_mb = 0
    if file_path and os.path.exists(file_path):
        if os.path.isdir(file_path): 
            size_mb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(file_path) for f in fn) / (1024**2)
        else:
            size_mb = os.path.getsize(file_path) / (1024**2)
    
    # 2. FLOPs / Params (PyTorch 모델인 경우만 재측정)
    params_m, flops_g = 0.0, 0.0
    
    # Pruned 모델은 FLOPs가 변했으므로 재측정 (단, .engine 파일은 안 됨)
    if file_path and str(file_path).endswith('.pt'):
        if hasattr(model_wrap, 'model') and isinstance(model_wrap.model, nn.Module):
            params_m, flops_g = get_model_info(model_wrap.model, prune_rate)
        
    # Quantization 모델은 이론값을 그대로 사용
    elif base_params and base_flops:
        params_m, flops_g = base_params, base_flops


    # 3. 속도 측정 (Warmup + Test)
    try:
        dummy_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        
        # Warmup (5회)
        for _ in range(5):
            model_wrap.predict(dummy_img, imgsz=IMG_SIZE, verbose=False, device='cpu')

        # Test (10회 평균)
        t_start = time.time()
        for _ in range(10):
            model_wrap.predict(dummy_img, imgsz=IMG_SIZE, verbose=False, device='cpu')
        t_end = time.time()
        
        avg_time = (t_end - t_start) / 10
        avg_latency = avg_time * 1000
        fps = 1.0 / avg_time
        
        print(f"      ✅ 결과: {fps:.1f} FPS | {size_mb:.2f} MB")

        # 4. CSV 저장
        save_result_to_csv({
            'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'Precision': precision,
            'Prune Rate': prune_rate,
            'Params(M)': round(params_m, 3),
            'FLOPs(G)': round(flops_g, 3),
            'FPS_App': round(fps, 2),
            'Latency(ms)_App': round(avg_latency, 2),
            'Size(MB)': round(size_mb, 2)
        })
        
    except Exception as e:
        print(f"      ❌ 측정 실패: {str(e)[:50]}")
        save_result_to_csv({
            'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'Precision': precision,
            'Prune Rate': prune_rate,
            'Params(M)': round(params_m, 3),
            'FLOPs(G)': round(flops_g, 3),
            'FPS_App': "Error",
            'Latency(ms)_App': "Error",
            'Size(MB)': round(size_mb, 2)
        })

# =========================================================
# 🎯 MAIN EXECUTION
# =========================================================
# =========================================================
# 🎯 MAIN EXECUTION (수정됨)
# =========================================================
def main():
    print("="*70)
    print("🚀 [통합 벤치마크] 파일 생성, FLOPs 측정, 속도 측정을 한번에 수행합니다.")
    print("="*70)
    print("⚠️ 이 스크립트는 로컬 PC/Mac에서 실행되어야 합니다.")

    if not ORIGINAL_MODEL_PATH.exists():
         raise FileNotFoundError(f"기반 모델 파일({ORIGINAL_MODEL_NAME})을 '{MODELS_DIR}'에서 찾을 수 없습니다.")

    # 0. Base Model 로드 및 FLOPs 기준값 설정
    yolo_base = YOLO(ORIGINAL_MODEL_PATH)
    base_params, base_flops = get_model_info(yolo_base.model, 0.0)
    
    print("\n[Step 0] 기본 모델 측정 (Baseline)")
    benchmark_speed(yolo_base, "Base Model", "fp32", 0.0, ORIGINAL_MODEL_PATH, base_params, base_flops)

    
    # --- 실험 목록 ---
    prune_rates = [0.3, 0.5, 0.7]
    quantization_formats = ["coreml", "tflite"] # INT8 측정

    # 1. Structured Pruning (가지치기 모델 생성 및 측정)
    if HAS_TP:
        print("\n[Step 1] Structured Pruning 모델 생성 및 측정...")
        for rate in prune_rates:
            # 🚨 FIX: Pruning 실패 시를 대비하여 변수 초기화 (Base 값으로 설정)
            pruned_params, pruned_flops = base_params, base_flops 
            
            prune_pct = int(rate * 100)
            save_path_tmp = MODELS_DIR / f"yolo11n_hand_pose_pruned_s_{prune_pct}_tmp.pt"
            save_path_final = MODELS_DIR / f"yolo11n_hand_pose_pruned_s_{prune_pct}.pt"
            
            # 1-1. 모델 생성 및 가지치기
            if not save_path_final.exists():
                print(f"   -> {rate} 모델 생성 중...")
                
                # --- 1. 원본 모델 로드 ---
                yolo_tmp = YOLO(ORIGINAL_MODEL_PATH)
                model_raw = yolo_tmp.model # PyTorch Module
                
                try:
                    # --- 2. Pruning 적용 (마스크 생성) ---
                    example_inputs = torch.randn(1, 3, 640, 640).to('cpu')
                    imp = tp.importance.MagnitudeImportance(p=1)
                    ignored_layers = []
                    for m in model_raw.modules():
                        # YOLO 헤드 레이어 제외 로직
                        if isinstance(m, torch.nn.Linear) and m.out_features == model_raw.head.nc:
                            ignored_layers.append(m)
                    
                    pruner = tp.pruner.MagnitudePruner(
                        model_raw, example_inputs, importance=imp, iterative_steps=1, pruning_ratio=rate, ignored_layers=ignored_layers
                    )
                    pruner.step()
                    
                    
                    # --- 3. FLOPs/Params 측정 (성공하면 변수 업데이트) ---
                    pruned_params, pruned_flops = get_model_info(model_raw, rate) 
                    print(f"   -> 이론 복잡도 측정 성공: {pruned_params:.3f}M Params, {pruned_flops:.3f}G FLOPs")
                    
                    
                    # --- 4. 마스크 제거 및 최종 저장 ---
                    for name, m in model_raw.named_modules():
                        if hasattr(m, "weight_mask"):
                            # 마스크 영구 제거
                            prune.remove(m, "weight")
                    
                    # 원본 YOLO 파일에 덮어씌워 저장
                    torch.save(model_raw, save_path_tmp)
                    
                    # YOLO 객체를 다시 로드하고 Ultralytics의 save()를 사용하여 메타데이터 포함
                    YOLO(save_path_tmp).save(save_path_final)
                    
                    # 임시 파일 삭제
                    os.remove(save_path_tmp)
                    print(f"   -> {save_path_final.name} 저장 완료.")


                except Exception as e:
                    print(f"   ❌ Pruning 실패 및 스킵: {e}")
                    # 실패한 경우 CSV에 기록 (0 값) -> Baseline 값은 유지되지만, 속도 측정은 건너뜀
                    save_result_to_csv({
                        'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'Precision': 'fp32', 'Prune Rate': rate,
                        'Params(M)': base_params, 'FLOPs(G)': base_flops, # 실패했으므로 베이스라인 값 기록
                        'FPS_App': "Error (Gen)", 'Latency(ms)_App': "Error (Gen)",
                        'Size(MB)': 0.0
                    })
                    continue # 속도 측정 건너뛰기
            
            # 1-2. 측정 (생성된 최종 파일을 다시 로드하여 속도 측정)
            pruned_model = YOLO(save_path_final)
            benchmark_speed(pruned_model, f"Pruned ({prune_pct}%)", "fp32", rate, save_path_final, pruned_params, pruned_flops)

    # 2. Quantization (양자화 모델 생성 및 측정)
    print("\n[Step 2] Quantization 모델 생성 및 측정...")
    for fmt in quantization_formats:
        export_path = MODELS_DIR / f"yolo11n_hand_pose_int8.{fmt}"

        # 2-1. 파일 생성
        if not export_path.exists() and fmt != "coreml":
             try:
                # CoreML 파일은 export 이름이 달라 다시 정의해야 함
                if fmt == 'coreml':
                    YOLO(ORIGINAL_MODEL_PATH).export(format=fmt, int8=True, nms=True)
                else:
                    YOLO(ORIGINAL_MODEL_PATH).export(format=fmt, int8=True, imgsz=IMG_SIZE)
                
                # CoreML/TFLite 경로 보정
                if fmt == 'coreml':
                    export_path = ORIGINAL_MODEL_PATH.parent / ORIGINAL_MODEL_NAME.replace('.pt', '.mlpackage')
                elif fmt == 'tflite':
                    tflite_folder = ORIGINAL_MODEL_PATH.parent / ORIGINAL_MODEL_NAME.replace('.pt', '_saved_model')
                    export_path = tflite_folder / f"{ORIGINAL_MODEL_NAME.replace('.pt', '_int8.tflite')}"
                
                if not export_path.exists(): 
                    print(f"   ❌ {fmt.upper()} INT8 생성되었으나 최종 파일 찾기 실패.")
                    continue
                
             except Exception as e:
                print(f"   ❌ {fmt.upper()} INT8 생성 실패: {e}")
                continue

        # 2-2. 측정
        if export_path.exists():
            name = f"INT8 ({fmt.upper()})"
            
            yolo_export = YOLO(export_path, task='pose')
            
            # FLOPs/Params는 Baseline 값을 사용
            benchmark_speed(yolo_export, name, "int8", 0.0, export_path, base_params, base_flops)
        else:
             print(f"   ❌ {fmt.upper()} INT8 측정 스킵 (파일 없음)")


if __name__ == "__main__":
    main()
    print("\n" + "="*70)
    print(f"✅ 통합 벤치마크 완료: 모든 결과가 {RESULTS_FILE}에 저장되었습니다.")
    print("이제 이 CSV 파일과 모델들을 Jetson Nano로 옮겨서 실제 FPS를 측정하세요.")
    print("="*70)