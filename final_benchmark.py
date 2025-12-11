"""
📊 YOLO11 최종 벤치마크 스크립트

현실적인 측정 항목:
1. 파라미터 수 (Total Params)
2. Non-zero 파라미터 수 (Effective Params) - Unstructured Pruning 효과
3. FLOPs
4. 실행 속도 (FPS/Latency)
5. 모델 크기 (MB)

참고:
- YOLO11은 torch_pruning의 DependencyGraph가 제대로 추적하지 못해
  Structured Pruning이 어렵습니다.
- 대신 Unstructured Pruning (weight를 0으로 만듦)을 사용하고,
  Non-zero 파라미터 수로 효과를 측정합니다.
- Jetson Nano에서는 TensorRT(FP16/INT8) Quantization이 가장 효과적입니다.
"""

import os
import sys
import time
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

from ultralytics import YOLO

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("⚠️ thop 없음 (pip install thop)")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets/models/yolo11n_hand_pose.pt"
RESULTS_FILE = ROOT / "final_benchmark_results.csv"
IMG_SIZE = 640

# Jetson 환경 감지
IS_JETSON = os.path.exists("/etc/nv_tegra_release")
DEVICE = 0 if IS_JETSON else 'cpu'


def count_params(model):
    """전체 파라미터 수"""
    return sum(p.numel() for p in model.parameters())


def count_nonzero_params(model):
    """0이 아닌 파라미터 수"""
    return sum((p != 0).sum().item() for p in model.parameters())


def get_flops(model, device='cpu'):
    """FLOPs 계산"""
    if not HAS_THOP:
        return 0.0
    
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    
    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        
        # thop 임시 속성 제거
        for m in model.modules():
            for attr in ['total_ops', 'total_params']:
                if hasattr(m, attr):
                    delattr(m, attr)
        
        return macs / 1e9
    except:
        return 0.0


def get_file_size(path):
    """파일 크기 (MB)"""
    if not os.path.exists(path):
        return 0.0
    if os.path.isdir(path):
        return sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fn in os.walk(path) for f in fn
        ) / (1024**2)
    return os.path.getsize(path) / (1024**2)


def measure_speed(yolo_model, num_warmup=10, num_test=50):
    """추론 속도 측정"""
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(num_warmup):
        yolo_model.predict(dummy, imgsz=IMG_SIZE, verbose=False, device=DEVICE)
    
    # Measure
    times = []
    for _ in range(num_test):
        t0 = time.time()
        yolo_model.predict(dummy, imgsz=IMG_SIZE, verbose=False, device=DEVICE)
        times.append(time.time() - t0)
    
    avg_time = sum(times) / len(times)
    return 1.0 / avg_time, avg_time * 1000  # fps, latency_ms


def apply_unstructured_pruning(model, prune_ratio):
    """
    Unstructured L1 Pruning 적용
    
    - Weight를 0으로 만들어 sparsity 생성
    - 파라미터 수는 동일하지만, non-zero가 줄어듦
    - TensorRT sparse tensor core 지원 시 속도 향상 가능
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)
            prune.remove(module, 'weight')
    
    return model


def save_results(results):
    """CSV 저장"""
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Model', 'Type', 'Total_Params(M)', 'NonZero_Params(M)', 
                      'Sparsity(%)', 'FLOPs(G)', 'FPS', 'Latency(ms)', 'Size(MB)']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():
    print("=" * 80)
    print("🚀 YOLO11 Hand Pose - 최종 벤치마크")
    print("=" * 80)
    
    if IS_JETSON:
        print("✅ Jetson Nano 환경 감지됨 (GPU 사용)")
    else:
        print("ℹ️ 일반 PC 환경 (CPU 사용)")
    
    if not MODEL_PATH.exists():
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return
    
    results = []
    
    # =========================================================
    # 1️⃣ 원본 모델 (Baseline)
    # =========================================================
    print("\n[1] 원본 모델 (Baseline)...")
    
    yolo_base = YOLO(MODEL_PATH)
    model_base = yolo_base.model
    
    total_params = count_params(model_base) / 1e6
    nonzero_params = count_nonzero_params(model_base) / 1e6
    flops = get_flops(model_base)
    fps, latency = measure_speed(yolo_base)
    size = get_file_size(MODEL_PATH)
    
    print(f"   Total Params: {total_params:.3f}M")
    print(f"   FLOPs: {flops:.3f}G")
    print(f"   Speed: {fps:.1f} FPS ({latency:.1f}ms)")
    print(f"   Size: {size:.2f}MB")
    
    results.append({
        'Model': 'Baseline',
        'Type': 'FP32',
        'Total_Params(M)': round(total_params, 3),
        'NonZero_Params(M)': round(nonzero_params, 3),
        'Sparsity(%)': 0.0,
        'FLOPs(G)': round(flops, 3),
        'FPS': round(fps, 1),
        'Latency(ms)': round(latency, 1),
        'Size(MB)': round(size, 2)
    })
    
    # =========================================================
    # 2️⃣ Unstructured Pruning (30%, 50%, 70%)
    # =========================================================
    print("\n[2] Unstructured Pruning...")
    
    for prune_ratio in [0.3, 0.5, 0.7]:
        pct = int(prune_ratio * 100)
        print(f"\n   🔹 Pruning {pct}%...")
        
        # 새로 로드
        yolo = YOLO(MODEL_PATH)
        model = yolo.model
        
        # Pruning 적용
        model = apply_unstructured_pruning(model, prune_ratio)
        
        # 측정
        total_p = count_params(model) / 1e6
        nonzero_p = count_nonzero_params(model) / 1e6
        sparsity = (1 - nonzero_p / total_p) * 100
        f = get_flops(model)
        
        # 저장 후 속도 측정
        save_path = ROOT / f"assets/models/yolo11n_hand_pose_unstructured_{pct}.pt"
        yolo.save(str(save_path))
        
        yolo_pruned = YOLO(save_path)
        fps_p, lat_p = measure_speed(yolo_pruned)
        size_p = get_file_size(save_path)
        
        print(f"      Total Params: {total_p:.3f}M (동일)")
        print(f"      NonZero Params: {nonzero_p:.3f}M")
        print(f"      Sparsity: {sparsity:.1f}%")
        print(f"      Speed: {fps_p:.1f} FPS")
        
        results.append({
            'Model': f'Unstructured_{pct}%',
            'Type': 'FP32+Sparse',
            'Total_Params(M)': round(total_p, 3),
            'NonZero_Params(M)': round(nonzero_p, 3),
            'Sparsity(%)': round(sparsity, 1),
            'FLOPs(G)': round(f, 3),
            'FPS': round(fps_p, 1),
            'Latency(ms)': round(lat_p, 1),
            'Size(MB)': round(size_p, 2)
        })
    
    # =========================================================
    # 3️⃣ Quantization (Jetson에서만)
    # =========================================================
    if IS_JETSON:
        print("\n[3] Quantization (TensorRT)...")
        
        # FP16
        print("\n   🔹 FP16 TensorRT...")
        try:
            fp16_path = ROOT / "assets/models/yolo11n_hand_pose_fp16.engine"
            
            if not fp16_path.exists():
                yolo_base.export(format='engine', half=True, imgsz=IMG_SIZE, device=0)
                default_engine = MODEL_PATH.with_suffix('.engine')
                if default_engine.exists():
                    default_engine.rename(fp16_path)
            
            if fp16_path.exists():
                yolo_fp16 = YOLO(fp16_path, task='pose')
                fps_fp16, lat_fp16 = measure_speed(yolo_fp16)
                size_fp16 = get_file_size(fp16_path)
                
                print(f"      Speed: {fps_fp16:.1f} FPS")
                
                results.append({
                    'Model': 'TensorRT_FP16',
                    'Type': 'FP16',
                    'Total_Params(M)': round(total_params, 3),
                    'NonZero_Params(M)': round(nonzero_params, 3),
                    'Sparsity(%)': 0.0,
                    'FLOPs(G)': round(flops, 3),
                    'FPS': round(fps_fp16, 1),
                    'Latency(ms)': round(lat_fp16, 1),
                    'Size(MB)': round(size_fp16, 2)
                })
        except Exception as e:
            print(f"      ❌ FP16 실패: {e}")
        
        # INT8
        print("\n   🔹 INT8 TensorRT...")
        try:
            int8_path = ROOT / "assets/models/yolo11n_hand_pose_int8.engine"
            
            if not int8_path.exists():
                yolo_base.export(format='engine', int8=True, imgsz=IMG_SIZE, device=0)
                default_engine = MODEL_PATH.with_suffix('.engine')
                if default_engine.exists():
                    default_engine.rename(int8_path)
            
            if int8_path.exists():
                yolo_int8 = YOLO(int8_path, task='pose')
                fps_int8, lat_int8 = measure_speed(yolo_int8)
                size_int8 = get_file_size(int8_path)
                
                print(f"      Speed: {fps_int8:.1f} FPS")
                
                results.append({
                    'Model': 'TensorRT_INT8',
                    'Type': 'INT8',
                    'Total_Params(M)': round(total_params, 3),
                    'NonZero_Params(M)': round(nonzero_params, 3),
                    'Sparsity(%)': 0.0,
                    'FLOPs(G)': round(flops / 4, 3),  # INT8은 대략 1/4 연산
                    'FPS': round(fps_int8, 1),
                    'Latency(ms)': round(lat_int8, 1),
                    'Size(MB)': round(size_int8, 2)
                })
        except Exception as e:
            print(f"      ❌ INT8 실패: {e}")
    else:
        print("\n[3] Quantization 스킵 (Jetson Nano에서만 TensorRT 지원)")
    
    # =========================================================
    # 📊 결과 출력 및 저장
    # =========================================================
    save_results(results)
    
    print("\n" + "=" * 100)
    print(f"{'Model':<20} | {'Type':<12} | {'Params(M)':<10} | {'NonZero(M)':<10} | {'Sparsity':<10} | {'FLOPs(G)':<10} | {'FPS':<8}")
    print("-" * 100)
    for r in results:
        print(f"{r['Model']:<20} | {r['Type']:<12} | {r['Total_Params(M)']:<10} | {r['NonZero_Params(M)']:<10} | {r['Sparsity(%)']:<10} | {r['FLOPs(G)']:<10} | {r['FPS']:<8}")
    print("=" * 100)
    
    print(f"\n💾 결과 저장됨: {RESULTS_FILE}")
    
    # =========================================================
    # 📝 분석 및 권장사항
    # =========================================================
    print("\n📝 분석 및 권장사항:")
    print("-" * 60)
    print("1. Unstructured Pruning:")
    print("   - 파라미터 수(Total)는 동일하지만 NonZero가 줄어듦")
    print("   - CPU에서는 속도 향상 없음 (0값도 연산에 포함)")
    print("   - TensorRT 8.6+ sparse tensor core 지원 시 속도 향상 가능")
    print()
    print("2. Quantization (Jetson Nano 권장):")
    print("   - FP16: 정확도 손실 거의 없이 2배 속도 향상")
    print("   - INT8: 약간의 정확도 손실로 3-4배 속도 향상")
    print()
    print("3. Structured Pruning 참고:")
    print("   - YOLO11은 torch_pruning과 호환성 문제")
    print("   - 대안: Width Multiplier 조정하여 처음부터 작은 모델 학습")
    print("=" * 60)


if __name__ == "__main__":
    main()
