"""
🔧 YOLO11 BN-Based Structured Pruning

이 방법은 BatchNorm의 gamma(scale) 파라미터를 기반으로 
중요하지 않은 채널을 식별하고 제거합니다.

핵심 아이디어:
- BN의 gamma가 작으면 해당 채널은 중요하지 않음
- gamma < threshold인 채널을 제거
- Conv-BN 쌍을 함께 처리

참고 논문: "Learning Efficient Convolutional Networks through Network Slimming"
https://arxiv.org/abs/1708.06519
"""

import os
import sys
import copy
import time
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from ultralytics import YOLO

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets/models/yolo11n_hand_pose.pt"
IMG_SIZE = 640


# =========================================================
# BN Gamma 기반 채널 중요도 분석
# =========================================================

def analyze_bn_gamma(model):
    """모든 BN 레이어의 gamma 분석"""
    bn_info = []
    
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            gamma = m.weight.data.abs()
            bn_info.append({
                'name': name,
                'module': m,
                'channels': m.num_features,
                'gamma': gamma,
                'gamma_mean': gamma.mean().item(),
                'gamma_sorted': gamma.sort()[0],
            })
    
    return bn_info


def get_pruning_threshold(model, prune_ratio):
    """
    전체 BN gamma를 모아서 prune_ratio에 해당하는 threshold 계산
    
    예: prune_ratio=0.3 → 하위 30% gamma 값을 threshold로 설정
    """
    all_gamma = []
    
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            all_gamma.append(m.weight.data.abs().clone())
    
    all_gamma = torch.cat(all_gamma)
    sorted_gamma, _ = torch.sort(all_gamma)
    
    # 하위 prune_ratio% 에 해당하는 값
    threshold_idx = int(len(sorted_gamma) * prune_ratio)
    threshold = sorted_gamma[threshold_idx].item()
    
    return threshold


def get_channel_mask(bn_layer, threshold, min_channels=8):
    """
    BN layer의 gamma 기반으로 유지할 채널 마스크 생성
    
    Returns:
        mask: 유지할 채널 True, 제거할 채널 False
        keep_indices: 유지할 채널 인덱스
    """
    gamma = bn_layer.weight.data.abs()
    mask = gamma > threshold
    
    # 최소 채널 수 보장
    if mask.sum() < min_channels:
        # 가장 큰 gamma를 가진 min_channels 개 유지
        _, indices = torch.topk(gamma, min_channels)
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask[indices] = True
    
    keep_indices = mask.nonzero().squeeze(-1)
    
    return mask, keep_indices


# =========================================================
# Conv-BN 쌍 Pruning
# =========================================================

def prune_conv_bn_pair(conv, bn, keep_indices, next_conv=None):
    """
    Conv + BN 쌍의 출력 채널을 pruning
    
    Args:
        conv: Conv2d 레이어
        bn: BatchNorm2d 레이어
        keep_indices: 유지할 채널 인덱스
        next_conv: 다음 Conv2d (입력 채널 조정용)
    """
    n_keep = len(keep_indices)
    
    # Conv 출력 채널 pruning
    conv.weight.data = conv.weight.data[keep_indices]
    conv.out_channels = n_keep
    if conv.bias is not None:
        conv.bias.data = conv.bias.data[keep_indices]
    
    # BN 동기화
    bn.weight.data = bn.weight.data[keep_indices]
    bn.bias.data = bn.bias.data[keep_indices]
    bn.running_mean.data = bn.running_mean.data[keep_indices]
    bn.running_var.data = bn.running_var.data[keep_indices]
    bn.num_features = n_keep
    
    # 다음 Conv의 입력 채널 조정
    if next_conv is not None and next_conv.groups == 1:
        next_conv.weight.data = next_conv.weight.data[:, keep_indices]
        next_conv.in_channels = n_keep


# =========================================================
# YOLO11 전체 모델 Pruning
# =========================================================

def prune_yolo11_bn_based(model_path, prune_ratio=0.3):
    """
    BN Gamma 기반 YOLO11 Structured Pruning
    
    주의: YOLO11의 skip connection과 Concat 때문에
    일부 레이어만 안전하게 pruning 가능
    """
    print(f"\n{'='*70}")
    print(f"🔧 YOLO11 BN-Based Structured Pruning")
    print(f"   Prune ratio: {prune_ratio*100:.0f}%")
    print(f"{'='*70}")
    
    # 모델 로드
    yolo = YOLO(model_path)
    model = copy.deepcopy(yolo.model)
    model.eval()
    
    # Before 측정
    before_params = sum(p.numel() for p in model.parameters())
    print(f"\nBefore: {before_params/1e6:.3f}M params")
    
    # Threshold 계산
    threshold = get_pruning_threshold(model, prune_ratio)
    print(f"Gamma threshold: {threshold:.4f}")
    
    # BN gamma 분석
    bn_info = analyze_bn_gamma(model)
    print(f"총 BN 레이어: {len(bn_info)}개")
    
    # === 안전한 Pruning: 독립적인 Conv-BN 쌍만 처리 ===
    # YOLO의 첫 몇 개 레이어 (Concat/Skip 영향 없는 부분)
    
    pruned_count = 0
    
    # model.model 내의 각 블록 처리
    for block_idx, block in enumerate(model.model):
        block_type = type(block).__name__
        
        # 독립적인 Conv 블록만 처리 (block_idx 0, 1만 - 나머지는 skip connection 영향)
        if block_type == 'Conv' and block_idx <= 1:
            conv = block.conv
            bn = block.bn
            
            # RGB 입력 스킵
            if conv.in_channels == 3:
                continue
            
            # 채널이 너무 작으면 스킵
            if conv.out_channels <= 16:
                continue
            
            # Gamma 기반 마스크 생성
            mask, keep_indices = get_channel_mask(bn, threshold, min_channels=8)
            
            if len(keep_indices) < conv.out_channels:
                old_ch = conv.out_channels
                
                # Pruning 적용
                prune_conv_bn_pair(conv, bn, keep_indices)
                
                pruned_count += 1
                print(f"  ✂️ Block {block_idx} ({block_type}): {old_ch} -> {len(keep_indices)} channels")
    
    print(f"\n총 {pruned_count}개 블록 pruned")
    
    # After 측정
    after_params = sum(p.numel() for p in model.parameters())
    print(f"After: {after_params/1e6:.3f}M params")
    print(f"Reduction: {(1-after_params/before_params)*100:.1f}%")
    
    # Forward 테스트
    try:
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            output = model(dummy)
        print("✅ Forward 성공!")
        return model, True
    except Exception as e:
        print(f"❌ Forward 실패: {e}")
        return model, False


# =========================================================
# 더 공격적인 방법: Filter Reconstruction
# =========================================================

def slim_yolo_by_width(model_path, width_mult=0.5):
    """
    YOLO 모델의 width를 줄여 더 작은 모델 생성
    
    이 방법은:
    1. 각 Conv의 출력 채널을 width_mult 비율로 줄임
    2. 다음 레이어의 입력 채널도 맞춤
    3. 새 모델을 생성하고 가장 중요한 채널의 weight 복사
    
    주의: 학습 없이는 정확도가 떨어질 수 있음
    """
    print(f"\n{'='*70}")
    print(f"🔧 YOLO11 Width Slimming")
    print(f"   Width multiplier: {width_mult}")
    print(f"{'='*70}")
    
    yolo = YOLO(model_path)
    model = yolo.model.eval()
    
    before_params = sum(p.numel() for p in model.parameters())
    print(f"Before: {before_params/1e6:.3f}M params")
    
    # 모델 구조 정보 가져오기
    yaml_cfg = model.yaml
    print(f"Original scale: {yaml_cfg.get('scale', 'n')}")
    
    # scales 수정으로 더 작은 모델 생성은 YOLO 재학습이 필요
    # 대신, 현재 weight에서 중요한 채널만 선택하는 방식 사용
    
    slim_count = 0
    
    for name, module in model.named_modules():
        if type(module).__name__ == 'Conv' and hasattr(module, 'conv') and hasattr(module, 'bn'):
            conv = module.conv
            bn = module.bn
            
            # 스킵 조건
            if conv.in_channels == 3:
                continue
            if conv.out_channels <= 8:
                continue
            
            # Detection head 스킵
            if 'model.23' in name:
                continue
            
            # 새 채널 수
            old_ch = conv.out_channels
            new_ch = max(8, int(old_ch * width_mult))
            new_ch = (new_ch // 8) * 8  # 8의 배수
            new_ch = max(8, min(new_ch, old_ch))
            
            if new_ch < old_ch:
                # BN gamma 기반 중요 채널 선택
                gamma = bn.weight.data.abs()
                _, keep_indices = torch.topk(gamma, new_ch)
                keep_indices = keep_indices.sort().values
                
                # Pruning
                conv.weight.data = conv.weight.data[keep_indices]
                conv.out_channels = new_ch
                if conv.bias is not None:
                    conv.bias.data = conv.bias.data[keep_indices]
                
                bn.weight.data = bn.weight.data[keep_indices]
                bn.bias.data = bn.bias.data[keep_indices]
                bn.running_mean.data = bn.running_mean.data[keep_indices]
                bn.running_var.data = bn.running_var.data[keep_indices]
                bn.num_features = new_ch
                
                slim_count += 1
    
    print(f"총 {slim_count}개 레이어 slimmed")
    
    after_params = sum(p.numel() for p in model.parameters())
    print(f"After: {after_params/1e6:.3f}M params")
    print(f"Reduction: {(1-after_params/before_params)*100:.1f}%")
    
    return model


# =========================================================
# 가장 현실적인 방법: Sparsity Training + Pruning
# =========================================================

def train_with_sparsity(model, sparsity_lambda=1e-4):
    """
    BN gamma에 L1 regularization을 추가하여 sparsity 유도
    
    학습 시 loss += sparsity_lambda * sum(|gamma|)
    
    이렇게 학습하면 불필요한 채널의 gamma가 0에 가까워짐
    → 이후 pruning이 더 효과적
    """
    # 이 함수는 학습 코드에 통합되어야 함
    sparsity_loss = 0
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            sparsity_loss += m.weight.abs().sum()
    
    return sparsity_loss * sparsity_lambda


# =========================================================
# 벤치마크
# =========================================================

def benchmark(model, name, device='cpu'):
    if model is None:
        return {'name': name, 'params': 0, 'flops': 0, 'fps': 0}
    
    model = model.to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    flops = 0.0
    try:
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        if HAS_THOP:
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            flops = macs / 1e9
            for m in model.modules():
                for attr in ['total_ops', 'total_params']:
                    if hasattr(m, attr):
                        delattr(m, attr)
    except:
        pass
    
    fps = 0.0
    try:
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            for _ in range(5):
                model(dummy)
        
        times = []
        with torch.no_grad():
            for _ in range(20):
                t0 = time.time()
                model(dummy)
                times.append(time.time() - t0)
        fps = 1.0 / (sum(times) / len(times))
    except:
        pass
    
    return {'name': name, 'params': params, 'flops': flops, 'fps': fps}


def main():
    print("=" * 70)
    print("🚀 YOLO11 대체 Pruning 방법 테스트")
    print("=" * 70)
    
    if not MODEL_PATH.exists():
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return
    
    results = []
    
    # 1. Baseline
    print("\n[1] Baseline...")
    yolo_base = YOLO(MODEL_PATH)
    base_result = benchmark(yolo_base.model, "Baseline")
    results.append(base_result)
    print(f"   Params: {base_result['params']:.3f}M, FLOPs: {base_result['flops']:.3f}G")
    
    # 2. BN-based Pruning
    print("\n[2] BN-Based Pruning...")
    for ratio in [0.3, 0.5]:
        model, success = prune_yolo11_bn_based(MODEL_PATH, prune_ratio=ratio)
        if success:
            result = benchmark(model, f"BN_Pruning_{int(ratio*100)}%")
            results.append(result)
            print(f"   Params: {result['params']:.3f}M, FPS: {result['fps']:.1f}")
    
    # 3. Width Slimming
    print("\n[3] Width Slimming...")
    for mult in [0.75, 0.5]:
        try:
            model = slim_yolo_by_width(MODEL_PATH, width_mult=mult)
            
            # Forward 테스트
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            try:
                with torch.no_grad():
                    model(dummy)
                result = benchmark(model, f"Width_{int(mult*100)}%")
                results.append(result)
                print(f"   Params: {result['params']:.3f}M, FPS: {result['fps']:.1f}")
            except Exception as e:
                print(f"   ❌ Forward 실패: {str(e)[:50]}")
        except Exception as e:
            print(f"   ❌ Slimming 실패: {e}")
    
    # 결과
    print("\n" + "=" * 70)
    print(f"{'Model':<25} | {'Params(M)':<12} | {'FLOPs(G)':<12} | {'FPS':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} | {r['params']:<12.3f} | {r['flops']:<12.3f} | {r['fps']:<10.1f}")
    print("=" * 70)
    
    print("\n📝 참고:")
    print("- BN-Based Pruning: 안전한 레이어만 pruning (일부만 감소)")
    print("- Width Slimming: 채널 수를 줄이지만 forward 호환성 문제 가능")
    print("- 최선의 방법: Sparsity Training 후 Pruning (재학습 필요)")


if __name__ == "__main__":
    main()
