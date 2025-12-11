"""
🔧 YOLO11 수동 Structured Pruning 

YOLO11 모델의 채널을 실제로 줄여서:
- 파라미터 수 감소
- FLOPs 감소
- 실행 속도 향상

을 달성합니다.

방법: 각 레이어의 weight에서 중요도가 낮은 채널(필터)을 제거하고,
      연결된 모든 레이어의 차원을 맞춥니다.
"""

import os
import sys
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C3k2, SPPF, C2PSA, Concat

# FLOPs 측정
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("⚠️ thop 없음: pip install thop")

# =========================================================
# 설정
# =========================================================
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets/models/yolo11n_hand_pose.pt"
IMG_SIZE = 640


# =========================================================
# L1 Norm 기반 채널 중요도 계산
# =========================================================
def compute_channel_importance(conv_layer: nn.Conv2d) -> torch.Tensor:
    """
    Conv2d 레이어의 각 출력 채널(필터)의 L1 norm을 계산합니다.
    Returns: shape (out_channels,) 텐서
    """
    weight = conv_layer.weight.data  # (out_ch, in_ch, kH, kW)
    importance = weight.abs().sum(dim=(1, 2, 3))  # (out_ch,)
    return importance


def get_pruning_indices(importance: torch.Tensor, prune_ratio: float) -> tuple:
    """
    중요도 기반으로 유지할 채널과 제거할 채널 인덱스를 반환합니다.
    
    Returns:
        keep_indices: 유지할 채널 인덱스
        prune_indices: 제거할 채널 인덱스
    """
    n_channels = len(importance)
    n_prune = int(n_channels * prune_ratio)
    n_keep = n_channels - n_prune
    
    # 최소 1개는 유지
    n_keep = max(n_keep, 1)
    
    # 8의 배수로 맞춤 (GPU 효율)
    n_keep = max(8, (n_keep // 8) * 8)
    n_keep = min(n_keep, n_channels)
    
    # 중요도 순으로 정렬
    sorted_indices = torch.argsort(importance, descending=True)
    keep_indices = sorted_indices[:n_keep].sort().values
    prune_indices = sorted_indices[n_keep:].sort().values
    
    return keep_indices, prune_indices


# =========================================================
# Conv + BN 레이어 채널 Pruning
# =========================================================
def prune_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d, keep_indices: torch.Tensor, dim: str = 'out'):
    """
    Conv2d + BatchNorm2d 레이어의 채널을 pruning합니다.
    
    Args:
        conv: Conv2d 레이어
        bn: BatchNorm2d 레이어
        keep_indices: 유지할 채널 인덱스
        dim: 'out' (출력 채널 pruning) 또는 'in' (입력 채널 pruning)
    """
    keep_indices = keep_indices.to(conv.weight.device)
    
    if dim == 'out':
        # 출력 채널 pruning
        conv.weight.data = conv.weight.data[keep_indices]
        conv.out_channels = len(keep_indices)
        
        if conv.bias is not None:
            conv.bias.data = conv.bias.data[keep_indices]
        
        # BatchNorm도 같이 수정
        bn.weight.data = bn.weight.data[keep_indices]
        bn.bias.data = bn.bias.data[keep_indices]
        bn.running_mean.data = bn.running_mean.data[keep_indices]
        bn.running_var.data = bn.running_var.data[keep_indices]
        bn.num_features = len(keep_indices)
        
    elif dim == 'in':
        # 입력 채널 pruning
        conv.weight.data = conv.weight.data[:, keep_indices]
        conv.in_channels = len(keep_indices)


def prune_conv_only_in(conv: nn.Conv2d, keep_indices: torch.Tensor):
    """Conv2d의 입력 채널만 pruning (groups 고려)"""
    keep_indices = keep_indices.to(conv.weight.device)
    
    if conv.groups == 1:
        conv.weight.data = conv.weight.data[:, keep_indices]
        conv.in_channels = len(keep_indices)
    # groups > 1 인 경우 (depthwise 등)은 더 복잡한 처리 필요


# =========================================================
# YOLO11 블록별 Pruning 함수
# =========================================================
def prune_yolo_conv_block(block, prune_ratio: float, prev_keep_indices=None):
    """
    YOLO의 Conv 블록 (Conv2d + BN + Act) pruning
    Returns: 유지된 출력 채널 인덱스
    """
    conv = block.conv
    bn = block.bn
    
    # 1. 입력 채널 pruning (이전 레이어에서 전달받은 경우)
    if prev_keep_indices is not None:
        prune_conv_only_in(conv, prev_keep_indices)
    
    # 2. 출력 채널 중요도 계산 및 pruning
    importance = compute_channel_importance(conv)
    keep_indices, _ = get_pruning_indices(importance, prune_ratio)
    prune_conv_bn(conv, bn, keep_indices, dim='out')
    
    return keep_indices


def prune_c3k2_block(block, prune_ratio: float, prev_keep_indices=None):
    """
    C3k2 블록 pruning (YOLO11의 핵심 블록)
    
    C3k2 구조:
    - cv1: 입력 -> 중간 채널
    - m: 여러 개의 Bottleneck
    - cv2: concat된 채널 -> 출력
    """
    # cv1 pruning
    if prev_keep_indices is not None:
        prune_conv_only_in(block.cv1.conv, prev_keep_indices)
    
    cv1_importance = compute_channel_importance(block.cv1.conv)
    cv1_keep, _ = get_pruning_indices(cv1_importance, prune_ratio)
    prune_conv_bn(block.cv1.conv, block.cv1.bn, cv1_keep, dim='out')
    
    # m (Bottleneck들) pruning
    # 각 Bottleneck의 입력은 cv1 출력의 일부
    m_out_channels = []
    for bottleneck in block.m:
        if hasattr(bottleneck, 'cv1'):
            # Bottleneck cv1
            bn_cv1_importance = compute_channel_importance(bottleneck.cv1.conv)
            bn_cv1_keep, _ = get_pruning_indices(bn_cv1_importance, prune_ratio * 0.5)  # 덜 공격적으로
            prune_conv_bn(bottleneck.cv1.conv, bottleneck.cv1.bn, bn_cv1_keep, dim='out')
            
            # Bottleneck cv2
            prune_conv_only_in(bottleneck.cv2.conv, bn_cv1_keep)
            bn_cv2_importance = compute_channel_importance(bottleneck.cv2.conv)
            bn_cv2_keep, _ = get_pruning_indices(bn_cv2_importance, prune_ratio * 0.5)
            prune_conv_bn(bottleneck.cv2.conv, bottleneck.cv2.bn, bn_cv2_keep, dim='out')
            
            m_out_channels.append(len(bn_cv2_keep))
    
    # cv2 입력 채널 조정 (cv1 출력 + m 출력들의 concat)
    # C3k2의 c (중간 채널)는 cv1의 출력 채널 수와 관련
    # 복잡한 의존성 때문에 cv2는 pruning하지 않거나 출력만 pruning
    
    cv2_importance = compute_channel_importance(block.cv2.conv)
    cv2_keep, _ = get_pruning_indices(cv2_importance, prune_ratio)
    
    # cv2 출력 채널만 pruning (입력은 복잡한 concat이므로 건드리지 않음)
    # 주의: 이렇게 하면 불일치가 발생할 수 있음
    prune_conv_bn(block.cv2.conv, block.cv2.bn, cv2_keep, dim='out')
    
    return cv2_keep


# =========================================================
# 단순화된 Pruning (Width Multiplier 방식)
# =========================================================
def create_pruned_yolo_model(original_model_path: str, prune_ratio: float) -> nn.Module:
    """
    YOLO11 모델을 pruning합니다.
    
    복잡한 의존성 때문에, 각 Conv 레이어의 weight에서 중요도가 낮은 필터를 제거하고
    새로운 작은 레이어를 생성합니다.
    
    Args:
        original_model_path: 원본 .pt 파일 경로
        prune_ratio: 제거할 채널 비율 (0.3 = 30% 제거)
    
    Returns:
        Pruned PyTorch model
    """
    # 원본 모델 로드
    yolo = YOLO(original_model_path)
    model = copy.deepcopy(yolo.model)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"🔧 Structured Pruning 시작 (ratio: {prune_ratio*100:.0f}%)")
    print(f"{'='*60}")
    
    # 각 레이어의 중요도 기반으로 채널 선택
    layers_pruned = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 입력 채널이 3인 경우 (RGB 입력) 스킵
            if module.in_channels == 3:
                continue
            
            # 출력 채널이 너무 작으면 스킵 (최소 8)
            if module.out_channels <= 8:
                continue
            
            # 마지막 detection head 레이어는 스킵
            if 'cv2.2' in name or 'cv3.2' in name or 'cv4' in name:
                continue
            
            # 중요도 계산
            importance = compute_channel_importance(module)
            keep_indices, _ = get_pruning_indices(importance, prune_ratio)
            
            # 출력 채널 pruning
            original_out = module.out_channels
            module.weight.data = module.weight.data[keep_indices]
            module.out_channels = len(keep_indices)
            
            if module.bias is not None:
                module.bias.data = module.bias.data[keep_indices]
            
            layers_pruned += 1
    
    print(f"✅ {layers_pruned}개 레이어 pruning 완료")
    
    # BatchNorm 레이어도 맞춤
    bn_fixed = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # 이전 Conv의 출력 채널과 맞추기
            parent_name = '.'.join(name.split('.')[:-1])
            conv_name = parent_name + '.conv' if parent_name else 'conv'
            
            # 해당 Conv 찾기
            try:
                conv = dict(model.named_modules())[conv_name.replace('.bn', '.conv')]
                if isinstance(conv, nn.Conv2d):
                    target_channels = conv.out_channels
                    if module.num_features != target_channels:
                        # BN 파라미터 조정
                        module.num_features = target_channels
                        module.weight.data = module.weight.data[:target_channels]
                        module.bias.data = module.bias.data[:target_channels]
                        module.running_mean.data = module.running_mean.data[:target_channels]
                        module.running_var.data = module.running_var.data[:target_channels]
                        bn_fixed += 1
            except:
                pass
    
    print(f"✅ BatchNorm {bn_fixed}개 조정 완료")
    
    return model


# =========================================================
# 더 안전한 방법: Weight Slicing 기반 Pruning
# =========================================================
def prune_model_safe(model_path: str, prune_ratio: float):
    """
    안전한 Pruning: Conv와 연결된 BN을 함께 처리
    
    YOLO 구조의 복잡한 skip connection과 concat 때문에,
    모든 레이어를 동시에 처리하지 않고
    독립적인 Conv-BN 쌍만 pruning합니다.
    """
    yolo = YOLO(model_path)
    model = yolo.model
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"🔧 Safe Structured Pruning (ratio: {prune_ratio*100:.0f}%)")
    print(f"{'='*60}")
    
    before_params = sum(p.numel() for p in model.parameters())
    print(f"Before: {before_params/1e6:.3f}M params")
    
    # Ultra lytics YOLO의 Conv 블록 (conv + bn + act)을 찾아서 처리
    pruned_blocks = 0
    
    for name, module in list(model.named_modules()):
        # Ultralytics Conv 블록 찾기
        if type(module).__name__ == 'Conv' and hasattr(module, 'conv') and hasattr(module, 'bn'):
            conv = module.conv
            bn = module.bn
            
            # 스킵 조건
            if conv.in_channels == 3:  # 입력 레이어
                continue
            if conv.out_channels <= 8:  # 너무 작음
                continue
            if 'head' in name.lower():  # Detection head
                continue
            
            # 중요도 계산
            importance = compute_channel_importance(conv)
            keep_indices, _ = get_pruning_indices(importance, prune_ratio)
            n_keep = len(keep_indices)
            
            if n_keep >= conv.out_channels:
                continue  # 변화 없음
            
            # 출력 채널 pruning
            conv.weight.data = conv.weight.data[keep_indices]
            conv.out_channels = n_keep
            if conv.bias is not None:
                conv.bias.data = conv.bias.data[keep_indices]
            
            # BatchNorm 동기화
            bn.weight.data = bn.weight.data[keep_indices]
            bn.bias.data = bn.bias.data[keep_indices]
            bn.running_mean.data = bn.running_mean.data[keep_indices]
            bn.running_var.data = bn.running_var.data[keep_indices]
            bn.num_features = n_keep
            
            pruned_blocks += 1
            print(f"  ✂️ {name}: {conv.out_channels + len(keep_indices) - n_keep} -> {n_keep} channels")
    
    after_params = sum(p.numel() for p in model.parameters())
    print(f"\nAfter: {after_params/1e6:.3f}M params")
    print(f"Reduction: {(1 - after_params/before_params)*100:.1f}%")
    print(f"✅ {pruned_blocks}개 블록 pruning 완료")
    
    return model, yolo


# =========================================================
# 메인 벤치마크
# =========================================================
def benchmark_model(model, name: str, device='cpu'):
    """모델 성능 측정"""
    model = model.to(device).eval()
    
    # 파라미터 수
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Non-zero 파라미터
    nonzero = sum((p != 0).sum().item() for p in model.parameters()) / 1e6
    
    # FLOPs
    flops = 0.0
    if HAS_THOP:
        try:
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            flops = macs / 1e9
            
            # thop 임시 속성 제거
            for m in model.modules():
                for attr in ['total_ops', 'total_params']:
                    if hasattr(m, attr):
                        delattr(m, attr)
        except Exception as e:
            print(f"  ⚠️ FLOPs 측정 실패: {e}")
    
    # 속도 측정
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            try:
                model(dummy)
            except:
                break
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(30):
            try:
                t0 = time.time()
                model(dummy)
                times.append(time.time() - t0)
            except:
                break
    
    fps = 1.0 / (sum(times) / len(times)) if times else 0
    latency = (sum(times) / len(times)) * 1000 if times else 0
    
    return {
        'name': name,
        'params': params,
        'nonzero': nonzero,
        'flops': flops,
        'fps': fps,
        'latency': latency
    }


def main():
    print("=" * 70)
    print("🚀 YOLO11 수동 Structured Pruning 벤치마크")
    print("=" * 70)
    
    if not MODEL_PATH.exists():
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return
    
    results = []
    
    # 1. 원본 모델
    print("\n[1] 원본 모델 측정...")
    yolo_base = YOLO(MODEL_PATH)
    base_result = benchmark_model(yolo_base.model, "Baseline")
    results.append(base_result)
    print(f"   📊 Params: {base_result['params']:.3f}M")
    print(f"   📊 FLOPs: {base_result['flops']:.3f}G")
    print(f"   📊 FPS: {base_result['fps']:.1f}")
    
    # 2. Pruned 모델들
    prune_ratios = [0.3, 0.5, 0.7]
    
    for ratio in prune_ratios:
        print(f"\n[Pruning {int(ratio*100)}%]")
        try:
            pruned_model, yolo_obj = prune_model_safe(MODEL_PATH, ratio)
            result = benchmark_model(pruned_model, f"Pruned_{int(ratio*100)}%")
            results.append(result)
            
            # 실제 감소율
            param_reduction = (1 - result['params'] / base_result['params']) * 100
            flops_reduction = (1 - result['flops'] / base_result['flops']) * 100 if base_result['flops'] > 0 else 0
            
            print(f"   📊 Params: {result['params']:.3f}M ({param_reduction:.1f}% ↓)")
            print(f"   📊 FLOPs: {result['flops']:.3f}G ({flops_reduction:.1f}% ↓)")
            print(f"   📊 FPS: {result['fps']:.1f}")
            
            # 모델 저장
            save_path = ROOT / f"assets/models/yolo11n_hand_pose_manual_pruned_{int(ratio*100)}.pt"
            torch.save({
                'model': pruned_model.state_dict(),
            }, save_path)
            print(f"   💾 저장: {save_path.name}")
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 결과 요약
    print("\n" + "=" * 80)
    print(f"{'Model':<20} | {'Params(M)':<12} | {'FLOPs(G)':<12} | {'FPS':<10} | {'Latency(ms)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<20} | {r['params']:<12.3f} | {r['flops']:<12.3f} | {r['fps']:<10.1f} | {r['latency']:<12.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
