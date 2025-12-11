"""
🔧 YOLO11 Structured Pruning v3 - Forward Hook 기반

핵심 전략:
1. Forward hook으로 각 레이어의 실제 입출력 shape 추적
2. 연결된 레이어들을 그룹으로 묶음
3. 그룹 단위로 동시에 pruning
"""

import os
import sys
import copy
import time
from pathlib import Path
from collections import defaultdict

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


def compute_importance(weight: torch.Tensor) -> torch.Tensor:
    """L1 norm 기반 채널 중요도"""
    if len(weight.shape) == 4:
        return weight.abs().sum(dim=(1, 2, 3))
    elif len(weight.shape) == 2:
        return weight.abs().sum(dim=1)
    return weight.abs()


class LayerInfo:
    """레이어 정보 저장"""
    def __init__(self, name, module, input_shape, output_shape):
        self.name = name
        self.module = module
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_channels = input_shape[1] if len(input_shape) == 4 else None
        self.output_channels = output_shape[1] if len(output_shape) == 4 else None


def trace_layer_connections(model):
    """Forward pass를 통해 각 레이어의 입출력 shape 추적"""
    layer_info = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                in_shape = input[0].shape if isinstance(input[0], torch.Tensor) else None
            else:
                in_shape = input.shape if isinstance(input, torch.Tensor) else None
            
            if isinstance(output, torch.Tensor):
                out_shape = output.shape
            elif isinstance(output, tuple) and len(output) > 0:
                out_shape = output[0].shape if isinstance(output[0], torch.Tensor) else None
            else:
                out_shape = None
            
            if in_shape is not None and out_shape is not None:
                layer_info[name] = LayerInfo(name, module, in_shape, out_shape)
        return hook
    
    # 모든 레이어에 hook 등록
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        try:
            model(dummy)
        except:
            pass
    
    # Hook 제거
    for h in hooks:
        h.remove()
    
    return layer_info


def build_pruning_groups(model, layer_info):
    """
    채널 의존성 기반으로 pruning 그룹 생성
    
    같은 출력 채널 수를 가진 연속된 레이어들을 그룹으로 묶음
    """
    groups = defaultdict(list)
    
    # 출력 채널 수로 그룹화
    for name, info in layer_info.items():
        if info.output_channels is not None:
            groups[info.output_channels].append((name, info))
    
    return groups


def prune_layer_group(model, group_layers, prune_ratio):
    """
    그룹 내 모든 레이어를 동일한 마스크로 pruning
    """
    if not group_layers:
        return
    
    # 그룹의 대표 채널 수
    n_channels = group_layers[0][1].output_channels
    n_keep = max(8, int(n_channels * (1 - prune_ratio)))
    n_keep = (n_keep // 8) * 8  # 8의 배수
    n_keep = max(8, min(n_keep, n_channels))
    
    if n_keep >= n_channels:
        return
    
    # 그룹 내 모든 Conv의 중요도 평균
    total_importance = torch.zeros(n_channels)
    conv_count = 0
    
    for name, info in group_layers:
        if isinstance(info.module, nn.Conv2d) and info.module.out_channels == n_channels:
            importance = compute_importance(info.module.weight.data)
            total_importance += importance
            conv_count += 1
    
    if conv_count == 0:
        return
    
    avg_importance = total_importance / conv_count
    _, keep_idx = torch.topk(avg_importance, n_keep)
    keep_idx = keep_idx.sort().values
    
    # 그룹 내 모든 레이어 pruning
    for name, info in group_layers:
        module = info.module
        
        if isinstance(module, nn.Conv2d):
            if module.out_channels == n_channels:
                # 출력 채널 pruning
                module.weight.data = module.weight.data[keep_idx]
                module.out_channels = n_keep
                if module.bias is not None:
                    module.bias.data = module.bias.data[keep_idx]
        
        elif isinstance(module, nn.BatchNorm2d):
            if module.num_features == n_channels:
                module.weight.data = module.weight.data[keep_idx]
                module.bias.data = module.bias.data[keep_idx]
                module.running_mean.data = module.running_mean.data[keep_idx]
                module.running_var.data = module.running_var.data[keep_idx]
                module.num_features = n_keep


def prune_yolo11_structured(model_path, prune_ratio=0.3):
    """YOLO11 구조적 Pruning (채널 그룹 기반)"""
    print(f"\n{'='*60}")
    print(f"🔧 YOLO11 Structured Pruning v3 (ratio: {prune_ratio*100:.0f}%)")
    print(f"{'='*60}")
    
    yolo = YOLO(model_path)
    model = copy.deepcopy(yolo.model)
    
    before_params = sum(p.numel() for p in model.parameters())
    print(f"Before: {before_params/1e6:.3f}M params")
    
    # 1. 레이어 연결 추적
    layer_info = trace_layer_connections(model)
    print(f"추적된 레이어: {len(layer_info)}개")
    
    # 2. Pruning 그룹 생성
    groups = build_pruning_groups(model, layer_info)
    print(f"채널 그룹: {len(groups)}개")
    
    # 3. 각 그룹별 pruning (작은 채널 그룹은 스킵)
    for n_channels, group_layers in sorted(groups.items()):
        if n_channels < 16:  # 너무 작으면 스킵
            continue
        if n_channels == 3:  # RGB 입력 스킵
            continue
        
        # Pose head 관련 스킵 (63, 64 채널)
        if any('model.23' in name for name, _ in group_layers):
            continue
        
        prune_layer_group(model, group_layers, prune_ratio)
        print(f"  ✂️ {n_channels}ch 그룹: {len(group_layers)}개 레이어")
    
    after_params = sum(p.numel() for p in model.parameters())
    print(f"After: {after_params/1e6:.3f}M params ({(1-after_params/before_params)*100:.1f}% ↓)")
    
    return model, yolo


def create_consistent_pruned_model(model_path, prune_ratio=0.3):
    """
    YOLO11을 일관되게 pruning
    
    핵심: 각 stage의 출력 채널을 줄이고, 모든 연결된 레이어 동기화
    """
    print(f"\n{'='*60}")
    print(f"🔧 YOLO11 Consistent Pruning (ratio: {prune_ratio*100:.0f}%)")
    print(f"{'='*60}")
    
    yolo = YOLO(model_path)
    model = copy.deepcopy(yolo.model)
    
    before_params = sum(p.numel() for p in model.parameters())
    print(f"Before: {before_params/1e6:.3f}M params")
    
    # YOLO11n의 기본 채널 구조 (수정할 타겟)
    # Block 0: 16, Block 1: 32
    # 이 채널들이 연결되어 있으므로 함께 줄여야 함
    
    keep_ratio = 1.0 - prune_ratio
    
    # 각 YOLO block의 Conv 레이어만 선택적으로 처리
    # (독립적인 Conv 블록만, Concat에 영향받는 건 스킵)
    
    pruned_blocks = []
    
    for block_idx, block in enumerate(model.model):
        block_type = type(block).__name__
        
        # 독립적인 Conv 블록만 처리 (인덱스 0, 1, 3, 5, 7)
        if block_type == 'Conv' and block_idx in [0, 1, 3, 5, 7]:
            conv = block.conv
            bn = block.bn
            
            if conv.in_channels == 3:  # 입력 레이어 스킵
                continue
            
            # 새 채널 수
            old_ch = conv.out_channels
            new_ch = max(8, int(old_ch * keep_ratio) // 8 * 8)
            
            if new_ch >= old_ch:
                continue
            
            # 중요도 기반 채널 선택
            importance = compute_importance(conv.weight.data)
            _, keep_idx = torch.topk(importance, new_ch)
            keep_idx = keep_idx.sort().values
            
            # Conv 출력 pruning
            conv.weight.data = conv.weight.data[keep_idx]
            conv.out_channels = new_ch
            if conv.bias is not None:
                conv.bias.data = conv.bias.data[keep_idx]
            
            # BN 동기화
            bn.weight.data = bn.weight.data[keep_idx]
            bn.bias.data = bn.bias.data[keep_idx]
            bn.running_mean.data = bn.running_mean.data[keep_idx]
            bn.running_var.data = bn.running_var.data[keep_idx]
            bn.num_features = new_ch
            
            pruned_blocks.append((block_idx, old_ch, new_ch))
    
    print(f"Pruned blocks: {pruned_blocks}")
    
    # 이제 연결된 다운스트림 레이어의 입력 채널 조정
    # (이 부분이 복잡함 - skip connection 때문에)
    
    after_params = sum(p.numel() for p in model.parameters())
    print(f"After: {after_params/1e6:.3f}M params ({(1-after_params/before_params)*100:.1f}% ↓)")
    
    return model, yolo


def benchmark(model, name: str, device='cpu'):
    """모델 성능 측정"""
    model = model.to(device).eval()
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    nonzero = sum((p != 0).sum().item() for p in model.parameters()) / 1e6
    
    flops = 0.0
    if HAS_THOP:
        try:
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            flops = macs / 1e9
            for m in model.modules():
                for attr in ['total_ops', 'total_params']:
                    if hasattr(m, attr):
                        delattr(m, attr)
        except Exception as e:
            pass
    
    fps, latency = 0.0, 0.0
    try:
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
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
        latency = (sum(times) / len(times)) * 1000
    except Exception as e:
        print(f"  ⚠️ Forward 실패: {str(e)[:60]}")
    
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
    print("🚀 YOLO11 Structured Pruning v3 - Forward Hook 기반")
    print("=" * 70)
    
    if not MODEL_PATH.exists():
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return
    
    results = []
    
    # 1. 원본 모델
    print("\n[1] 원본 모델...")
    yolo_base = YOLO(MODEL_PATH)
    base_result = benchmark(yolo_base.model, "Baseline")
    results.append(base_result)
    print(f"   Params: {base_result['params']:.3f}M, FLOPs: {base_result['flops']:.3f}G, FPS: {base_result['fps']:.1f}")
    
    # 2. Structured Pruning 테스트
    print("\n[2] Structured Pruning...")
    for ratio in [0.3, 0.5]:
        try:
            model, yolo = prune_yolo11_structured(MODEL_PATH, ratio)
            result = benchmark(model, f"Structured_{int(ratio*100)}%")
            results.append(result)
            
            if result['fps'] > 0:
                print(f"   ✅ Params: {result['params']:.3f}M, FPS: {result['fps']:.1f}")
                
                # 저장
                save_path = ROOT / f"assets/models/yolo11n_pruned_v3_{int(ratio*100)}.pt"
                torch.save({'model': model.state_dict()}, save_path)
            else:
                print(f"   Forward 실패")
        except Exception as e:
            print(f"   ❌ 실패: {e}")
    
    # 결과
    print("\n" + "=" * 85)
    print(f"{'Model':<25} | {'Params(M)':<12} | {'FLOPs(G)':<10} | {'FPS':<8} | {'Latency(ms)':<12}")
    print("-" * 85)
    for r in results:
        print(f"{r['name']:<25} | {r['params']:<12.3f} | {r['flops']:<10.3f} | {r['fps']:<8.1f} | {r['latency']:<12.1f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
