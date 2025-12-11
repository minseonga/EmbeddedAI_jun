"""
🔧 YOLO11 Structured Pruning - 공식 torch_pruning 예제 기반

핵심 기술:
1. C3k2 블록을 pruning-friendly 버전으로 교체
2. GroupNormPruner 사용
3. Detect/Pose head는 ignored_layers로 지정

참고: https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/yolov8_pruning.py
"""

import os
import sys
import copy
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C3k2, Bottleneck, SPPF, C2PSA

import torch_pruning as tp

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets/models/yolo11n_hand_pose.pt"
IMG_SIZE = 640


# =========================================================
# C3k2 블록을 Pruning-Friendly 버전으로 교체
# =========================================================

def infer_shortcut(bottleneck):
    """Bottleneck의 shortcut 연결 여부 확인"""
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add


class C3k2_v2(nn.Module):
    """
    C3k2의 Pruning-Friendly 버전
    
    원본 C3k2는 내부적으로 torch.chunk를 사용하여 채널을 나누는데,
    이것이 torch_pruning의 그래프 추적을 방해합니다.
    
    이 버전은 chunk 대신 두 개의 별도 Conv를 사용합니다.
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        
        # YOLO 내부에서 요구하는 속성들
        self.f = -1  # from (이전 레이어 인덱스)
        self.i = 0   # index (현재 레이어 인덱스)
        self.np = 0  # number of parameters
        self.type = type(self).__name__
        
        # 두 개의 별도 Conv (chunk 대신)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        # Bottleneck들
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
    
    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_c3k2_weights(old_block, new_block):
    """C3k2 -> C3k2_v2 weight 전송"""
    # YOLO 필수 속성 복사
    if hasattr(old_block, 'f'):
        new_block.f = old_block.f
    if hasattr(old_block, 'i'):
        new_block.i = old_block.i
    if hasattr(old_block, 'np'):
        new_block.np = old_block.np
    
    new_block.cv2 = old_block.cv2
    new_block.m = old_block.m
    
    state_dict = old_block.state_dict()
    state_dict_v2 = new_block.state_dict()
    
    # cv1의 weight를 cv0와 cv1으로 분할
    if 'cv1.conv.weight' in state_dict:
        old_weight = state_dict['cv1.conv.weight']
        half = old_weight.shape[0] // 2
        state_dict_v2['cv0.conv.weight'] = old_weight[:half]
        state_dict_v2['cv1.conv.weight'] = old_weight[half:]
        
        # BatchNorm도 분할
        for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
            key = f'cv1.bn.{bn_key}'
            if key in state_dict:
                old_bn = state_dict[key]
                state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half]
                state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half:]
    
    # 나머지 weight 복사
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]
    
    new_block.load_state_dict(state_dict_v2, strict=False)


def replace_c3k2_with_v2(module):
    """모델의 모든 C3k2를 C3k2_v2로 교체"""
    for name, child in module.named_children():
        if isinstance(child, C3k2):
            # C3k2_v2 생성
            c1 = child.cv1.conv.in_channels
            c2 = child.cv2.conv.out_channels
            n = len(child.m)
            c = child.c
            
            shortcut = infer_shortcut(child.m[0]) if len(child.m) > 0 else False
            e = c / c2 if c2 > 0 else 0.5
            
            new_block = C3k2_v2(c1, c2, n=n, e=e, shortcut=shortcut)
            transfer_c3k2_weights(child, new_block)
            
            setattr(module, name, new_block)
        else:
            replace_c3k2_with_v2(child)


# =========================================================
# Pruning 함수
# =========================================================

def prune_yolo11(model_path, prune_ratio=0.3, device='cpu'):
    """
    YOLO11 모델 Structured Pruning
    
    Args:
        model_path: .pt 파일 경로
        prune_ratio: 제거할 비율 (0.3 = 30%)
        device: 'cpu' 또는 'cuda'
    
    Returns:
        pruned model
    """
    print(f"\n{'='*70}")
    print(f"🔧 YOLO11 Structured Pruning (공식 방법)")
    print(f"   Prune ratio: {prune_ratio*100:.0f}%")
    print(f"{'='*70}")
    
    # 모델 로드
    yolo = YOLO(model_path)
    model = copy.deepcopy(yolo.model).to(device)
    model.train()  # pruning은 train 모드에서
    
    # Before 측정
    example_inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    before_macs, before_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"\nBefore: {before_params/1e6:.3f}M params, {before_macs/1e9:.3f}G MACs")
    
    # C3k2 -> C3k2_v2 교체 (pruning-friendly)
    print("\n[1] C3k2 -> C3k2_v2 교체...")
    replace_c3k2_with_v2(model)
    print("   ✅ 교체 완료")
    
    # Forward 테스트
    model.eval()
    try:
        with torch.no_grad():
            _ = model(example_inputs)
        print("   ✅ Forward 성공")
    except Exception as e:
        print(f"   ❌ Forward 실패: {e}")
        return None
    
    # Ignored layers 설정 (Pose/Detect head)
    ignored_layers = []
    for m in model.modules():
        # Pose/Detect head 모듈
        if type(m).__name__ in ['Detect', 'Pose', 'Segment']:
            ignored_layers.append(m)
    
    print(f"\n[2] Pruning 설정...")
    print(f"   ignored_layers: {len(ignored_layers)}개")
    
    # Pruner 생성
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    
    try:
        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(p=2),
            iterative_steps=1,
            pruning_ratio=prune_ratio,
            ignored_layers=ignored_layers,
            round_to=8,  # GPU 효율
        )
        
        print("\n[3] Pruning 실행...")
        pruner.step()
        print("   ✅ Pruning 완료")
        
    except Exception as e:
        print(f"   ❌ Pruning 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # After 측정
    model.eval()
    after_macs, after_params = tp.utils.count_ops_and_params(model, example_inputs)
    
    print(f"\nAfter: {after_params/1e6:.3f}M params, {after_macs/1e9:.3f}G MACs")
    print(f"Reduction: Params {(1-after_params/before_params)*100:.1f}%, MACs {(1-after_macs/before_macs)*100:.1f}%")
    
    # Forward 테스트
    try:
        with torch.no_grad():
            _ = model(example_inputs)
        print("✅ Pruned model forward 성공!")
    except Exception as e:
        print(f"❌ Pruned model forward 실패: {e}")
        return None
    
    return model


def benchmark(model, name: str, device='cpu'):
    """모델 성능 측정"""
    if model is None:
        return {'name': name, 'params': 0, 'flops': 0, 'fps': 0, 'latency': 0}
    
    model = model.to(device).eval()
    
    # 파라미터
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # FLOPs
    flops = 0.0
    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    try:
        macs, _ = tp.utils.count_ops_and_params(model, example)
        flops = macs / 1e9
    except:
        if HAS_THOP:
            try:
                macs, _ = profile(model, inputs=(example,), verbose=False)
                flops = macs / 1e9
            except:
                pass
    
    # 속도
    fps, latency = 0.0, 0.0
    try:
        with torch.no_grad():
            for _ in range(5):
                model(example)
        
        times = []
        with torch.no_grad():
            for _ in range(20):
                t0 = time.time()
                model(example)
                times.append(time.time() - t0)
        
        fps = 1.0 / (sum(times) / len(times))
        latency = (sum(times) / len(times)) * 1000
    except Exception as e:
        print(f"  ⚠️ 속도 측정 실패: {str(e)[:50]}")
    
    return {
        'name': name,
        'params': params,
        'flops': flops,
        'fps': fps,
        'latency': latency
    }


def main():
    print("=" * 70)
    print("🚀 YOLO11 Structured Pruning Benchmark (공식 torch_pruning 방법)")
    print("=" * 70)
    
    if not MODEL_PATH.exists():
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return
    
    results = []
    device = 'cpu'
    
    # 1. 원본 모델
    print("\n[1] 원본 모델...")
    yolo_base = YOLO(MODEL_PATH)
    base_result = benchmark(yolo_base.model, "Baseline", device)
    results.append(base_result)
    print(f"   Params: {base_result['params']:.3f}M, FLOPs: {base_result['flops']:.3f}G, FPS: {base_result['fps']:.1f}")
    
    # 2. Pruning
    for ratio in [0.3, 0.5, 0.7]:
        print(f"\n[Pruning {int(ratio*100)}%]")
        pruned_model = prune_yolo11(MODEL_PATH, prune_ratio=ratio, device=device)
        
        if pruned_model is not None:
            result = benchmark(pruned_model, f"Pruned_{int(ratio*100)}%", device)
            results.append(result)
            
            # 저장
            save_path = ROOT / f"assets/models/yolo11n_pruned_official_{int(ratio*100)}.pt"
            torch.save({
                'model': pruned_model.state_dict(),
                'prune_ratio': ratio,
            }, save_path)
            print(f"   💾 저장: {save_path.name}")
        else:
            results.append({'name': f"Pruned_{int(ratio*100)}%", 'params': 0, 'flops': 0, 'fps': 0, 'latency': 0})
    
    # 결과
    print("\n" + "=" * 85)
    print(f"{'Model':<25} | {'Params(M)':<12} | {'FLOPs(G)':<12} | {'FPS':<10} | {'Latency(ms)':<12}")
    print("-" * 85)
    for r in results:
        print(f"{r['name']:<25} | {r['params']:<12.3f} | {r['flops']:<12.3f} | {r['fps']:<10.1f} | {r['latency']:<12.1f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
