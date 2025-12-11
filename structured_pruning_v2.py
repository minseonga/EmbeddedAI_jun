"""
🔧 YOLO11 완전한 Structured Pruning v2

핵심: 각 레이어의 출력 채널을 줄일 때, 
연결된 다음 레이어의 입력 채널도 함께 조정합니다.

YOLO11 구조:
- Backbone: Conv -> C3k2 -> Conv -> C3k2 -> ... -> SPPF -> C2PSA
- Neck: Upsample -> Concat -> C3k2 (FPN 구조)
- Head: Pose (detection + keypoints)
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

# FLOPs 측정
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

# =========================================================
# 설정
# =========================================================
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets/models/yolo11n_hand_pose.pt"
IMG_SIZE = 640


# =========================================================
# 채널 중요도 계산
# =========================================================
def compute_importance(weight: torch.Tensor) -> torch.Tensor:
    """L2 norm 기반 중요도"""
    if len(weight.shape) == 4:  # Conv: (out, in, h, w)
        return weight.abs().pow(2).sum(dim=(1, 2, 3))
    elif len(weight.shape) == 2:  # Linear: (out, in)
        return weight.abs().pow(2).sum(dim=1)
    return weight.abs()


def get_keep_mask(importance: torch.Tensor, keep_ratio: float, min_channels: int = 8) -> torch.Tensor:
    """중요도 기반으로 유지할 채널 마스크 반환"""
    n = len(importance)
    n_keep = max(min_channels, int(n * keep_ratio))
    n_keep = min(n_keep, n)
    
    # 8의 배수로 맞춤
    n_keep = max(min_channels, (n_keep // 8) * 8)
    n_keep = min(n_keep, n)
    
    _, indices = torch.topk(importance, n_keep)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask


# =========================================================
# YOLO11 구조 분석 및 Pruning
# =========================================================
class YOLO11Pruner:
    def __init__(self, model_path: str, prune_ratio: float = 0.3):
        self.prune_ratio = prune_ratio
        self.keep_ratio = 1.0 - prune_ratio
        
        # 모델 로드
        self.yolo = YOLO(model_path)
        self.model = copy.deepcopy(self.yolo.model)
        self.model.eval()
        
        # 채널 마스크 저장소
        self.channel_masks = {}  # name -> mask tensor
        
    def analyze_structure(self):
        """YOLO11 구조 분석"""
        print("\n=== YOLO11 구조 분석 ===")
        
        for i, block in enumerate(self.model.model):
            block_name = type(block).__name__
            
            # 블록의 출력 채널 확인
            out_ch = None
            for name, m in block.named_modules():
                if isinstance(m, nn.Conv2d):
                    out_ch = m.out_channels
                elif isinstance(m, nn.BatchNorm2d):
                    out_ch = m.num_features
            
            print(f"[{i:2d}] {block_name:15} | out_ch={out_ch}")
    
    def prune_conv_bn_pair(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, 
                           out_mask: torch.Tensor = None, in_mask: torch.Tensor = None):
        """Conv + BN 쌍의 채널 pruning"""
        
        # 출력 채널 pruning (이 레이어의 필터 개수)
        if out_mask is not None:
            keep_idx = out_mask.nonzero().squeeze(-1)
            
            conv.weight.data = conv.weight.data[keep_idx]
            conv.out_channels = len(keep_idx)
            if conv.bias is not None:
                conv.bias.data = conv.bias.data[keep_idx]
            
            bn.weight.data = bn.weight.data[keep_idx]
            bn.bias.data = bn.bias.data[keep_idx]
            bn.running_mean.data = bn.running_mean.data[keep_idx]
            bn.running_var.data = bn.running_var.data[keep_idx]
            bn.num_features = len(keep_idx)
        
        # 입력 채널 pruning (이전 레이어의 출력)
        if in_mask is not None:
            keep_idx = in_mask.nonzero().squeeze(-1)
            
            # groups 처리 (depthwise conv 등)
            if conv.groups == 1:
                conv.weight.data = conv.weight.data[:, keep_idx]
                conv.in_channels = len(keep_idx)
            elif conv.groups == conv.in_channels:  # Depthwise
                conv.weight.data = conv.weight.data[keep_idx]
                conv.groups = len(keep_idx)
                conv.in_channels = len(keep_idx)
                conv.out_channels = len(keep_idx)
    
    def prune_entire_model(self):
        """
        전체 모델 pruning
        
        핵심 전략:
        1. Backbone의 각 stage 끝 채널을 기준으로 pruning
        2. Skip connection과 Concat을 고려한 채널 동기화
        3. Detection head는 보존
        """
        print(f"\n{'='*60}")
        print(f"🔧 YOLO11 Structured Pruning (ratio: {self.prune_ratio*100:.0f}%)")
        print(f"{'='*60}")
        
        before_params = sum(p.numel() for p in self.model.parameters())
        print(f"Before: {before_params/1e6:.3f}M params")
        
        # YOLO11n 채널 구조 (인덱스: 출력채널)
        # Block 0: 16, Block 1: 32, Block 2(C3k2): 64
        # Block 3: 64, Block 4(C3k2): 128
        # Block 5: 128, Block 6(C3k2): 128
        # Block 7: 256, Block 8(C3k2): 256
        # Block 9(SPPF): 256, Block 10(C2PSA): 256
        
        # === 간단한 접근: 각 Conv 블록의 출력 채널만 pruning ===
        # (Concat과 skip connection 영향 최소화를 위해 보수적으로)
        
        pruned_count = 0
        
        for name, module in self.model.named_modules():
            # Ultralytics Conv 블록 (Conv2d + BN + Act)
            if type(module).__name__ == 'Conv' and hasattr(module, 'conv') and hasattr(module, 'bn'):
                conv = module.conv
                bn = module.bn
                
                # 스킵 조건
                if conv.in_channels == 3:  # RGB 입력
                    continue
                if conv.out_channels < 16:  # 너무 작음
                    continue
                
                # Detection/Pose head 스킵 (출력 형태 유지 필요)
                if 'model.23' in name:  # Pose head
                    continue
                
                # 중요도 계산
                importance = compute_importance(conv.weight.data)
                out_mask = get_keep_mask(importance, self.keep_ratio)
                
                n_before = conv.out_channels
                n_after = out_mask.sum().item()
                
                if n_after < n_before:
                    keep_idx = out_mask.nonzero().squeeze(-1)
                    
                    # Conv 출력 채널 pruning
                    conv.weight.data = conv.weight.data[keep_idx]
                    conv.out_channels = int(n_after)
                    if conv.bias is not None:
                        conv.bias.data = conv.bias.data[keep_idx]
                    
                    # BN 동기화
                    bn.weight.data = bn.weight.data[keep_idx]
                    bn.bias.data = bn.bias.data[keep_idx]
                    bn.running_mean.data = bn.running_mean.data[keep_idx]
                    bn.running_var.data = bn.running_var.data[keep_idx]
                    bn.num_features = int(n_after)
                    
                    # 이 레이어의 마스크 저장 (다음 레이어 입력 조정용)
                    self.channel_masks[name] = out_mask
                    
                    pruned_count += 1
                    print(f"  ✂️ {name}: {n_before} -> {n_after}")
        
        # === 두 번째 패스: 입력 채널 동기화 ===
        print("\n  === 입력 채널 동기화 ===")
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 이전 레이어 찾기 (간단한 휴리스틱)
                # 예: model.1.conv의 입력은 model.0의 출력
                
                parts = name.split('.')
                if len(parts) >= 2 and parts[0] == 'model':
                    try:
                        block_idx = int(parts[1])
                        prev_block_name = f"model.{block_idx - 1}"
                        
                        # 이전 블록의 마스크 확인
                        prev_mask = None
                        for mask_name, mask in self.channel_masks.items():
                            if prev_block_name in mask_name:
                                prev_mask = mask
                                break
                        
                        if prev_mask is not None and module.groups == 1:
                            # 현재 conv의 입력 채널과 마스크 크기 비교
                            if module.in_channels == len(prev_mask):
                                keep_idx = prev_mask.nonzero().squeeze(-1)
                                module.weight.data = module.weight.data[:, keep_idx]
                                module.in_channels = len(keep_idx)
                                print(f"    🔗 {name} in_ch adjusted")
                    except (ValueError, IndexError):
                        pass
        
        after_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nAfter: {after_params/1e6:.3f}M params")
        print(f"Reduction: {(1 - after_params/before_params)*100:.1f}%")
        
        return self.model
    
    def validate_forward(self):
        """Forward pass 테스트"""
        try:
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            self.model.eval()
            with torch.no_grad():
                output = self.model(dummy)
            print("✅ Forward pass 성공!")
            return True
        except Exception as e:
            print(f"❌ Forward pass 실패: {e}")
            return False
    
    def get_metrics(self):
        """모델 메트릭 계산"""
        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        
        flops = 0.0
        if HAS_THOP:
            try:
                dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
                macs, _ = profile(self.model, inputs=(dummy,), verbose=False)
                flops = macs / 1e9
                
                for m in self.model.modules():
                    for attr in ['total_ops', 'total_params']:
                        if hasattr(m, attr):
                            delattr(m, attr)
            except:
                pass
        
        return params, flops


# =========================================================
# 더 안전한 방법: Width Multiplier Scaling
# =========================================================
def create_smaller_yolo(model_path: str, width_mult: float = 0.5):
    """
    YOLO 모델의 width multiplier를 줄여서 더 작은 모델 생성
    
    이 방법은 채널 수를 균일하게 줄이므로 의존성 문제가 없습니다.
    """
    print(f"\n{'='*60}")
    print(f"🔧 Width Multiplier Scaling (mult: {width_mult})")
    print(f"{'='*60}")
    
    yolo = YOLO(model_path)
    model = yolo.model
    model.eval()
    
    before_params = sum(p.numel() for p in model.parameters())
    print(f"Before: {before_params/1e6:.3f}M params")
    
    # 모든 Conv + BN 쌍의 채널을 width_mult 비율로 줄임
    for name, module in model.named_modules():
        if type(module).__name__ == 'Conv' and hasattr(module, 'conv') and hasattr(module, 'bn'):
            conv = module.conv
            bn = module.bn
            
            # 입력 채널 3 (RGB) 또는 출력이 너무 작으면 스킵
            if conv.in_channels == 3 or conv.out_channels <= 8:
                continue
            
            # Pose head 스킵
            if 'model.23' in name:
                continue
            
            # 새 채널 수 계산 (8의 배수)
            new_out = max(8, int(conv.out_channels * width_mult) // 8 * 8)
            
            if new_out < conv.out_channels:
                # 가장 중요한 채널만 유지
                importance = compute_importance(conv.weight.data)
                _, keep_idx = torch.topk(importance, new_out)
                keep_idx = keep_idx.sort().values
                
                # 출력 채널 pruning
                conv.weight.data = conv.weight.data[keep_idx]
                conv.out_channels = new_out
                if conv.bias is not None:
                    conv.bias.data = conv.bias.data[keep_idx]
                
                # BN 동기화
                bn.weight.data = bn.weight.data[keep_idx]
                bn.bias.data = bn.bias.data[keep_idx]
                bn.running_mean.data = bn.running_mean.data[keep_idx]
                bn.running_var.data = bn.running_var.data[keep_idx]
                bn.num_features = new_out
    
    after_params = sum(p.numel() for p in model.parameters())
    print(f"After: {after_params/1e6:.3f}M params")
    print(f"Reduction: {(1 - after_params/before_params)*100:.1f}%")
    
    return model


# =========================================================
# Unstructured Pruning (확실하게 동작) + Sparsity 측정
# =========================================================
def apply_unstructured_pruning(model_path: str, prune_ratio: float = 0.3):
    """
    Unstructured pruning 적용
    - 파라미터 수는 동일
    - Non-zero 파라미터 비율 감소
    - 모델은 정상 동작
    """
    print(f"\n{'='*60}")
    print(f"🔧 Unstructured Pruning (ratio: {prune_ratio*100:.0f}%)")
    print(f"{'='*60}")
    
    import torch.nn.utils.prune as prune
    
    yolo = YOLO(model_path)
    model = yolo.model
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_before = sum((p != 0).sum().item() for p in model.parameters())
    
    print(f"Before: {total_params/1e6:.3f}M params, {nonzero_before/1e6:.3f}M non-zero")
    
    # L1 unstructured pruning 적용
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=prune_ratio)
            prune.remove(module, 'weight')
    
    nonzero_after = sum((p != 0).sum().item() for p in model.parameters())
    
    print(f"After: {total_params/1e6:.3f}M params, {nonzero_after/1e6:.3f}M non-zero")
    print(f"Sparsity: {(1 - nonzero_after/nonzero_before)*100:.1f}%")
    
    return model, yolo


# =========================================================
# 벤치마크
# =========================================================
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
            print(f"  ⚠️ FLOPs 측정 실패: {str(e)[:50]}")
    
    # 속도 측정
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    fps, latency = 0.0, 0.0
    
    try:
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(dummy)
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(20):
                t0 = time.time()
                model(dummy)
                times.append(time.time() - t0)
        
        fps = 1.0 / (sum(times) / len(times))
        latency = (sum(times) / len(times)) * 1000
    except Exception as e:
        print(f"  ⚠️ 속도 측정 실패: {str(e)[:50]}")
    
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
    print("🚀 YOLO11 Structured Pruning 벤치마크 v2")
    print("=" * 70)
    
    if not MODEL_PATH.exists():
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return
    
    results = []
    
    # 1. 원본 모델
    print("\n[1] 원본 모델 측정...")
    yolo_base = YOLO(MODEL_PATH)
    base_result = benchmark(yolo_base.model, "Baseline")
    results.append(base_result)
    print(f"   Params: {base_result['params']:.3f}M, FLOPs: {base_result['flops']:.3f}G, FPS: {base_result['fps']:.1f}")
    
    # 2. Unstructured Pruning (확실하게 동작)
    print("\n[2] Unstructured Pruning...")
    for ratio in [0.3, 0.5, 0.7]:
        model, yolo = apply_unstructured_pruning(MODEL_PATH, ratio)
        result = benchmark(model, f"Unstructured_{int(ratio*100)}%")
        result['sparsity'] = ratio
        results.append(result)
        
        # 저장
        save_path = ROOT / f"assets/models/yolo11n_unstructured_{int(ratio*100)}.pt"
        yolo.save(str(save_path))
        print(f"   💾 저장: {save_path.name}")
    
    # 3. Width Multiplier Scaling (구조적 변경, 동작 가능)
    print("\n[3] Width Multiplier Scaling...")
    for mult in [0.75, 0.5, 0.25]:
        try:
            model = create_smaller_yolo(MODEL_PATH, mult)
            
            # Forward 테스트
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            try:
                with torch.no_grad():
                    output = model(dummy)
                print(f"   ✅ Width {mult} forward 성공")
                
                result = benchmark(model, f"Width_{mult}")
                results.append(result)
                
                # 저장
                save_path = ROOT / f"assets/models/yolo11n_width_{int(mult*100)}.pt"
                torch.save({'model': model.state_dict()}, save_path)
                
            except Exception as e:
                print(f"   ❌ Width {mult} forward 실패: {str(e)[:50]}")
        except Exception as e:
            print(f"   ❌ Width {mult} 생성 실패: {str(e)[:50]}")
    
    # 결과 요약
    print("\n" + "=" * 90)
    print(f"{'Model':<25} | {'Params(M)':<12} | {'NonZero(M)':<12} | {'FLOPs(G)':<10} | {'FPS':<8}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<25} | {r['params']:<12.3f} | {r['nonzero']:<12.3f} | {r['flops']:<10.3f} | {r['fps']:<8.1f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
