"""
🚀 Hand Pose Estimation 대안 모델 + Pruning/Quantization 벤치마크

YOLO11-pose는 torch_pruning과 호환되지 않으므로,
pruning이 가능한 대안 모델들을 테스트합니다:

1. MobileNetV2 기반 Hand Keypoint 모델
2. MediaPipe Hands (이미 최적화됨)
3. 간단한 CNN 기반 Hand Pose 모델

핵심: MobileNet/ResNet은 torch_pruning 100% 지원!
"""

import os
import sys
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import torch_pruning as tp

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

ROOT = Path(__file__).resolve().parent
IMG_SIZE = 256  # Hand pose 모델은 보통 더 작은 이미지 사용
NUM_KEYPOINTS = 21  # 손 21개 관절


# =========================================================
# Hand Pose 모델 정의 (MobileNetV2 기반)
# =========================================================

class HandPoseNet(nn.Module):
    """
    MobileNetV2 Backbone + Hand Keypoint Head
    
    입력: (B, 3, 256, 256)
    출력: (B, 21, 2) - 21개 keypoint의 x, y 좌표
    """
    def __init__(self, num_keypoints=21, pretrained_backbone=True):
        super().__init__()
        
        # MobileNetV2 backbone
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        
        if pretrained_backbone:
            backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            backbone = mobilenet_v2(weights=None)
        
        # features만 사용 (classifier 제거)
        self.backbone = backbone.features  # (B, 1280, 8, 8) for 256x256 input
        
        # Keypoint head
        self.keypoint_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, 1280, 1, 1)
            nn.Flatten(),              # (B, 1280)
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_keypoints * 2),  # (B, 42) = 21 * 2
        )
        
        self.num_keypoints = num_keypoints
    
    def forward(self, x):
        features = self.backbone(x)  # (B, 1280, 8, 8)
        keypoints = self.keypoint_head(features)  # (B, 42)
        keypoints = keypoints.view(-1, self.num_keypoints, 2)  # (B, 21, 2)
        return keypoints


class HandPoseNetLite(nn.Module):
    """
    경량 Hand Pose 모델 (MobileNetV3-Small 기반)
    """
    def __init__(self, num_keypoints=21, pretrained_backbone=True):
        super().__init__()
        
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        
        if pretrained_backbone:
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            backbone = mobilenet_v3_small(weights=None)
        
        self.backbone = backbone.features  # (B, 576, 8, 8)
        
        self.keypoint_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Linear(128, num_keypoints * 2),
        )
        
        self.num_keypoints = num_keypoints
    
    def forward(self, x):
        features = self.backbone(x)
        keypoints = self.keypoint_head(features)
        keypoints = keypoints.view(-1, self.num_keypoints, 2)
        return keypoints


# =========================================================
# Pruning 함수
# =========================================================

def prune_model(model, prune_ratio, ignored_layers=None):
    """
    torch_pruning으로 structured pruning
    """
    model = copy.deepcopy(model)
    example_inputs = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    
    before_params = sum(p.numel() for p in model.parameters())
    
    # ignored_layers 기본값: 마지막 Linear 레이어 (출력 크기 유지)
    if ignored_layers is None:
        ignored_layers = []
        for m in model.modules():
            if isinstance(m, nn.Linear):
                # 마지막 Linear만 무시
                ignored_layers = [m]
    
    try:
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(p=2),
            iterative_steps=1,
            pruning_ratio=prune_ratio,
            ignored_layers=ignored_layers,
            round_to=8,
        )
        
        pruner.step()
        
        after_params = sum(p.numel() for p in model.parameters())
        
        # Forward 테스트
        with torch.no_grad():
            output = model(example_inputs)
        
        return model, before_params, after_params, True
        
    except Exception as e:
        print(f"   Pruning 실패: {e}")
        return model, before_params, before_params, False


# =========================================================
# 벤치마크
# =========================================================

def benchmark(model, name, device='cpu'):
    model = model.to(device).eval()
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    flops = 0.0
    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    if HAS_THOP:
        try:
            macs, _ = profile(model, inputs=(example,), verbose=False)
            flops = macs / 1e9
            for m in model.modules():
                for attr in ['total_ops', 'total_params']:
                    if hasattr(m, attr):
                        delattr(m, attr)
        except:
            pass
    
    # 속도 측정
    fps = 0.0
    try:
        with torch.no_grad():
            for _ in range(10):
                model(example)
        
        times = []
        with torch.no_grad():
            for _ in range(50):
                t0 = time.time()
                model(example)
                times.append(time.time() - t0)
        
        fps = 1.0 / (sum(times) / len(times))
    except:
        pass
    
    return {
        'name': name,
        'params': params,
        'flops': flops,
        'fps': fps,
    }


def main():
    print("=" * 80)
    print("🚀 Hand Pose 대안 모델 + Pruning 벤치마크")
    print("=" * 80)
    print("YOLO11-pose는 torch_pruning 호환 X → MobileNet 기반 모델 테스트")
    print("=" * 80)
    
    results = []
    
    # =========================================================
    # 1️⃣ MobileNetV2 기반 Hand Pose
    # =========================================================
    print("\n[1] HandPoseNet (MobileNetV2 Backbone)...")
    
    model = HandPoseNet(num_keypoints=21, pretrained_backbone=True)
    result = benchmark(model, "HandPoseNet_MobileNetV2")
    results.append(result)
    print(f"   Params: {result['params']:.3f}M, FLOPs: {result['flops']:.3f}G, FPS: {result['fps']:.1f}")
    
    # Pruning 테스트
    for ratio in [0.3, 0.5, 0.7]:
        pruned, before, after, success = prune_model(model, ratio)
        if success:
            result = benchmark(pruned, f"HandPoseNet_Pruned_{int(ratio*100)}%")
            results.append(result)
            print(f"   Pruned {int(ratio*100)}%: {result['params']:.3f}M, FLOPs: {result['flops']:.3f}G, FPS: {result['fps']:.1f}")
            
            # 저장
            save_path = ROOT / f"assets/models/handpose_mobilenetv2_pruned_{int(ratio*100)}.pt"
            torch.save(pruned.state_dict(), save_path)
    
    # =========================================================
    # 2️⃣ MobileNetV3-Small 기반 (더 경량)
    # =========================================================
    print("\n[2] HandPoseNetLite (MobileNetV3-Small Backbone)...")
    
    model_lite = HandPoseNetLite(num_keypoints=21, pretrained_backbone=True)
    result = benchmark(model_lite, "HandPoseNetLite_MobileNetV3")
    results.append(result)
    print(f"   Params: {result['params']:.3f}M, FLOPs: {result['flops']:.3f}G, FPS: {result['fps']:.1f}")
    
    # Pruning 테스트
    for ratio in [0.3, 0.5, 0.7]:
        pruned, before, after, success = prune_model(model_lite, ratio)
        if success:
            result = benchmark(pruned, f"HandPoseNetLite_Pruned_{int(ratio*100)}%")
            results.append(result)
            print(f"   Pruned {int(ratio*100)}%: {result['params']:.3f}M, FLOPs: {result['flops']:.3f}G, FPS: {result['fps']:.1f}")
    
    # =========================================================
    # 3️⃣ YOLO11n-pose (비교용)
    # =========================================================
    print("\n[3] YOLO11n-pose (비교용)...")
    try:
        from ultralytics import YOLO
        yolo_path = ROOT / "assets/models/yolo11n_hand_pose.pt"
        if yolo_path.exists():
            yolo = YOLO(yolo_path)
            
            yolo_params = sum(p.numel() for p in yolo.model.parameters()) / 1e6
            
            # YOLO 속도 측정
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(10):
                yolo.predict(dummy, imgsz=640, verbose=False, device='cpu')
            
            times = []
            for _ in range(30):
                t0 = time.time()
                yolo.predict(dummy, imgsz=640, verbose=False, device='cpu')
                times.append(time.time() - t0)
            
            yolo_fps = 1.0 / (sum(times) / len(times))
            
            results.append({
                'name': 'YOLO11n-pose',
                'params': yolo_params,
                'flops': 3.96,  # 이전 측정값
                'fps': yolo_fps,
            })
            print(f"   Params: {yolo_params:.3f}M, FLOPs: 3.96G, FPS: {yolo_fps:.1f}")
    except Exception as e:
        print(f"   YOLO 로드 실패: {e}")
    
    # =========================================================
    # 결과 비교
    # =========================================================
    print("\n" + "=" * 90)
    print(f"{'Model':<40} | {'Params(M)':<12} | {'FLOPs(G)':<12} | {'FPS':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<40} | {r['params']:<12.3f} | {r['flops']:<12.3f} | {r['fps']:<10.1f}")
    print("=" * 90)
    
    print("\n" + "=" * 60)
    print("📊 분석")
    print("=" * 60)
    print("""
┌──────────────────────────────────────────────────────────────┐
│ ✅ MobileNet 기반 모델의 장점:                               │
│    • torch_pruning 100% 지원                                 │
│    • Structured Pruning으로 실제 파라미터/FLOPs 감소          │
│    • Quantization (TensorRT/TFLite) 완벽 지원                 │
│    • 속도가 더 빠름 (256x256 입력)                            │
├──────────────────────────────────────────────────────────────┤
│ ⚠️ 고려사항:                                                 │
│    • Hand detection 필요 (YOLO는 detection+keypoint 통합)    │
│    • 학습 데이터 필요 (현재는 ImageNet pretrained backbone)   │
│    • YOLO11-pose 대비 정확도 확인 필요                        │
├──────────────────────────────────────────────────────────────┤
│ 💡 권장 파이프라인:                                          │
│    • Hand Detection: YOLOv8n-detect (작은 모델)               │
│    • Hand Keypoint: MobileNet 기반 모델 (pruning 가능)        │
│    • 또는: MediaPipe Hands (이미 최적화됨)                    │
└──────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
