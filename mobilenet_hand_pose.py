"""
🖐️ MobileNet Hand Pose Estimation - 학습 파이프라인

특징:
- MobileNetV2/V3 backbone (pruning/quantization 100% 지원)
- 21개 hand keypoint (YOLO11-pose와 동일)
- FreiHAND 데이터셋 사용
- Pruning + Quantization 파이프라인 포함

구조:
1. Hand Detection: 기존 YOLO 또는 별도 detector
2. Hand Keypoint: 이 MobileNet 모델

학습 데이터:
- FreiHAND: https://lmb.informatik.uni-freiburg.de/projects/freihand/
- 또는 직접 수집한 hand keypoint 데이터
"""

import os
import sys
import json
import time
import copy
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    from torchvision.models import (
        mobilenet_v2, MobileNet_V2_Weights,
        mobilenet_v3_small, MobileNet_V3_Small_Weights,
        mobilenet_v3_large, MobileNet_V3_Large_Weights,
    )
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("⚠️ torchvision 필요: pip install torchvision")

try:
    import torch_pruning as tp
    HAS_PRUNING = True
except ImportError:
    HAS_PRUNING = False
    print("⚠️ torch_pruning 필요: pip install torch-pruning")

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "assets/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Hand Pose 모델 정의
# =========================================================

class HandPoseNet(nn.Module):
    """
    MobileNet Backbone + Hand Keypoint Head
    
    Args:
        backbone: 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'
        num_keypoints: 손 관절 수 (기본 21)
        pretrained_backbone: ImageNet pretrained 사용 여부
    
    입력: (B, 3, 224, 224) 또는 (B, 3, 256, 256)
    출력: (B, 21, 2) - 21개 keypoint의 normalized (x, y) 좌표 [0, 1]
    """
    def __init__(
        self, 
        backbone: str = 'mobilenet_v2',
        num_keypoints: int = 21,
        pretrained_backbone: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.backbone_name = backbone
        
        # Backbone 선택
        if backbone == 'mobilenet_v2':
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            base_model = mobilenet_v2(weights=weights)
            self.backbone = base_model.features
            self.feature_dim = 1280
            
        elif backbone == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            base_model = mobilenet_v3_small(weights=weights)
            self.backbone = base_model.features
            self.feature_dim = 576
            
        elif backbone == 'mobilenet_v3_large':
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            base_model = mobilenet_v3_large(weights=weights)
            self.backbone = base_model.features
            self.feature_dim = 960
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Keypoint Regression Head
        self.keypoint_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_keypoints * 2),
            nn.Sigmoid(),  # [0, 1] normalized output
        )
        
        # 마지막 레이어 초기화
        self._init_head()
    
    def _init_head(self):
        for m in self.keypoint_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        keypoints = self.keypoint_head(features)
        keypoints = keypoints.view(-1, self.num_keypoints, 2)
        return keypoints
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_flops(self, input_size=224):
        if not HAS_THOP:
            return 0
        dummy = torch.randn(1, 3, input_size, input_size)
        macs, _ = profile(self, inputs=(dummy,), verbose=False)
        # 임시 속성 제거
        for m in self.modules():
            for attr in ['total_ops', 'total_params']:
                if hasattr(m, attr):
                    delattr(m, attr)
        return macs


# =========================================================
# 데이터셋
# =========================================================

class HandKeypointDataset(Dataset):
    """
    Hand Keypoint 데이터셋
    
    기대하는 데이터 형식:
    - images/: 손 이미지들 (cropped hand region)
    - annotations.json: [{"image": "xxx.jpg", "keypoints": [[x1,y1], [x2,y2], ...]}]
    
    keypoints는 이미지 기준 normalized [0, 1] 좌표
    """
    def __init__(
        self, 
        data_dir: str,
        img_size: int = 224,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        
        # 어노테이션 로드
        ann_file = self.data_dir / "annotations.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            # 어노테이션 없으면 이미지 파일 목록만
            self.annotations = []
            if (self.data_dir / "images").exists():
                for img_path in (self.data_dir / "images").glob("*.jpg"):
                    self.annotations.append({
                        "image": img_path.name,
                        "keypoints": [[0.5, 0.5]] * 21  # 더미
                    })
        
        print(f"[Dataset] {len(self.annotations)} samples loaded from {data_dir}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 이미지 로드
        img_path = self.data_dir / "images" / ann["image"]
        
        if HAS_CV2 and img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
        else:
            # 더미 이미지
            img = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Keypoints
        keypoints = np.array(ann["keypoints"], dtype=np.float32)
        
        # Augmentation
        if self.augment:
            img, keypoints = self._augment(img, keypoints)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        
        return torch.FloatTensor(img), torch.FloatTensor(keypoints)
    
    def _augment(self, img, keypoints):
        # 간단한 augmentation
        # 1. 좌우 반전 (50% 확률)
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            keypoints[:, 0] = 1.0 - keypoints[:, 0]
        
        # 2. 밝기 변화
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        return img, keypoints


class DummyDataset(Dataset):
    """학습 테스트용 더미 데이터셋"""
    def __init__(self, num_samples=1000, img_size=224, num_keypoints=21):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_keypoints = num_keypoints
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 랜덤 이미지
        img = torch.randn(3, self.img_size, self.img_size)
        # 랜덤 keypoints [0, 1]
        keypoints = torch.rand(self.num_keypoints, 2)
        return img, keypoints


# =========================================================
# 학습 함수
# =========================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, keypoints) in enumerate(dataloader):
        images = images.to(device)
        keypoints = keypoints.to(device)
        
        optimizer.zero_grad()
        
        pred = model(images)
        loss = criterion(pred, keypoints)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, keypoints in dataloader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            pred = model(images)
            loss = criterion(pred, keypoints)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cpu',
    save_path: Optional[str] = None,
):
    """
    모델 학습
    
    Args:
        model: HandPoseNet 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더 (optional)
        epochs: 학습 에폭 수
        lr: 학습률
        device: 'cpu' 또는 'cuda'
        save_path: 모델 저장 경로
    """
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    print(f"\n[Training] {epochs} epochs, lr={lr}, device={device}")
    print(f"[Model] {model.get_num_params()/1e6:.3f}M params")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    torch.save(model.state_dict(), save_path)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
    
    if save_path and not val_loader:
        torch.save(model.state_dict(), save_path)
    
    print(f"\n[Training Complete] Best Val Loss: {best_val_loss:.6f}")
    
    return model


# =========================================================
# Pruning
# =========================================================

def prune_model(model, prune_ratio=0.5, input_size=224):
    """
    Structured Pruning 적용
    
    Args:
        model: HandPoseNet 모델
        prune_ratio: 제거할 비율 (0.5 = 50%)
        input_size: 입력 이미지 크기
    
    Returns:
        pruned model
    """
    if not HAS_PRUNING:
        print("⚠️ torch_pruning 없음, pruning 스킵")
        return model
    
    model = copy.deepcopy(model)
    model.eval()
    
    example_inputs = torch.randn(1, 3, input_size, input_size)
    
    # ignored: 마지막 Linear (출력 크기 고정)
    ignored_layers = []
    for m in model.keypoint_head:
        if isinstance(m, nn.Linear):
            ignored_layers = [m]  # 마지막 것만
    
    before_params = model.get_num_params()
    
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
    
    after_params = model.get_num_params()
    
    print(f"[Pruning] {prune_ratio*100:.0f}%: {before_params/1e6:.3f}M -> {after_params/1e6:.3f}M ({(1-after_params/before_params)*100:.1f}%↓)")
    
    return model


# =========================================================
# Quantization
# =========================================================

def quantize_model_dynamic(model):
    """Dynamic Quantization (CPU용)"""
    try:
        quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        return quantized
    except Exception as e:
        print(f"   ⚠️ Dynamic Quantization 실패 (Mac 미지원): {str(e)[:50]}")
        print("   → Jetson/Linux에서 TensorRT INT8 사용 권장")
        return model


def quantize_model_static(model, calibration_loader, device='cpu'):
    """Static Quantization (더 효과적)"""
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    model_prepared = torch.quantization.prepare(model)
    
    # Calibration
    with torch.no_grad():
        for images, _ in calibration_loader:
            model_prepared(images.to(device))
    
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized


def export_onnx(model, save_path, input_size=224):
    """ONNX 내보내기 (TensorRT 변환용)"""
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    
    torch.onnx.export(
        model,
        dummy,
        save_path,
        input_names=['input'],
        output_names=['keypoints'],
        dynamic_axes={'input': {0: 'batch'}, 'keypoints': {0: 'batch'}},
        opset_version=11,
    )
    print(f"[Export] ONNX saved: {save_path}")


# =========================================================
# 메인 함수
# =========================================================

def main():
    print("=" * 70)
    print("🖐️ MobileNet Hand Pose - 학습 및 최적화 파이프라인")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # =========================================================
    # 1. 모델 생성
    # =========================================================
    print("\n[1] 모델 생성...")
    
    model = HandPoseNet(
        backbone='mobilenet_v2',
        num_keypoints=21,
        pretrained_backbone=True,
        dropout=0.3,
    )
    
    params = model.get_num_params() / 1e6
    flops = model.get_flops(224) / 1e9
    print(f"   Backbone: MobileNetV2")
    print(f"   Params: {params:.3f}M")
    print(f"   FLOPs: {flops:.3f}G")
    
    # =========================================================
    # 2. 데이터셋 준비 (더미 데이터로 테스트)
    # =========================================================
    print("\n[2] 데이터셋 준비...")
    
    # 더미 데이터로 테스트 (실제로는 FreiHAND 등 사용)
    train_dataset = DummyDataset(num_samples=200, img_size=224)
    val_dataset = DummyDataset(num_samples=50, img_size=224)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    
    # =========================================================
    # 3. 학습 (짧은 테스트)
    # =========================================================
    print("\n[3] 학습 (데모: 5 epochs)...")
    
    save_path = MODELS_DIR / "handpose_mobilenetv2.pt"
    
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=5,  # 데모용 짧은 학습
        lr=1e-3,
        device=device,
        save_path=str(save_path),
    )
    
    # =========================================================
    # 4. Pruning
    # =========================================================
    print("\n[4] Pruning...")
    
    results = [{"name": "Original", "params": model.get_num_params()/1e6, "flops": model.get_flops(224)/1e9}]
    
    for ratio in [0.3, 0.5, 0.7]:
        pruned = prune_model(model, prune_ratio=ratio)
        
        # 저장
        save_pruned = MODELS_DIR / f"handpose_mobilenetv2_pruned_{int(ratio*100)}.pt"
        torch.save(pruned.state_dict(), save_pruned)
        
        results.append({
            "name": f"Pruned_{int(ratio*100)}%",
            "params": pruned.get_num_params()/1e6,
            "flops": pruned.get_flops(224)/1e9,
        })
    
    # =========================================================
    # 5. Quantization (CPU)
    # =========================================================
    print("\n[5] Quantization (Dynamic)...")
    
    # Dynamic Quantization
    model_quantized = quantize_model_dynamic(copy.deepcopy(model))
    
    save_quantized = MODELS_DIR / "handpose_mobilenetv2_int8.pt"
    torch.save(model_quantized.state_dict(), save_quantized)
    print(f"   Saved: {save_quantized.name}")
    
    # =========================================================
    # 6. ONNX Export
    # =========================================================
    print("\n[6] ONNX Export...")
    
    onnx_path = MODELS_DIR / "handpose_mobilenetv2.onnx"
    export_onnx(model, str(onnx_path), input_size=224)
    
    # Pruned 모델도 ONNX
    pruned_50 = prune_model(model, prune_ratio=0.5)
    onnx_pruned_path = MODELS_DIR / "handpose_mobilenetv2_pruned_50.onnx"
    export_onnx(pruned_50, str(onnx_pruned_path), input_size=224)
    
    # =========================================================
    # 결과 요약
    # =========================================================
    print("\n" + "=" * 70)
    print("📊 결과 요약")
    print("=" * 70)
    print(f"{'Model':<25} | {'Params(M)':<12} | {'FLOPs(G)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} | {r['params']:<12.3f} | {r['flops']:<12.3f}")
    print("=" * 70)
    
    print(f"""
📁 저장된 파일:
   • {save_path.name} - 원본 모델
   • handpose_mobilenetv2_pruned_*.pt - Pruned 모델들
   • handpose_mobilenetv2_int8.pt - Quantized 모델
   • handpose_mobilenetv2.onnx - ONNX (TensorRT 변환용)

🚀 다음 단계:
   1. FreiHAND 데이터셋으로 실제 학습
   2. Jetson에서 TensorRT로 ONNX 변환
   3. Hand Detection (YOLO) + Hand Keypoint (MobileNet) 파이프라인 구축
""")


if __name__ == "__main__":
    main()
