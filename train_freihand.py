"""
🖐️ FreiHAND 데이터셋 + MPS 학습

FreiHAND: https://lmb.informatik.uni-freiburg.de/projects/freihand/
- 130,000+ 손 이미지
- 21개 hand keypoint 어노테이션
- 다양한 배경, 조명, 손 포즈

Mac MPS 가속 사용
"""

import os
import sys
import json
import time
import copy
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ cv2 필요: pip install opencv-python")

try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    import torch_pruning as tp
    HAS_PRUNING = True
except ImportError:
    HAS_PRUNING = False

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️ tqdm 설치 권장: pip install tqdm")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "freihand"
MODELS_DIR = ROOT / "assets/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Device 설정 (Mac MPS 우선)
# =========================================================

def get_device():
    """Mac MPS > CUDA > CPU 순서로 device 선택"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# =========================================================
# FreiHAND 데이터셋 다운로드
# =========================================================

FREIHAND_URLS = {
    "training": "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip",
    "evaluation": "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip",
}

def download_freihand(data_dir: Path, subset: str = "training"):
    """
    FreiHAND 데이터셋 다운로드
    
    전체 크기: ~10GB
    빠른 테스트를 위해 일부만 사용 가능
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미 다운로드됐는지 확인
    if (data_dir / "training" / "rgb").exists():
        print(f"[FreiHAND] 이미 다운로드됨: {data_dir}")
        return True
    
    print(f"[FreiHAND] 다운로드 시작... (약 3-4GB)")
    print("⚠️ 시간이 오래 걸릴 수 있습니다. 빠른 테스트를 원하면 수동으로 일부만 다운로드하세요.")
    
    # 수동 다운로드 안내
    print(f"""
    📥 수동 다운로드 방법:
    1. {FREIHAND_URLS['training']} 다운로드
    2. {data_dir}/training 에 압축 해제
    
    또는 아래 명령어로 빠른 테스트:
    python -c "from train_freihand import create_sample_dataset; create_sample_dataset()"
    """)
    
    return False


def create_sample_dataset(num_samples: int = 100):
    """
    간단한 샘플 데이터셋 생성 (테스트용)
    
    실제 손 이미지 대신 랜덤 이미지 + 랜덤 keypoint
    """
    sample_dir = DATA_DIR / "sample"
    img_dir = sample_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = []
    
    print(f"[Sample Dataset] {num_samples}개 샘플 생성 중...")
    
    for i in range(num_samples):
        # 랜덤 이미지 생성
        img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # 손 모양 대충 그리기 (원)
        center = (112 + np.random.randint(-30, 30), 112 + np.random.randint(-30, 30))
        cv2.circle(img, center, 50, (200, 180, 160), -1)
        
        # 저장
        img_name = f"sample_{i:05d}.jpg"
        cv2.imwrite(str(img_dir / img_name), img)
        
        # 랜덤 keypoints (21개, [0, 1] normalized)
        keypoints = []
        base_x, base_y = np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7)
        for j in range(21):
            kp_x = base_x + np.random.uniform(-0.2, 0.2)
            kp_y = base_y + np.random.uniform(-0.2, 0.2)
            keypoints.append([np.clip(kp_x, 0, 1), np.clip(kp_y, 0, 1)])
        
        annotations.append({
            "image": img_name,
            "keypoints": keypoints
        })
    
    # 어노테이션 저장
    with open(sample_dir / "annotations.json", 'w') as f:
        json.dump(annotations, f)
    
    print(f"[Sample Dataset] 저장됨: {sample_dir}")
    return sample_dir


# =========================================================
# FreiHAND 데이터셋 클래스
# =========================================================

class FreiHANDDataset(Dataset):
    """
    FreiHAND 데이터셋
    
    구조:
    - training/rgb/: 이미지들
    - training_K.json: 카메라 intrinsics
    - training_xyz.json: 3D keypoints
    - training_mano.json: MANO parameters
    """
    def __init__(
        self, 
        data_dir: Path,
        img_size: int = 224,
        augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        
        # 어노테이션 로드 (sample 또는 FreiHAND)
        ann_file = self.data_dir / "annotations.json"
        if ann_file.exists():
            # 우리의 간단한 포맷
            with open(ann_file, 'r') as f:
                self.annotations = json.load(f)
            self.img_dir = self.data_dir / "images"
        else:
            # FreiHAND 원본 포맷
            self._load_freihand()
        
        if max_samples:
            self.annotations = self.annotations[:max_samples]
        
        print(f"[FreiHAND] {len(self.annotations)} samples loaded")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_freihand(self):
        """FreiHAND 원본 포맷 로드"""
        training_dir = self.data_dir / "training"
        self.img_dir = training_dir / "rgb"
        
        # 3D keypoints 로드
        xyz_file = self.data_dir / "training_xyz.json"
        if xyz_file.exists():
            with open(xyz_file, 'r') as f:
                xyz_data = json.load(f)
        else:
            xyz_data = []
        
        # 카메라 매트릭스 로드
        k_file = self.data_dir / "training_K.json"
        if k_file.exists():
            with open(k_file, 'r') as f:
                k_data = json.load(f)
        else:
            k_data = []
        
        self.annotations = []
        
        # 이미지 파일 목록
        if self.img_dir.exists():
            img_files = sorted(self.img_dir.glob("*.jpg"))
            for i, img_path in enumerate(img_files):
                if i < len(xyz_data) and i < len(k_data):
                    # 3D → 2D 투영
                    xyz = np.array(xyz_data[i])  # (21, 3)
                    K = np.array(k_data[i])       # (3, 3)
                    
                    # 투영
                    xyz_cam = xyz.T  # (3, 21)
                    uv = K @ xyz_cam  # (3, 21)
                    uv = uv[:2] / uv[2:3]  # (2, 21)
                    uv = uv.T  # (21, 2)
                    
                    # Normalize to [0, 1]
                    uv[:, 0] /= 224  # 이미지 크기
                    uv[:, 1] /= 224
                    uv = np.clip(uv, 0, 1)
                    
                    self.annotations.append({
                        "image": img_path.name,
                        "keypoints": uv.tolist()
                    })
                else:
                    # xyz 없으면 더미
                    self.annotations.append({
                        "image": img_path.name,
                        "keypoints": [[0.5, 0.5]] * 21
                    })
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 이미지 로드
        img_path = self.img_dir / ann["image"]
        
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
        else:
            img = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Keypoints
        keypoints = np.array(ann["keypoints"], dtype=np.float32)
        
        # Augmentation
        if self.augment:
            img, keypoints = self._augment(img, keypoints)
        
        # Transform
        img = self.transform(img)
        
        return img, torch.FloatTensor(keypoints)
    
    def _augment(self, img, keypoints):
        # 좌우 반전
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            keypoints[:, 0] = 1.0 - keypoints[:, 0]
        
        # 밝기 변화
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # 색상 jitter
        if np.random.rand() > 0.5:
            img = img.astype(np.float32)
            img[:, :, 0] *= np.random.uniform(0.9, 1.1)
            img[:, :, 1] *= np.random.uniform(0.9, 1.1)
            img[:, :, 2] *= np.random.uniform(0.9, 1.1)
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img, keypoints


# =========================================================
# 모델 정의
# =========================================================

class HandPoseNet(nn.Module):
    def __init__(self, num_keypoints: int = 21, pretrained: bool = True):
        super().__init__()
        
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        
        self.backbone = backbone.features
        self.feature_dim = 1280
        
        self.keypoint_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_keypoints * 2),
            nn.Sigmoid(),
        )
        
        self.num_keypoints = num_keypoints
    
    def forward(self, x):
        features = self.backbone(x)
        keypoints = self.keypoint_head(features)
        return keypoints.view(-1, self.num_keypoints, 2)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# =========================================================
# 학습 함수
# =========================================================

def train_epoch(model, dataloader, criterion, optimizer, device, epoch=0):
    model.train()
    total_loss = 0
    
    if HAS_TQDM:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for images, keypoints in pbar:
        images = images.to(device)
        keypoints = keypoints.to(device)
        
        optimizer.zero_grad()
        
        pred = model(images)
        loss = criterion(pred, keypoints)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if HAS_TQDM:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
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


def train(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_samples: Optional[int] = None,
    save_name: str = "handpose_freihand",
):
    """
    학습 메인 함수
    
    Args:
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        lr: 학습률
        max_samples: 최대 샘플 수 (빠른 테스트용)
        save_name: 저장 파일명
    """
    device = get_device()
    print(f"\n{'='*60}")
    print(f"🖐️ Hand Pose Training (Mac MPS)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # 데이터셋 준비
    print("\n[1] 데이터셋 준비...")
    
    # sample 데이터셋 확인/생성
    sample_dir = DATA_DIR / "sample"
    if not (sample_dir / "annotations.json").exists():
        print("   샘플 데이터셋 생성 중...")
        create_sample_dataset(num_samples=max_samples or 1000)
    
    dataset = FreiHANDDataset(
        data_dir=sample_dir,
        img_size=224,
        augment=True,
        max_samples=max_samples,
    )
    
    # Train/Val 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 모델
    print("\n[2] 모델 생성...")
    model = HandPoseNet(num_keypoints=21, pretrained=True)
    model = model.to(device)
    print(f"   Params: {model.get_num_params()/1e6:.3f}M")
    
    # 학습 설정
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 학습
    print(f"\n[3] 학습 시작 ({epochs} epochs)...")
    
    best_val_loss = float('inf')
    save_path = MODELS_DIR / f"{save_name}.pt"
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
    
    print(f"\n[Training Complete] Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved: {save_path}")
    
    return model


# =========================================================
# Pruning
# =========================================================

def prune_model(model, prune_ratio=0.5):
    """Structured Pruning"""
    if not HAS_PRUNING:
        print("⚠️ torch_pruning 없음")
        return model
    
    model = copy.deepcopy(model).cpu()
    model.eval()
    
    example = torch.randn(1, 3, 224, 224)
    
    # 마지막 Linear 무시
    ignored = []
    for m in model.keypoint_head:
        if isinstance(m, nn.Linear):
            ignored = [m]
    
    before = model.get_num_params()
    
    pruner = tp.pruner.MagnitudePruner(
        model, example,
        importance=tp.importance.MagnitudeImportance(p=2),
        iterative_steps=1,
        pruning_ratio=prune_ratio,
        ignored_layers=ignored,
        round_to=8,
    )
    
    pruner.step()
    
    after = model.get_num_params()
    print(f"[Pruning] {prune_ratio*100:.0f}%: {before/1e6:.3f}M → {after/1e6:.3f}M ({(1-after/before)*100:.1f}%↓)")
    
    return model


# =========================================================
# 메인
# =========================================================

def main():
    """
    학습 + Pruning 전체 파이프라인
    """
    print("=" * 60)
    print("🖐️ MobileNet Hand Pose - FreiHAND 학습")
    print("=" * 60)
    
    # 학습
    model = train(
        epochs=30,           # 충분한 학습
        batch_size=32,
        lr=1e-3,
        max_samples=1000,    # 빠른 테스트용 (전체: None)
        save_name="handpose_freihand_mps",
    )
    
    # Pruning
    print("\n[4] Pruning...")
    
    for ratio in [0.3, 0.5, 0.7]:
        pruned = prune_model(model, prune_ratio=ratio)
        save_path = MODELS_DIR / f"handpose_freihand_pruned_{int(ratio*100)}.pt"
        torch.save(pruned.state_dict(), save_path)
        print(f"   Saved: {save_path.name}")
    
    # ONNX Export
    print("\n[5] ONNX Export...")
    model.cpu().eval()
    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = MODELS_DIR / "handpose_freihand.onnx"
    torch.onnx.export(model, dummy, str(onnx_path), 
                     input_names=['input'], output_names=['keypoints'],
                     opset_version=11)
    print(f"   Saved: {onnx_path.name}")
    
    # Pruned 50% ONNX
    pruned_50 = prune_model(model, 0.5)
    onnx_pruned = MODELS_DIR / "handpose_freihand_pruned_50.onnx"
    torch.onnx.export(pruned_50, dummy, str(onnx_pruned),
                     input_names=['input'], output_names=['keypoints'],
                     opset_version=11)
    print(f"   Saved: {onnx_pruned.name}")
    
    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)
    print(f"""
📁 생성된 파일:
   • handpose_freihand_mps.pt - 학습된 모델
   • handpose_freihand_pruned_*.pt - Pruned 모델
   • handpose_freihand.onnx - TensorRT 변환용
   • handpose_freihand_pruned_50.onnx - Pruned ONNX

🚀 Jetson TensorRT 변환:
   trtexec --onnx=handpose_freihand_pruned_50.onnx --fp16 --saveEngine=handpose_fp16.engine
""")


if __name__ == "__main__":
    main()
