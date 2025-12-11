"""
🖐️ MobileHand Pruning + Quantization 벤치마크

MobileHand: https://github.com/gmntu/mobilehand
- MobileNetV3-Small backbone
- 21개 hand keypoint + 3D mesh  
- FreiHAND/STB 데이터로 학습됨
- 3.82M params

Pruning/Quantization 완전 지원!
"""

import sys
sys.path.insert(0, 'mobilehand_repo/code')

import os
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

from utils_mobilenet_v3 import mobilenetv3_small
from utils_linear_model import LinearModel

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "assets/models"
PRETRAINED_PATH = ROOT / "mobilehand_repo/model/hmr_model_freihand_auc.pth"


# =========================================================
# MobileHand 모델 정의 (간소화 버전)
# =========================================================

class Regressor(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, num_param, num_iters, max_batch_size):
        super().__init__(fc_layers, use_dropout, drop_prob, use_ac_func)
        self.num_param = num_param
        self.num_iters = num_iters
        mean = np.zeros(self.num_param, dtype=np.float32)
        mean_param = np.tile(mean, max_batch_size).reshape((max_batch_size, -1))
        self.register_buffer('mean_param', torch.from_numpy(mean_param).float())
    
    def forward(self, inputs):
        bs = inputs.shape[0]
        param = self.mean_param[:bs, :]
        for _ in range(self.num_iters):
            total = torch.cat([inputs, param], dim=1)
            param = param + self.fc_blocks(total)
        return param


class MobileHandEncoder(nn.Module):
    """
    MobileHand의 Encoder 부분만 (Pruning 가능)
    
    입력: (B, 3, 224, 224)
    출력: (B, 576) - 특징 벡터
    """
    def __init__(self):
        super().__init__()
        self.encoder = mobilenetv3_small()
    
    def forward(self, x):
        return self.encoder(x)
    
    def load_pretrained(self, path):
        """Pretrained weight에서 encoder 부분만 로드"""
        state_dict = torch.load(path, map_location='cpu')
        
        # encoder. 접두사 제거
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                encoder_state[k.replace('encoder.', '')] = v
        
        self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"[Pretrained] Loaded encoder from {path}")


# =========================================================
# Pruning
# =========================================================

def prune_encoder(model, prune_ratio=0.5):
    """MobileNetV3 Encoder Pruning"""
    model = copy.deepcopy(model).cpu()  # CPU에서 pruning
    model.eval()
    
    example = torch.randn(1, 3, 224, 224)
    
    # Linear 레이어 무시 (출력 크기 유지)
    ignored = [m for m in model.modules() if isinstance(m, nn.Linear)]
    
    before = sum(p.numel() for p in model.parameters())
    
    pruner = tp.pruner.MagnitudePruner(
        model,
        example,
        importance=tp.importance.MagnitudeImportance(p=2),
        iterative_steps=1,
        pruning_ratio=prune_ratio,
        ignored_layers=ignored,
        round_to=8,
    )
    
    pruner.step()
    
    after = sum(p.numel() for p in model.parameters())
    
    print(f"[Pruning] {prune_ratio*100:.0f}%: {before/1e6:.3f}M → {after/1e6:.3f}M ({(1-after/before)*100:.1f}%↓)")
    
    return model


def get_flops(model, input_size=224):
    if not HAS_THOP:
        return 0
    
    model = copy.deepcopy(model).eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    
    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        for m in model.modules():
            for attr in ['total_ops', 'total_params']:
                if hasattr(m, attr):
                    delattr(m, attr)
        return macs / 1e9
    except:
        return 0


def measure_speed(model, input_size=224, device='cpu', num_test=100):
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            model(dummy)
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_test):
            t0 = time.time()
            model(dummy)
            if device == 'mps':
                torch.mps.synchronize()
            times.append(time.time() - t0)
    
    avg_time = sum(times) / len(times)
    return 1.0 / avg_time, avg_time * 1000


# =========================================================
# 메인
# =========================================================

def main():
    print("=" * 70)
    print("🖐️ MobileHand Pruning + Quantization 벤치마크")
    print("=" * 70)
    
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"Device: mps")
    else:
        device = "cpu"
        print(f"Device: cpu")
    
    results = []
    
    # =========================================================
    # 1. Original MobileHand Encoder
    # =========================================================
    print("\n[1] Original MobileHand Encoder...")
    
    model = MobileHandEncoder()
    
    if PRETRAINED_PATH.exists():
        model.load_pretrained(PRETRAINED_PATH)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    flops = get_flops(model)
    fps, latency = measure_speed(model, device=device)
    
    print(f"   Params: {params:.3f}M, FLOPs: {flops:.3f}G, FPS: {fps:.1f}")
    
    results.append({
        'name': 'MobileHand_Original',
        'params': params,
        'flops': flops,
        'fps': fps,
        'latency': latency,
    })
    
    # 저장
    save_path = MODELS_DIR / "mobilehand_encoder.pt"
    torch.save(model.state_dict(), save_path)
    
    # =========================================================
    # 2. Pruning
    # =========================================================
    print("\n[2] Pruning...")
    
    for ratio in [0.3, 0.5, 0.7]:
        pruned = prune_encoder(model, prune_ratio=ratio)
        
        p_params = sum(p.numel() for p in pruned.parameters()) / 1e6
        p_flops = get_flops(pruned)
        p_fps, p_lat = measure_speed(pruned, device=device)
        
        print(f"   Params: {p_params:.3f}M, FLOPs: {p_flops:.3f}G, FPS: {p_fps:.1f}")
        
        results.append({
            'name': f'MobileHand_Pruned_{int(ratio*100)}%',
            'params': p_params,
            'flops': p_flops,
            'fps': p_fps,
            'latency': p_lat,
        })
        
        # 저장
        save_pruned = MODELS_DIR / f"mobilehand_encoder_pruned_{int(ratio*100)}.pt"
        torch.save(pruned.state_dict(), save_pruned)
    
    # =========================================================
    # 3. Quantization (INT8)
    # =========================================================
    print("\n[3] Quantization (INT8)...")
    
    # 1. Measure Size (Static Quantization Conversion)
    try:
        # Prepare model copy
        model_int8 = copy.deepcopy(model).cpu()
        model_int8.eval()
        
        # Fuse (Optional/Hard for MobileNetV3 blocks without code change)
        
        # Config
        model_int8.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
        torch.backends.quantized.engine = 'qnnpack'
        
        # Prepare
        torch.ao.quantization.prepare(model_int8, inplace=True)
        
        # Calibrate
        print("   Calibrating...")
        with torch.no_grad():
            for _ in range(10):
                model_int8(torch.randn(1, 3, 224, 224))
        
        # Convert
        torch.ao.quantization.convert(model_int8, inplace=True)
        print("   Quantization done.")
        
        # Save & Size Check
        torch.save(model_int8.state_dict(), MODELS_DIR / "temp_int8.pt")
        q_size_mb = os.path.getsize(MODELS_DIR / "temp_int8.pt") / 1e6
        os.remove(MODELS_DIR / "temp_int8.pt")
        
        # Attempt Inference
        try:
             print("   Benchmarking INT8 Speed (CPU)...")
             q_fps, q_lat = measure_speed(model_int8, device='cpu', num_test=50)
             q_note = "CPU Only"
        except RuntimeError as e:
             print(f"   ⚠️ INT8 Inference failed (likely architecture incompatibility on PyTorch CPU): {e}")
             q_fps, q_lat = 0, 0
             q_note = "Size Only"
        
        print(f"   Size: {q_size_mb:.2f}MB, FPS: {q_fps:.1f}")
        
        results.append({
            'name': 'MobileHand_INT8 (PTQ)',
            'params': q_size_mb,
            'flops': 0,
            'fps': q_fps,
            'latency': q_lat,
            'note': q_note
        })
        
        # Save valid model
        torch.save(model_int8.state_dict(), MODELS_DIR / "mobilehand_encoder_int8.pt")
        
    except Exception as e:
        print(f"   ❌ Quantization failed: {e}")

    # =========================================================
    # 4. FP16 Benchmark (MPS/CUDA)
    # =========================================================
    if device in ['cuda', 'mps']:
        print(f"\n[4] FP16 Benchmark ({device})...")
        try:
            model_fp16 = copy.deepcopy(model).half().to(device)
            dummy_fp16 = torch.randn(1, 3, 224, 224).half().to(device)
            
            # Warmup & Measure
            with torch.no_grad():
                model_fp16(dummy_fp16)
            
            fp16_times = []
            for _ in range(100):
                t0 = time.time()
                with torch.no_grad():
                    model_fp16(dummy_fp16)
                    if device == 'mps': torch.mps.synchronize()
                    elif device == 'cuda': torch.cuda.synchronize()
                fp16_times.append(time.time() - t0)
                
            fp16_lat = sum(fp16_times) / len(fp16_times) * 1000
            fp16_fps = 1000 / fp16_lat
            
            print(f"   FPS: {fp16_fps:.1f} ({device})")
            
            results.append({
                'name': 'MobileHand_FP16',
                'params': params,
                'flops': flops,
                'fps': fp16_fps,
                'latency': fp16_lat,
                'note': device.upper()
            })
        except Exception as e:
            print(f"   ❌ FP16 Benchmark failed: {e}")

    # =========================================================
    # 5. ONNX Export
    # =========================================================
    print("\n[5] ONNX Export...")
    
    try:
        model.cpu().eval()
        dummy = torch.randn(1, 3, 224, 224)
        
        onnx_path = MODELS_DIR / "mobilehand_encoder.onnx"
        torch.onnx.export(model, dummy, str(onnx_path),
                        input_names=['input'], output_names=['features'],
                        opset_version=11)
        print(f"   Saved: {onnx_path.name}")
        
        # Pruned 50% ONNX
        pruned_50 = prune_encoder(model, 0.5)
        onnx_pruned = MODELS_DIR / "mobilehand_encoder_pruned_50.onnx"
        torch.onnx.export(pruned_50.cpu(), dummy, str(onnx_pruned),
                        input_names=['input'], output_names=['features'],
                        opset_version=11)
        print(f"   Saved: {onnx_pruned.name}")
    except Exception as e:
        print(f"   ONNX Export Warning: {e}")
    
    # =========================================================
    # 결과 보고서
    # =========================================================
    print("\n" + "=" * 100)
    print(f"{'Model':<30} | {'Size/Params':<12} | {'FLOPs(G)':<12} | {'FPS':<10} | {'Latency(ms)':<12} | {'Note':<10}")
    print("-" * 100)
    for r in results:
        note = r.get('note', '')
        # Handle formatting for params vs size
        if 'INT8' in r['name']:
            p_str = f"{r['params']:.2f}MB"
        else:
            p_str = f"{r['params']:.2f}M"
            
        # Handle N/A FPS
        if r['fps'] == 0:
            fps_str = "N/A"
            lat_str = "N/A"
        else:
            fps_str = f"{r['fps']:.1f}"
            lat_str = f"{r['latency']:.1f}"
            
        print(f"{r['name']:<30} | {p_str:<12} | {r['flops']:<12.3f} | {fps_str:<10} | {lat_str:<12} | {note:<10}")
    print("=" * 100)
    
    print(f"""
✅ 완료!

📁 생성된 파일:
   • mobilehand_encoder.pt
   • mobilehand_encoder_pruned_*.pt
   • mobilehand_encoder_int8.pt (Quantized)
   • mobilehand_encoder.onnx
   • mobilehand_encoder_pruned_50.onnx

🚀 Jetson Nano 최적화 가이드:
   1. Pruning된 모델 (Pruned 50%)을 선택합니다.
   2. Jetson에서 TensorRT(FP16)로 변환하여 사용합니다.
      $ trtexec --onnx=mobilehand_encoder_pruned_50.onnx --fp16 --saveEngine=mobilehand_fp16.engine
   3. 예상 성능: Pruning(30% Speedup) + FP16(2~3x Speedup) = 약 3~4배 성능 향상
""")


if __name__ == "__main__":
    main()
