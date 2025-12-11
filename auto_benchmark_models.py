"""
자동 모델 벤치마크 스크립트
- 대상: 베이스, pruned_*_fixed.pt, structured, CoreML/TFLite 등 현재 assets/models 안에 있는 파일
- 측정: 파라미터 수(M), FLOPs(G, thop 필요), 파일 사이즈(MB), CPU 예측 속도(FPS/Latency)

실행:
  python auto_benchmark_models.py
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import PoseModel

# thop(Optional): FLOPs/Params 측정에 필요. 없으면 FLOPs/Params는 0으로 기록.
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("⚠️ thop 미설치: FLOPs/Params는 0으로 기록됩니다. (pip install thop)")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "assets" / "models"
RESULTS_FILE = ROOT / "final_experiment_results.csv"
IMG_SIZE = 640
BASE_MODEL_NAME = "yolo11n_hand_pose.pt"
BASE_MODEL_PATH = MODELS_DIR / BASE_MODEL_NAME


def cleanup_thop_attrs(model: nn.Module):
    """thop 사용 후 남는 속성 제거."""
    for m in model.modules():
        for attr in ("total_ops", "total_params", "n_macs", "n_params"):
            if hasattr(m, attr):
                delattr(m, attr)


def measure_flops_params(model: nn.Module):
    """FLOPs/Params(G/M) 측정 (없으면 0 반환)."""
    if not HAS_THOP:
        return 0.0, 0.0
    try:
        model = model.to("cpu").eval()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        cleanup_thop_attrs(model)
        return params / 1e6, macs / 1e9
    except Exception as e:
        print(f"   ⚠️ FLOPs 측정 실패: {e}")
        return 0.0, 0.0


def measure_size_mb(path: Path) -> float:
    if path.is_dir():
        total = 0
        for dp, _, files in os.walk(path):
            for f in files:
                total += (Path(dp) / f).stat().st_size
        return total / (1024 ** 2)
    return path.stat().st_size / (1024 ** 2)


def measure_speed(model_obj, is_yolo: bool, params_m: float, flops_g: float, precision: str, prune_rate: float, path: Path):
    """CPU 예측 FPS/Latency 측정 후 CSV 기록."""
    size_mb = measure_size_mb(path) if path.exists() else 0.0

    runs = 10
    # Warmup + 측정 루프
    def run_inference():
        if is_yolo:
            dummy_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            model_obj.predict(dummy_img, imgsz=IMG_SIZE, verbose=False, device="cpu")
        else:
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            with torch.no_grad():
                model_obj(dummy)

    for _ in range(3):
        run_inference()

    t0 = time.time()
    for _ in range(runs):
        run_inference()
    t1 = time.time()

    avg_time = (t1 - t0) / runs
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    latency = avg_time * 1000

    print(f"   ✅ {path.name:<35} | {fps:5.1f} FPS | {size_mb:6.2f} MB | Params {params_m:.3f}M | FLOPs {flops_g:.3f}G")

    write_result({
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Model": path.name,
        "Precision": precision,
        "Prune Rate": prune_rate,
        "Params(M)": round(params_m, 3),
        "FLOPs(G)": round(flops_g, 3),
        "FPS": round(fps, 2),
        "Latency(ms)": round(latency, 2),
        "Size(MB)": round(size_mb, 2),
    })


def write_result(row: dict):
    exists = RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def collect_models():
    """벤치마크 대상 모델 리스트 구성."""
    targets = []

    # Base
    if BASE_MODEL_PATH.exists():
        targets.append({"path": BASE_MODEL_PATH, "precision": "fp32", "prune": 0.0, "task": "pose"})

    # pruned_*_fixed.pt
    for p in sorted(MODELS_DIR.glob("yolo11n_hand_pose*_fixed.pt")):
        # prune rate 추출
        rate = 0.0
        for token in p.name.split("_"):
            if token.isdigit():
                try:
                    rate = int(token) / 100.0
                    break
                except:
                    pass
        targets.append({"path": p, "precision": "fp32", "prune": rate, "task": "pose"})

    # structured
    for p in sorted(MODELS_DIR.glob("yolo11n_hand_pose_structured*.pt")):
        targets.append({"path": p, "precision": "fp32", "prune": 0.3 if "30" in p.stem else 0.0, "task": "pose"})

    # Quantized (existing 파일만)
    mlpackage = MODELS_DIR / "yolo11n_hand_pose.mlpackage"
    if mlpackage.exists():
        targets.append({"path": mlpackage, "precision": "int8", "prune": 0.0, "task": "pose"})

    tflite_path = MODELS_DIR / "yolo11n_hand_pose_saved_model" / "yolo11n_hand_pose_int8.tflite"
    if tflite_path.exists():
        targets.append({"path": tflite_path, "precision": "int8", "prune": 0.0, "task": "pose"})

    return targets


def load_model(path: Path, task: str = "pose"):
    """
    모델 로드 시도:
    1) YOLO 형식 -> (obj, True)
    2) torch.load 로 NN 모듈 -> (module, False)
    실패 시 예외 전파
    """
    try:
        obj = YOLO(path, task=task)
        if hasattr(obj, "model"):
            return obj, True
    except Exception:
        pass

    # 순수 PyTorch 모듈 로드 (structured 등)
    torch.serialization.add_safe_globals([PoseModel, torch.nn.modules.container.Sequential])
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, nn.Module):
        raw.eval()
        return raw, False
    raise ValueError("지원하지 않는 모델 형식")


def main():
    print("=" * 70)
    print("🚀 자동 모델 벤치마크 (Params/FLOPs/속도/사이즈)")
    print("=" * 70)

    targets = collect_models()
    if not targets:
        print("❌ 측정할 모델이 없습니다. assets/models 에 파일을 넣어주세요.")
        return

    # Baseline Params/FLOPs (pt 모델만 가능)
    base_params, base_flops = 0.0, 0.0
    if BASE_MODEL_PATH.exists():
        base_model = YOLO(BASE_MODEL_PATH)
        base_params, base_flops = measure_flops_params(base_model.model)
        print(f"[Baseline] Params {base_params:.3f}M | FLOPs {base_flops:.3f}G")

    for item in targets:
        path = item["path"]
        precision = item["precision"]
        prune_rate = item["prune"]
        task = item["task"]

        if not path.exists():
            print(f"   ⚠️ 스킵 (파일 없음): {path}")
            continue

        try:
            model_obj, is_yolo = load_model(path, task)

            # FLOPs/Params
            params_m, flops_g = base_params, base_flops
            if isinstance(model_obj, YOLO):
                if path.suffix == ".pt" and hasattr(model_obj, "model") and isinstance(model_obj.model, nn.Module):
                    params_m, flops_g = measure_flops_params(model_obj.model)
            elif isinstance(model_obj, nn.Module):
                params_m, flops_g = measure_flops_params(model_obj)

            measure_speed(model_obj, is_yolo, params_m, flops_g, precision, prune_rate, path)
        except Exception as e:
            print(f"   ❌ 측정 실패 {path.name}: {e}")
            write_result({
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Model": path.name,
                "Precision": precision,
                "Prune Rate": prune_rate,
                "Params(M)": 0.0,
                "FLOPs(G)": 0.0,
                "FPS": "Error",
                "Latency(ms)": "Error",
                "Size(MB)": measure_size_mb(path) if path.exists() else 0.0,
            })

    print("\n✅ 완료! 결과: final_experiment_results.csv")


if __name__ == "__main__":
    main()
