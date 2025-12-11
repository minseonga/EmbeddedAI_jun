"""
Jetson Nano용 경량 모델 재생성 + 벤치마크 스크립트
- 기준: assets/models/yolo11n_hand_pose.pt (YOLO 포맷)
- 생성: 구조적 프루닝(30/50/70), 비구조적 프루닝(30/50/70), TFLite INT8
- 측정: auto_benchmark_models.py 호출로 Params/FLOPs/CPU FPS/Size 기록

실행:
  python rebuild_nano_models.py
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "assets" / "models"
BASE_MODEL = MODELS_DIR / "yolo11n_hand_pose.pt"
IMG_SIZE = 640

# 프루닝 비율
PRUNE_RATES = [0.3, 0.5, 0.7]


def load_base():
    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"Base model not found: {BASE_MODEL}")
    return YOLO(BASE_MODEL)


def structured_prune(rate: float):
    """torch_pruning 사용해 채널 단위 구조적 프루닝."""
    try:
        import torch_pruning as tp
    except ImportError:
        print("⚠️ torch_pruning 미설치: 구조적 프루닝 건너뜀")
        return

    pct = int(rate * 100)
    final_path = MODELS_DIR / f"yolo11n_hand_pose_pruned_s_{pct}_fixed.pt"
    if final_path.exists():
        final_path.unlink()

    yolo_obj = load_base()
    model_raw = yolo_obj.model

    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    imp = tp.importance.MagnitudeImportance(p=1)
    ignored = []
    for m in model_raw.modules():
        if isinstance(m, torch.nn.Linear) and getattr(model_raw, "head", None) and m.out_features == model_raw.head.nc:
            ignored.append(m)

    pruner = tp.pruner.MagnitudePruner(
        model_raw,
        example,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=rate,
        ignored_layers=ignored,
    )
    pruner.step()

    for _, m in model_raw.named_modules():
        if hasattr(m, "weight_mask"):
            prune.remove(m, "weight")

    yolo_obj.model = model_raw
    yolo_obj.save(final_path)
    print(f"[ok] structured {pct}% -> {final_path.name}")


def unstructured_prune(rate: float):
    """가중치 단위 비구조적 글로벌 프루닝."""
    pct = int(rate * 100)
    final_path = MODELS_DIR / f"yolo11n_hand_pose_pruned_{pct}_fixed.pt"
    if final_path.exists():
        final_path.unlink()

    yolo_obj = load_base()
    model_raw = yolo_obj.model
    params = []
    for m in model_raw.modules():
        if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
            if m.weight.dim() >= 2:  # conv/linear
                params.append((m, "weight"))
    if not params:
        print(f"[warn] 대상 파라미터 없음 -> skip {pct}%")
        return

    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=rate)
    for _, m in model_raw.named_modules():
        if hasattr(m, "weight_mask"):
            prune.remove(m, "weight")

    yolo_obj.model = model_raw
    yolo_obj.save(final_path)
    print(f"[ok] unstructured {pct}% -> {final_path.name}")


def export_tflite_int8():
    target = MODELS_DIR / "yolo11n_hand_pose_saved_model" / "yolo11n_hand_pose_int8.tflite"
    if target.exists():
        print("[skip] TFLite INT8 이미 존재")
        return
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("[skip] TFLite INT8 건너뜀 (tensorflow 미설치)")
        return
    print("[export] TFLite INT8 생성 중...")
    YOLO(BASE_MODEL).export(format="tflite", int8=True, imgsz=IMG_SIZE)
    if target.exists():
        print(f"[ok] TFLite INT8 -> {target}")
    else:
        print("[fail] TFLite INT8 생성 실패 (파일 없음)")


def run_benchmark():
    print("\n[bench] auto_benchmark_models.py 실행...")
    os.system(f"\"{sys.executable}\" auto_benchmark_models.py")  # same env python


def main():
    print("==========================================")
    print("재생성: 구조적/비구조적 프루닝 + TFLite INT8 + 벤치마크")
    print("기반 모델:", BASE_MODEL)
    print("==========================================")

    for r in PRUNE_RATES:
        structured_prune(r)
    for r in PRUNE_RATES:
        unstructured_prune(r)
    export_tflite_int8()
    run_benchmark()


if __name__ == "__main__":
    main()
