#!/usr/bin/env python3
"""
RTMPose Optimization Benchmark
Tests quantization and pruning effects on RTMPose model.
"""

import sys
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hand_tracking.pipeline import RTMPoseModel, MODEL_PATH


def get_model_size(model):
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def count_parameters(model):
    """Count total and non-zero parameters."""
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p != 0).sum().item()
    return total, nonzero


def apply_pruning(model, amount=0.3):
    """Apply unstructured L1 pruning to Conv2d and Linear layers."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make pruning permanent
    return model


def apply_quantization(model):
    """Apply dynamic INT8 quantization."""
    model_cpu = model.cpu()
    model_cpu.eval()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )
    return quantized


def benchmark_model(model, device, num_warmup=10, num_runs=100, use_fp16=False):
    """Benchmark model inference time."""
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 192)
    if device.type == 'cuda':
        dummy_input = dummy_input.to(device)
        if use_fp16:
            dummy_input = dummy_input.half()
            model = model.half()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

    return np.array(times)


def main():
    print("\n" + "=" * 80)
    print("RTMPose OPTIMIZATION BENCHMARK")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load base model
    print(f"\nLoading RTMPose model from {MODEL_PATH.name}...")
    base_model = RTMPoseModel(num_keypoints=133, input_size=(256, 192))

    if MODEL_PATH.exists():
        ckpt = torch.load(str(MODEL_PATH), map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        base_model.load_state_dict(state_dict)
        print("Weights loaded successfully!")
    else:
        print("Warning: Using random weights (model file not found)")

    # Configurations to test
    configs = [
        {"name": "FP32 Baseline", "fp16": False, "quantize": False, "prune": 0.0},
        {"name": "FP16 (CUDA)", "fp16": True, "quantize": False, "prune": 0.0},
        {"name": "30% Pruning", "fp16": False, "quantize": False, "prune": 0.3},
        {"name": "50% Pruning", "fp16": False, "quantize": False, "prune": 0.5},
        {"name": "FP16 + 30% Pruning", "fp16": True, "quantize": False, "prune": 0.3},
        {"name": "FP16 + 50% Pruning", "fp16": True, "quantize": False, "prune": 0.5},
    ]

    # Add INT8 quantization config (CPU only)
    if device.type == 'cpu':
        configs.insert(2, {"name": "INT8 Quantization", "fp16": False, "quantize": True, "prune": 0.0})
        configs.append({"name": "INT8 + 30% Pruning", "fp16": False, "quantize": True, "prune": 0.3})

    results = []

    for cfg in configs:
        print(f"\n{'─' * 80}")
        print(f"Configuration: {cfg['name']}")
        print(f"{'─' * 80}")

        try:
            # Copy model
            model = copy.deepcopy(base_model)

            # Apply pruning
            if cfg['prune'] > 0:
                print(f"Applying {cfg['prune']*100:.0f}% pruning...")
                model = apply_pruning(model, cfg['prune'])

            # Apply quantization (CPU only)
            if cfg['quantize']:
                print("Applying INT8 quantization...")
                model = apply_quantization(model)
                test_device = torch.device('cpu')
            else:
                test_device = device
                model = model.to(test_device)

            # Get stats
            total_params, nonzero_params = count_parameters(model)
            sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0
            size_mb = get_model_size(model)

            print(f"Parameters: {total_params:,} total, {nonzero_params:,} non-zero")
            print(f"Sparsity: {sparsity*100:.1f}%")
            print(f"Model size: {size_mb:.2f} MB")

            # Benchmark
            print(f"\nBenchmarking (warmup={10}, runs={100})...")
            times = benchmark_model(model, test_device, use_fp16=cfg['fp16'])

            mean_ms = np.mean(times)
            std_ms = np.std(times)
            p95_ms = np.percentile(times, 95)
            fps = 1000.0 / mean_ms

            results.append({
                "name": cfg["name"],
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "p95_ms": p95_ms,
                "fps": fps,
                "size_mb": size_mb,
                "sparsity": sparsity,
            })

            print(f"\nResults:")
            print(f"  Mean:     {mean_ms:>7.2f} ms")
            print(f"  Std:      {std_ms:>7.2f} ms")
            print(f"  P95:      {p95_ms:>7.2f} ms")
            print(f"  FPS:      {fps:>7.1f}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<25} {'Size':>8} {'Sparsity':>10} {'Mean':>10} {'P95':>10} {'FPS':>8} {'Speedup':>8}")
    print("─" * 80)

    baseline_fps = results[0]["fps"] if results else 1.0
    for r in results:
        speedup = r["fps"] / baseline_fps
        print(f"{r['name']:<25} {r['size_mb']:>7.1f}MB {r['sparsity']*100:>9.1f}% "
              f"{r['mean_ms']:>9.2f}ms {r['p95_ms']:>9.2f}ms {r['fps']:>7.1f} {speedup:>7.2f}x")

    print("=" * 80)

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    if device.type == 'cuda':
        print("  • Use FP16 for best GPU performance")
        print("  • FP16 + 30-50% pruning for memory-constrained devices")
    else:
        print("  • Use INT8 quantization for best CPU performance")
        print("  • INT8 + pruning for maximum optimization")
    print("  • Pruning reduces model size but may affect accuracy")
    print("  • Test accuracy after optimization before deployment")
    print("=" * 80)


if __name__ == "__main__":
    main()
