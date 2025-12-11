"""
Export optimized models for Jetson Nano
- TensorRT INT8 quantization
- TensorRT FP16
- Structured pruning
"""

import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets/models/yolo11n_hand_pose.pt"
OUTPUT_DIR = ROOT / "assets/models"


def export_tensorrt_int8():
    """
    Export YOLO11n to TensorRT INT8 format.
    Requires calibration data for INT8 quantization.
    """
    print("\n" + "=" * 80)
    print("Exporting TensorRT INT8 Model")
    print("=" * 80)

    model = YOLO(str(MODEL_PATH))

    print("\nExporting to TensorRT INT8...")
    print("This may take several minutes...")

    try:
        # Export to TensorRT with INT8
        model.export(
            format='engine',
            half=False,
            int8=True,
            imgsz=640,
            device=0  # Use GPU 0
        )

        engine_path = MODEL_PATH.with_suffix('.engine')
        int8_path = OUTPUT_DIR / "yolo11n_hand_pose_int8.engine"

        if engine_path.exists():
            engine_path.rename(int8_path)
            print(f"\n✓ INT8 TensorRT engine saved: {int8_path}")
            print(f"  Size: {int8_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print("\n✗ Export failed - TensorRT engine not found")

    except Exception as e:
        print(f"\n✗ INT8 export failed: {e}")
        print("\nNote: INT8 quantization requires:")
        print("  - NVIDIA GPU with CUDA")
        print("  - TensorRT installed")
        print("  - pycuda installed")
        print("\nFor Jetson Nano, run this script on the device itself.")


def export_tensorrt_fp16():
    """
    Export YOLO11n to TensorRT FP16 format.
    FP16 provides good speed/accuracy trade-off on Jetson Nano.
    """
    print("\n" + "=" * 80)
    print("Exporting TensorRT FP16 Model")
    print("=" * 80)

    model = YOLO(str(MODEL_PATH))

    print("\nExporting to TensorRT FP16...")
    print("This may take a few minutes...")

    try:
        # Export to TensorRT with FP16
        model.export(
            format='engine',
            half=True,  # FP16
            imgsz=640,
            device=0
        )

        engine_path = MODEL_PATH.with_suffix('.engine')
        fp16_path = OUTPUT_DIR / "yolo11n_hand_pose_fp16.engine"

        if engine_path.exists():
            engine_path.rename(fp16_path)
            print(f"\n✓ FP16 TensorRT engine saved: {fp16_path}")
            print(f"  Size: {fp16_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print("\n✗ Export failed - TensorRT engine not found")

    except Exception as e:
        print(f"\n✗ FP16 export failed: {e}")
        print("\nNote: FP16 export requires:")
        print("  - NVIDIA GPU with CUDA")
        print("  - TensorRT installed")


def prune_model(model, pruning_rate):
    """
    Apply structured pruning to YOLO model.
    Removes entire filters/channels to reduce model size and improve speed.
    """
    import torch.nn.utils.prune as prune

    total_params = 0
    pruned_params = 0

    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Only prune Conv2d layers
            total_params += module.weight.data.numel()

            # Apply L1 unstructured pruning to weights
            prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=0)

            # Make pruning permanent
            prune.remove(module, 'weight')

            # Count pruned parameters (zeros in weight)
            pruned_params += (module.weight.data == 0).sum().item()

    actual_prune_rate = pruned_params / total_params
    print(f"   Pruned {pruned_params}/{total_params} parameters ({actual_prune_rate:.1%})")

    return model


def export_pruned_models():
    """
    Export pruned models at different pruning rates.
    """
    print("\n" + "=" * 80)
    print("Exporting Pruned Models")
    print("=" * 80)

    pruning_rates = [0.1, 0.3, 0.5, 0.7]

    for rate in pruning_rates:
        print(f"\n[Pruning Rate: {rate*100:.0f}%]")

        # Load fresh model
        model = YOLO(str(MODEL_PATH))

        print(f"  Applying {rate*100:.0f}% pruning...")
        model = prune_model(model, rate)

        # Save pruned model
        pruned_path = OUTPUT_DIR / f"yolo11n_hand_pose_pruned_{int(rate*100)}.pt"

        try:
            # Save the pruned model
            torch.save({
                'model': model.model, 
                'pruning_rate': rate
            }, str(pruned_path))

            print(f"  ✓ Saved: {pruned_path}")
            print(f"    Size: {pruned_path.stat().st_size / (1024*1024):.2f} MB")

        except Exception as e:
            print(f"  ✗ Failed to save: {e}")


def export_onnx():
    """
    Export to ONNX format for maximum compatibility.
    """
    print("\n" + "=" * 80)
    print("Exporting ONNX Model")
    print("=" * 80)

    model = YOLO(str(MODEL_PATH))

    print("\nExporting to ONNX...")

    try:
        model.export(
            format='onnx',
            dynamic=False,
            simplify=True,
            opset=12,
            imgsz=640
        )

        onnx_path = MODEL_PATH.with_suffix('.onnx')
        final_path = OUTPUT_DIR / "yolo11n_hand_pose.onnx"

        if onnx_path.exists():
            if final_path.exists():
                final_path.unlink()
            onnx_path.rename(final_path)
            print(f"\n✓ ONNX model saved: {final_path}")
            print(f"  Size: {final_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print("\n✗ Export failed - ONNX file not found")

    except Exception as e:
        print(f"\n✗ ONNX export failed: {e}")


def main():
    print("=" * 80)
    print("YOLO11n Hand Pose - Model Optimization Export")
    print("=" * 80)

    if not MODEL_PATH.exists():
        print(f"\n✗ Error: Model not found at {MODEL_PATH}")
        return

    print(f"\nSource model: {MODEL_PATH}")
    print(f"Size: {MODEL_PATH.stat().st_size / (1024*1024):.2f} MB")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\nSelect export format:")
    print("  1) TensorRT INT8 (best speed, ~2% accuracy loss)")
    print("  2) TensorRT FP16 (good speed, minimal accuracy loss)")
    print("  3) Pruned models (10%, 30%, 50%, 70%)")
    print("  4) ONNX (for compatibility)")
    print("  5) All of the above")

    try:
        choice = input("\nEnter choice (1-5): ").strip()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return

    if choice == '1':
        export_tensorrt_int8()
    elif choice == '2':
        export_tensorrt_fp16()
    elif choice == '3':
        export_pruned_models()
    elif choice == '4':
        export_onnx()
    elif choice == '5':
        export_onnx()
        export_pruned_models()
        export_tensorrt_fp16()
        export_tensorrt_int8()
    else:
        print("Invalid choice")
        return

    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)
    print("\nTo use the optimized models:")
    print("  INT8: python src/emoji_reactor/app.py --precision int8")
    print("  FP16: python src/emoji_reactor/app.py --precision fp16")
    print("  FP32: python src/emoji_reactor/app.py --precision fp32")
    print("\nNote: TensorRT models (.engine) are platform-specific.")
    print("      Export INT8/FP16 on the target device (Jetson Nano).")
    print("=" * 80)


if __name__ == "__main__":
    main()
