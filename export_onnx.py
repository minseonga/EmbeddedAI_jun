"""
Export MMPose hand/face models to ONNX format.

Usage:
    python export_onnx.py

Requires mmcv and mmpose installed.
"""

import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MMPOSE_ROOT = ROOT / "mmpose"

# Model paths
HAND_MODEL_PATH = ROOT / "assets/models/mobilenetv2_coco_wholebody_hand_256x256-06b8c877_20210909.pth"
FACE_MODEL_PATH = ROOT / "assets/models/rtmpose-t_simcc-face6_pt-in1k_120e-256x256-df79d9a5_20230529.pth"

# Config paths
HAND_CONFIG = MMPOSE_ROOT / "configs/hand_2d_keypoint/topdown_heatmap/coco_wholebody_hand/td-hm_mobilenetv2_8xb32-210e_coco-wholebody-hand-256x256.py"
FACE_CONFIG = MMPOSE_ROOT / "configs/face_2d_keypoint/rtmpose/face6/rtmpose-t_8xb256-120e_face6-256x256.py"

OUTPUT_DIR = ROOT / "assets/models"


def export_hand_model():
    """Export hand model to ONNX."""
    from mmpose.apis import init_model

    print("[Hand] Loading model...")
    model = init_model(str(HAND_CONFIG), str(HAND_MODEL_PATH), device='cpu')
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, 256, 256)

    # Export
    output_path = OUTPUT_DIR / "hand_mobilenetv2_256x256.onnx"
    print(f"[Hand] Exporting to {output_path}...")

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=['input'],
        output_names=['heatmaps'],
        dynamic_axes={
            'input': {0: 'batch'},
            'heatmaps': {0: 'batch'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    print(f"[Hand] Done! Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


def export_face_model():
    """Export face model to ONNX."""
    from mmpose.apis import init_model

    print("[Face] Loading model...")
    model = init_model(str(FACE_CONFIG), str(FACE_MODEL_PATH), device='cpu')
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, 256, 256)

    # Export
    output_path = OUTPUT_DIR / "face_rtmpose_t_256x256.onnx"
    print(f"[Face] Exporting to {output_path}...")

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=['input'],
        output_names=['pred_x', 'pred_y'],
        dynamic_axes={
            'input': {0: 'batch'},
            'pred_x': {0: 'batch'},
            'pred_y': {0: 'batch'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    print(f"[Face] Done! Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


def verify_onnx(onnx_path: Path):
    """Verify ONNX model."""
    import onnx
    import onnxruntime as ort

    print(f"[Verify] Checking {onnx_path.name}...")

    # Check ONNX model
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print(f"  ONNX check passed")

    # Test inference
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    dummy = torch.randn(*[s if isinstance(s, int) else 1 for s in input_shape]).numpy()
    outputs = session.run(None, {input_name: dummy})

    print(f"  Input: {input_name} {input_shape}")
    for i, out in enumerate(session.get_outputs()):
        print(f"  Output {i}: {out.name} {outputs[i].shape}")

    print(f"  Inference OK")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("MMPose Model Export to ONNX")
    print("=" * 50)

    # Export
    hand_onnx = export_hand_model()
    face_onnx = export_face_model()

    print()

    # Verify
    verify_onnx(hand_onnx)
    verify_onnx(face_onnx)

    print()
    print("=" * 50)
    print("Export completed!")
    print(f"  Hand: {hand_onnx}")
    print(f"  Face: {face_onnx}")
    print("=" * 50)
