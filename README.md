<<<<<<< HEAD
# Emoji Reactor (PyTorch runtime)

Cleaned repository focused on a PyTorch-only hand/face pipeline and the emoji/monkey reactor demo with pruning/quantization options.

## Layout
- `src/hand_tracking/` — PyTorch-only pipeline that converts the cached ONNX models with onnx2pytorch (no onnxruntime).
- `src/emoji_reactor/app.py` — main demo (emoji/monkey modes, gestures, optional voice, background music).
- `assets/emojis/` — emoji images (default + monkey variants).
- `assets/audio/` — background music (`yessir.mp3`).
- `assets/models/` — cached/exported ONNX models (converted to PyTorch at runtime).
- `demos/simple_hand_app.py` — minimal hand-tracking viewer.

## Quickstart
```bash
pip install -r requirements.txt  # includes onnx2pytorch, mediapipe, etc.

# Emoji reactor (emoji mode by default)
python src/emoji_reactor/app.py --precision fp16 --prune 0.0

# Switch to monkey mode via hotword (if Vosk installed) or keys:
#   press 'm' for monkey, 'e' for emoji, 'q'/ESC to quit

# Minimal hand viewer
python demos/simple_hand_app.py --precision fp16
```

### Options (app.py)
- `--precision {fp32,fp16,int8}` / `--prune` — model export options.
- `--input-device` — mic device index/name (from `sounddevice.query_devices()`).
- `--raise-thresh`, `--confused-thresh`, `--frustrated-thresh` — gesture thresholds.
- `--no-voice` / `--voice-backend {hf,vosk}` / `--vosk-model` — voice control settings.
- HF voice: `--voice-backend hf --hf-asr-model <model or .safetensors> --hf-asr-base openai/whisper-tiny --hf-device cpu|cuda`. If `whisper.safetensors` is present in repo root, it auto-loads with HF backend.
- `--no-music` — disable looping background music.

### Custom weights → pruned/quantized ONNX
If you have your own `.pth` weights:
```bash
python tools/export_models.py \
  --detector-weights /path/to/blazepalm.pth \
  --landmark-weights /path/to/blazehand_landmark.pth \
  --precision fp16 --prune 0.3 --prune-mode channel_l1 --tag myrun
# ONNX saved to assets/models/palm_detector_fp16_pruned30_myrun.onnx, hand_landmark_fp16_pruned30_myrun.onnx
```
Then run the app/viewer with the same `--precision/--prune`/`--tag` to pick up those files.

### Runtime with pruning/quantization
```bash
# FP16 + 30% pruning (Jetson/TensorRT-friendly)
python src/emoji_reactor/app.py --precision fp16 --prune 0.3

# INT8 (CPU-friendly)
python src/emoji_reactor/app.py --precision int8 --prune 0.0

# FP32 baseline
python src/emoji_reactor/app.py --precision fp32 --prune 0.0
```

## Notes
- Assets live in `assets/emojis` and `assets/audio`; place additional emoji sets there if needed.
- ONNX models are cached in `assets/models`; they'll be generated on first run.
=======
# Embedded_AI
>>>>>>> c1e95a5438b4e7ef8db506243adadc3cffdca8af
