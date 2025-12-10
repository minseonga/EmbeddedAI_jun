#!/bin/bash
# Optimized run script for Jetson Nano
# This script launches the emoji reactor with optimal settings for Jetson Nano

echo "=========================================="
echo "Emoji Reactor - Jetson Nano Optimized"
echo "=========================================="
echo ""
echo "Optimizations enabled:"
echo "  • Haar Cascade face detection (ultra-fast, built-in)"
echo "  • MediaPipe Hands (model_complexity=0)"
echo "  • INT8 quantization"
echo "  • FP16 precision for Jetson"
echo "  • Compatible with apt-get OpenCV"
echo ""

# Set CUDA optimizations for Jetson Nano
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Run with optimized settings
python3 src/emoji_reactor/app.py \
    --precision fp16 \
    --no-gstreamer \
    --camera 0 \
    "$@"

# Usage examples:
# ./run_optimized.sh                           # Run with defaults (Haar Cascade)
# ./run_optimized.sh --use-dnn                 # Use DNN face detector (more accurate)
# ./run_optimized.sh —raise-thresh 0.3        # Adjust gesture threshold
# ./run_optimized.sh —no-voice                # Disable voice control
