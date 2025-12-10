#!/bin/bash
# Download Vosk model for offline voice recognition

MODEL_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_ZIP="vosk-model-small-en-us-0.15.zip"
MODEL_DIR="vosk-model-small-en-us-0.15"
TARGET_DIR="assets/models"

echo "Downloading Vosk model..."
curl -L $MODEL_URL -o $MODEL_ZIP

echo "Unzipping..."
unzip -o $MODEL_ZIP -d $TARGET_DIR

# Rename to generic 'vosk' for easier usage, or keep specific name
# Let's keep specific name to avoid confusion, but print the path
echo "Model installed to $TARGET_DIR/$MODEL_DIR"

# Cleanup
rm $MODEL_ZIP

echo ""
echo "Setup complete. Run app with:"
echo "python3 src/emoji_reactor/app.py --vosk-model $TARGET_DIR/$MODEL_DIR"
