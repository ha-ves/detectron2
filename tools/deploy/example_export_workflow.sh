#!/bin/bash
# Example script showing complete workflow for exporting and deploying
# mask_rcnn_R_50_FPN_100ep_LSJ model with LibTorch

set -e  # Exit on error

echo "=== Detectron2 LibTorch Export Example ==="
echo "This script demonstrates how to export a LazyConfig model to LibTorch"
echo ""

# Configuration
CONFIG_FILE="configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py"
OUTPUT_DIR="./libtorch_export"
SAMPLE_IMAGE="${1:-demo/sample.jpg}"
MODEL_WEIGHTS="${2:-}"  # Optional: path to model weights

# Step 1: Export model to TorchScript
echo "Step 1: Exporting model to TorchScript..."
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"

EXPORT_CMD="python tools/deploy/export_model_lazy.py \
    --config-file $CONFIG_FILE \
    --output $OUTPUT_DIR \
    --export-method tracing \
    --format torchscript"

if [ ! -z "$SAMPLE_IMAGE" ] && [ -f "$SAMPLE_IMAGE" ]; then
    echo "Sample image: $SAMPLE_IMAGE"
    EXPORT_CMD="$EXPORT_CMD --sample-image $SAMPLE_IMAGE"
else
    echo "Warning: No sample image provided or file not found"
    echo "Note: Tracing requires a sample image. You may need to provide one."
fi

if [ ! -z "$MODEL_WEIGHTS" ] && [ -f "$MODEL_WEIGHTS" ]; then
    echo "Model weights: $MODEL_WEIGHTS"
    EXPORT_CMD="$EXPORT_CMD train.init_checkpoint=$MODEL_WEIGHTS"
else
    echo "Note: No model weights specified. Exporting with random initialization."
    echo "      To use trained weights, provide path as second argument."
fi

# Add device specification
EXPORT_CMD="$EXPORT_CMD train.device=cpu"

echo ""
echo "Running export command:"
echo "$EXPORT_CMD"
echo ""

# Uncomment to actually run the export
# eval $EXPORT_CMD

echo "Export command prepared. To actually run it, uncomment the 'eval' line in this script."
echo ""

# Step 2: Build C++ code
echo "Step 2: Build C++ inference code with custom ops"
echo ""
echo "Build commands:"
echo "  cd tools/deploy"
echo "  mkdir -p build && cd build"
echo ""
echo "  # Configure CMake (adjust paths to your LibTorch and TorchVision installations)"
echo "  cmake -DCMAKE_PREFIX_PATH=\"/path/to/libtorch;/path/to/torchvision\" \\"
echo "        -DCMAKE_BUILD_TYPE=Release \\"
echo "        -C ../CMakeLists_with_ops.txt .."
echo ""
echo "  # Build"
echo "  make -j\$(nproc)"
echo ""

# Step 3: Run inference
echo "Step 3: Run inference in C++"
echo ""
echo "Inference command:"
echo "  ./tools/deploy/build/torchscript_mask_rcnn_lazy \\"
echo "      $OUTPUT_DIR/model.ts \\"
echo "      $SAMPLE_IMAGE \\"
echo "      tracing \\"
echo "      ./tools/deploy/build/libdetectron2_ops.so"
echo ""

echo "=== Next Steps ==="
echo "1. Install dependencies: LibTorch, TorchVision, OpenCV"
echo "2. Train or download model weights for mask_rcnn_R_50_FPN_100ep_LSJ"
echo "3. Uncomment the export command in this script and run it"
echo "4. Follow the build and inference commands above"
echo ""
echo "For detailed instructions, see tools/deploy/README_LAZYCONFIG.md"
