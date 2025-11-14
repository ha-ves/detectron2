# LibTorch Export for LazyConfig Models

This guide explains how to export Detectron2 models defined with LazyConfig (like `configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py`) for deployment with LibTorch (C++), including support for custom ops.

## Overview

The new baseline models in `configs/new_baselines/` use LazyConfig, which is a more flexible configuration system than the traditional YAML configs. This directory provides tools to:

1. Export LazyConfig models to TorchScript format
2. Build Detectron2 custom ops as a shared library for LibTorch
3. Run inference in C++ with the exported model and custom ops

## Prerequisites

- CMake >= 3.12
- LibTorch (C++ distribution of PyTorch)
- OpenCV
- TorchVision C++ library
- Detectron2 installed with custom ops built

## Step 1: Export Model to TorchScript

Use the `export_model_lazy.py` script to export a LazyConfig model:

```bash
# Export with tracing (recommended)
python tools/deploy/export_model_lazy.py \
    --config-file configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
    --output ./output \
    --export-method tracing \
    --format torchscript \
    --sample-image /path/to/sample_image.jpg \
    train.init_checkpoint=/path/to/model_weights.pth \
    train.device=cuda

# Or export with scripting
python tools/deploy/export_model_lazy.py \
    --config-file configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
    --output ./output \
    --export-method scripting \
    --format torchscript \
    train.init_checkpoint=/path/to/model_weights.pth
```

**Important Notes:**
- For `--export-method tracing`, you must provide a `--sample-image` 
- You can specify model weights using `train.init_checkpoint=...` in the opts
- Set the device with `train.device=cuda` or `train.device=cpu`

This will create:
- `output/model.ts` - The exported TorchScript model
- `output/model.txt` - Human-readable TorchScript IR

## Step 2: Build Custom Ops Library and Inference Executable

The Detectron2 custom ops (deformable convolution, rotated ROI align, etc.) need to be built as a shared library for LibTorch.

### Option A: Build with Custom Ops (Recommended)

Use the `CMakeLists_with_ops.txt` configuration:

```bash
cd tools/deploy
mkdir build
cd build

# Configure with CMake
cmake -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/torchvision" \
      -DCMAKE_BUILD_TYPE=Release \
      -C ../CMakeLists_with_ops.txt ..

# Build
cmake --build . --config Release

# This creates:
# - libdetectron2_ops.so - Shared library with custom ops
# - torchscript_mask_rcnn - Inference executable
```

### Option B: Build without Custom Ops

If your model doesn't use custom ops, you can use the original CMakeLists.txt:

```bash
cd tools/deploy
mkdir build
cd build

cmake -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/torchvision" \
      -DCMAKE_BUILD_TYPE=Release ..

cmake --build . --config Release
```

## Step 3: Run Inference in C++

### With Custom Ops

```bash
# Run inference with custom ops library
./build/torchscript_mask_rcnn_lazy \
    output/model.ts \
    input_image.jpg \
    tracing \
    ./build/libdetectron2_ops.so
```

### Without Custom Ops

If you built without custom ops or your model doesn't need them:

```bash
# Run inference using the original executable
./build/torchscript_mask_rcnn output/model.ts input_image.jpg tracing
```

## Example: Complete Workflow for mask_rcnn_R_50_FPN_100ep_LSJ

```bash
# 1. Export the model (assuming you have trained weights)
python tools/deploy/export_model_lazy.py \
    --config-file configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
    --output ./export_output \
    --export-method tracing \
    --format torchscript \
    --sample-image demo/sample.jpg \
    train.init_checkpoint=./model_weights.pth \
    train.device=cuda

# 2. Build the C++ inference code with custom ops
cd tools/deploy
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$TORCH_INSTALL_PATH;$TORCHVISION_PATH" \
      -DCMAKE_BUILD_TYPE=Release \
      -C ../CMakeLists_with_ops.txt ..
make -j4

# 3. Run inference
./torchscript_mask_rcnn_lazy \
    ../../../export_output/model.ts \
    ../../../demo/sample.jpg \
    tracing \
    ./libdetectron2_ops.so
```

## Custom Ops Included

The `libdetectron2_ops.so` library includes the following custom operations from `detectron2/layers/csrc/`:

- Deformable Convolution (`deformable/`)
- Rotated ROI Align (`ROIAlignRotated/`)
- Rotated NMS (`nms_rotated/`)
- Rotated Box IoU (`box_iou_rotated/`)
- COCO Evaluation utilities (`cocoeval/`)

These ops are automatically registered in the `detectron2` namespace when the library is loaded.

## Troubleshooting

### Model uses custom ops but they're not loading

Make sure you're passing the correct path to `libdetectron2_ops.so` as the 4th argument to the inference executable.

### CMake can't find LibTorch or TorchVision

Set the CMAKE_PREFIX_PATH to include both:
```bash
cmake -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/torchvision" ..
```

### Model exported on GPU but running on CPU (or vice versa)

Re-export the model with the correct device setting:
```bash
python tools/deploy/export_model_lazy.py ... train.device=cpu
```

### Undefined symbols when loading custom ops

Ensure that the LibTorch version used to build the custom ops matches the version used in the Python export.

## Differences from Original export_model.py

The new `export_model_lazy.py` differs from the original in:

1. **Config Loading**: Uses `LazyConfig.load()` instead of `get_cfg()`
2. **Model Instantiation**: Uses `instantiate(cfg.model)` instead of `build_model(cfg)`
3. **Dataloader**: Uses `instantiate(cfg.dataloader.test)` for lazy configs
4. **Simplified**: Removed caffe2_tracing support (not typically used with new models)

## Performance Tips

- Use `export-method=tracing` for better performance in most cases
- Build in Release mode (`-DCMAKE_BUILD_TYPE=Release`) for production
- Use CUDA/GPU for faster inference when available
- The benchmark runs multiple iterations to measure average latency

## See Also

- [Original deployment README](./README.md) for YAML config models
- [Detectron2 deployment tutorial](https://detectron2.readthedocs.io/tutorials/deployment.html)
- [LazyConfig documentation](../../docs/tutorials/lazyconfigs.md)
