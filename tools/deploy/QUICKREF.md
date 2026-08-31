# Quick Reference: LibTorch Export Files

This is a quick reference for the new LibTorch export functionality for LazyConfig models.

## New Files

### 1. export_model_lazy.py
**Purpose**: Export LazyConfig models (like mask_rcnn_R_50_FPN_100ep_LSJ.py) to TorchScript

**Usage**:
```bash
# Using base config file:
python tools/deploy/export_model_lazy.py \
    --config-file configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py \
    --output ./output \
    --export-method tracing \
    --sample-image image.jpg \
    train.init_checkpoint=model.pth

# Using saved config.yaml from training (recommended for trained models):
python tools/deploy/export_model_lazy.py \
    --config-file /path/to/training_output/config.yaml \
    --output ./output \
    --export-method tracing \
    --sample-image image.jpg \
    train.init_checkpoint=/path/to/training_output/model_final.pth
```

**Key Features**:
- Supports `--export-method tracing` and `--export-method scripting`
- Works with LazyConfig Python files (.py) and YAML files (.yaml)
- **Recommended**: Use `config.yaml` from training output for trained models
- Can override config values via command line (e.g., `train.device=cuda`)

### 2. CMakeLists_with_ops.txt
**Purpose**: CMake configuration to build custom ops library and inference executable

**What it builds**:
- `libdetectron2_ops.so` - Shared library containing all custom ops from layers/csrc/
- `torchscript_mask_rcnn` - Inference executable (links with custom ops)

**Custom Ops Included**:
- Deformable Convolution
- Rotated ROI Align
- Rotated NMS
- Rotated Box IoU
- COCO Evaluation utilities

### 3. torchscript_mask_rcnn_lazy.cpp
**Purpose**: C++ inference example with custom ops support

**Usage**:
```bash
./torchscript_mask_rcnn_lazy model.ts image.jpg tracing [libdetectron2_ops.so]
```

**Features**:
- Loads custom ops library (optional 4th argument)
- Supports tracing and scripting export methods
- Includes benchmarking code
- Detailed error messages

### 4. README_LAZYCONFIG.md
**Purpose**: Complete documentation for the export workflow

**Contents**:
- Step-by-step export instructions
- Build instructions for custom ops
- Example workflows
- Troubleshooting guide

### 5. example_export_workflow.sh
**Purpose**: Example shell script showing complete workflow

**Usage**:
```bash
./tools/deploy/example_export_workflow.sh [sample_image.jpg] [model_weights.pth]
```

## Typical Workflow

```
┌─────────────────────────────────────┐
│ 1. Train Model with LazyConfig      │
│    (or use pretrained weights)      │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 2. Export to TorchScript             │
│    python export_model_lazy.py ...   │
│    → Produces model.ts               │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 3. Build C++ Code with CMake        │
│    cmake -C CMakeLists_with_ops.txt │
│    → libdetectron2_ops.so           │
│    → torchscript_mask_rcnn_lazy     │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 4. Run Inference in C++             │
│    ./torchscript_mask_rcnn_lazy ... │
└─────────────────────────────────────┘
```

## Key Differences from Original export_model.py

| Feature | export_model.py | export_model_lazy.py |
|---------|----------------|---------------------|
| Config Format | YAML | Python (LazyConfig) |
| Config Loading | `get_cfg()` | `LazyConfig.load()` |
| Model Build | `build_model(cfg)` | `instantiate(cfg.model)` |
| Caffe2 Support | Yes | No (legacy) |
| Target Models | COCO configs | new_baselines configs |

## File Locations

```
tools/deploy/
├── export_model.py              # Original (for YAML configs)
├── export_model_lazy.py         # NEW (for LazyConfig)
├── CMakeLists.txt               # Original (no custom ops)
├── CMakeLists_with_ops.txt      # NEW (builds custom ops)
├── torchscript_mask_rcnn.cpp    # Original C++ example
├── torchscript_mask_rcnn_lazy.cpp  # NEW (with custom ops)
├── README.md                    # Original documentation
├── README_LAZYCONFIG.md         # NEW (LazyConfig guide)
└── example_export_workflow.sh   # NEW (example script)
```

## Common Issues

**Import Error**: If export fails with import errors, make sure detectron2 is installed:
```bash
pip install -e .
```

**Custom Ops Not Found**: When running C++ inference, ensure you pass the path to libdetectron2_ops.so

**LibTorch Version Mismatch**: Use the same PyTorch/LibTorch version for export and inference

## See Also

- `README_LAZYCONFIG.md` - Full documentation
- `../../configs/new_baselines/` - LazyConfig model configs
- `../../detectron2/layers/csrc/` - Custom ops source code
