#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Export script for models using LazyConfig.

This script supports exporting detectron2 models defined with LazyConfig
to TorchScript format for deployment with LibTorch.

The script accepts both .py config files and .yaml config files:
- Use .py files for base configs (e.g., configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py)
- Use .yaml files for trained model configs (e.g., training_output/config.yaml)

For trained models, it's recommended to use the config.yaml saved in the training
output directory, as it contains the exact configuration used during training
(including any command-line overrides).
"""
import argparse
import os
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    STABLE_ONNX_OPSET_VERSION,
    TracingAdapter,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger


def setup_lazy_cfg(args):
    """Load a lazy config file."""
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    return cfg


# experimental. API not yet final
def export_scripting(torch_model):
    assert TORCH_VERSION >= (1, 8)
    fields = {
        "proposal_boxes": Boxes,
        "objectness_logits": Tensor,
        "pred_boxes": Boxes,
        "scores": Tensor,
        "pred_classes": Tensor,
        "pred_masks": Tensor,
        "pred_keypoints": torch.Tensor,
        "pred_keypoint_heatmaps": torch.Tensor,
    }
    assert args.format == "torchscript", "Scripting only supports torchscript format."

    class ScriptableAdapterBase(nn.Module):
        # Use this adapter to workaround https://github.com/pytorch/pytorch/issues/46944
        # by not retuning instances but dicts. Otherwise the exported model is not deployable
        def __init__(self):
            super().__init__()
            self.model = torch_model
            self.eval()

    if isinstance(torch_model, GeneralizedRCNN):

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model.inference(inputs, do_postprocess=False)
                return [i.get_fields() for i in instances]

    else:

        class ScriptableAdapter(ScriptableAdapterBase):
            def forward(self, inputs: Tuple[Dict[str, torch.Tensor]]) -> List[Dict[str, Tensor]]:
                instances = self.model(inputs)
                return [i.get_fields() for i in instances]

    ts_model = scripting_with_instances(ScriptableAdapter(), fields)
    with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, args.output)
    # TODO inference in Python now missing postprocessing glue code
    return None


# experimental. API not yet final
def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    if args.format == "torchscript":
        ts_model = torch.jit.trace(traceable_model, (image,))
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)
    elif args.format == "onnx":
        with PathManager.open(os.path.join(args.output, "model.onnx"), "wb") as f:
            torch.onnx.export(traceable_model, (image,), f, opset_version=STABLE_ONNX_OPSET_VERSION)
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    if args.format != "torchscript":
        return None
    if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
        return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper


def get_sample_inputs(args, cfg):
    """Get sample inputs for export."""
    if args.sample_image is None:
        # get a first batch from dataset
        # For lazy configs, we need to instantiate the dataloader
        data_loader = instantiate(cfg.dataloader.test)
        first_batch = next(iter(data_loader))
        return first_batch
    else:
        # get a sample data
        original_image = detection_utils.read_image(args.sample_image, format="BGR")
        # Do same preprocessing as DefaultPredictor
        # For lazy configs, use typical values
        min_size = 800
        max_size = 1333
        aug = T.ResizeShortestEdge([min_size, min_size], max_size)
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs


def main() -> None:
    global logger, cfg, args
    parser = argparse.ArgumentParser(description="Export a model for deployment using LazyConfig.")
    parser.add_argument(
        "--format",
        choices=["onnx", "torchscript"],
        help="output format",
        default="torchscript",
    )
    parser.add_argument(
        "--export-method",
        choices=["tracing", "scripting"],
        help="Method to export models",
        default="tracing",
    )
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file (.py or .yaml). "
        "For trained models, use the config.yaml saved in the training output directory.",
    )
    parser.add_argument("--sample-image", default=None, type=str, help="sample image for input")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)
    # Disable re-specialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)

    cfg = setup_lazy_cfg(args)

    # create a torch model
    torch_model = instantiate(cfg.model)
    torch_model.eval()
    
    # Load weights if specified
    if hasattr(cfg.train, "init_checkpoint") and cfg.train.init_checkpoint:
        DetectionCheckpointer(torch_model).load(cfg.train.init_checkpoint)
        logger.info(f"Loaded checkpoint from {cfg.train.init_checkpoint}")

    # Move to device
    device = cfg.train.device if hasattr(cfg.train, "device") else "cpu"
    torch_model.to(device)
    logger.info(f"Model moved to {device}")

    # convert and save model
    if args.export_method == "scripting":
        exported_model = export_scripting(torch_model)
    elif args.export_method == "tracing":
        sample_inputs = get_sample_inputs(args, cfg)
        exported_model = export_tracing(torch_model, sample_inputs)

    # run evaluation with the converted model
    if args.run_eval:
        assert exported_model is not None, (
            "Python inference is not yet implemented for "
            f"export_method={args.export_method}, format={args.format}."
        )
        logger.info("Running evaluation ... this takes a long time if you export to CPU.")
        # For lazy configs, we need to instantiate the test dataloader
        data_loader = instantiate(cfg.dataloader.test)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        evaluator = instantiate(cfg.dataloader.evaluator)
        metrics = inference_on_dataset(exported_model, data_loader, evaluator)
        print_csv_format(metrics)
    logger.info("Success.")


if __name__ == "__main__":
    main()  # pragma: no cover
