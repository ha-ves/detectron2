// Copyright (c) Facebook, Inc. and its affiliates.
// @lint-ignore-every CLANGTIDY
// This is an example code that demonstrates how to run inference
// with a torchscript format Mask R-CNN model exported by ./export_model_lazy.py
// using export method=tracing or scripting, with custom ops support.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>

// only needed for export_method=tracing
#include <torchvision/vision.h> // @oss-only
// @fb-only: #include <torchvision/csrc/vision.h>

using namespace std;

c10::IValue get_tracing_inputs(cv::Mat& img, c10::Device device) {
  const int height = img.rows;
  const int width = img.cols;
  const int channels = 3;

  auto input =
      torch::from_blob(img.data, {height, width, channels}, torch::kUInt8);
  // HWC to CHW
  input = input.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();
  return input;
}

// create a Tuple[Dict[str, Tensor]] which is the input type of scripted model
c10::IValue get_scripting_inputs(cv::Mat& img, c10::Device device) {
  const int height = img.rows;
  const int width = img.cols;
  const int channels = 3;

  auto img_tensor =
      torch::from_blob(img.data, {height, width, channels}, torch::kUInt8);
  // HWC to CHW
  img_tensor =
      img_tensor.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();
  auto dic = c10::Dict<std::string, torch::Tensor>();
  dic.insert("image", img_tensor);
  return std::make_tuple(dic);
}

c10::IValue
get_inputs(std::string export_method, cv::Mat& img, c10::Device device) {
  // Given an image, create inputs in the format required by the model.
  if (export_method == "tracing")
    return get_tracing_inputs(img, device);
  if (export_method == "scripting")
    return get_scripting_inputs(img, device);
  abort();
}

struct MaskRCNNOutputs {
  at::Tensor pred_boxes, pred_classes, pred_masks, scores;
  int num_instances() const {
    return pred_boxes.sizes()[0];
  }
};

MaskRCNNOutputs get_outputs(std::string export_method, c10::IValue outputs) {
  // Given outputs of the model, extract tensors from it to turn into a
  // common MaskRCNNOutputs format.
  if (export_method == "tracing") {
    auto out_tuple = outputs.toTuple()->elements();
    // They are ordered alphabetically by their field name in Instances
    return MaskRCNNOutputs{
        out_tuple[0].toTensor(),
        out_tuple[1].toTensor(),
        out_tuple[2].toTensor(),
        out_tuple[3].toTensor()};
  }
  if (export_method == "scripting") {
    // With the ScriptableAdapter defined in export_model_lazy.py, the output is
    // List[Dict[str, Any]].
    auto out_dict = outputs.toList().get(0).toGenericDict();
    return MaskRCNNOutputs{
        out_dict.at("pred_boxes").toTensor(),
        out_dict.at("pred_classes").toTensor(),
        out_dict.at("pred_masks").toTensor(),
        out_dict.at("scores").toTensor()};
  }
  abort();
}

int main(int argc, const char* argv[]) {
  if (argc < 4 || argc > 5) {
    cerr << R"xx(
Usage:
   ./torchscript_mask_rcnn_lazy model.ts input.jpg EXPORT_METHOD [custom_ops.so]

   EXPORT_METHOD can be "tracing" or "scripting".
   custom_ops.so is optional path to detectron2 custom ops library
)xx";
    return 1;
  }
  std::string model_file = argv[1];
  std::string image_file = argv[2];
  std::string export_method = argv[3];
  assert(export_method == "tracing" || export_method == "scripting");

  // Load custom ops if provided
  if (argc == 5) {
    std::string custom_ops_path = argv[4];
    cout << "Loading custom ops from: " << custom_ops_path << endl;
    try {
      torch::jit::load(custom_ops_path);
      cout << "Custom ops loaded successfully" << endl;
    } catch (const c10::Error& e) {
      cerr << "Error loading custom ops: " << e.what() << endl;
      cerr << "Continuing without custom ops..." << endl;
    }
  }

  torch::jit::FusionStrategy strat = {{torch::jit::FusionBehavior::DYNAMIC, 1}};
  torch::jit::setFusionStrategy(strat);
  torch::autograd::AutoGradMode guard(false);
  
  cout << "Loading model from: " << model_file << endl;
  auto module = torch::jit::load(model_file);
  cout << "Model loaded successfully" << endl;

  assert(module.buffers().size() > 0);
  // Assume that the entire model is on the same device.
  // We just put input to this device.
  auto device = (*begin(module.buffers())).device();
  cout << "Model device: " << device << endl;

  cv::Mat input_img = cv::imread(image_file, cv::IMREAD_COLOR);
  if (input_img.empty()) {
    cerr << "Error: Could not read image from " << image_file << endl;
    return 1;
  }
  cout << "Image loaded: " << input_img.cols << "x" << input_img.rows << endl;

  auto inputs = get_inputs(export_method, input_img, device);

  // Run the network
  cout << "Running inference..." << endl;
  auto output = module.forward({inputs});
  if (device.is_cuda())
    c10::cuda::getCurrentCUDAStream().synchronize();

  // run 3 more times to benchmark
  int N_benchmark = 3, N_warmup = 1;
  auto start_time = chrono::high_resolution_clock::now();
  for (int i = 0; i < N_benchmark + N_warmup; ++i) {
    if (i == N_warmup)
      start_time = chrono::high_resolution_clock::now();
    output = module.forward({inputs});
    if (device.is_cuda())
      c10::cuda::getCurrentCUDAStream().synchronize();
  }
  auto end_time = chrono::high_resolution_clock::now();
  auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
                .count();
  cout << "Latency (should vary with different inputs): "
       << ms * 1.0 / 1e6 / N_benchmark << " seconds" << endl;

  // Parse Mask R-CNN outputs
  auto rcnn_outputs = get_outputs(export_method, output);
  cout << "Number of detected objects: " << rcnn_outputs.num_instances()
       << endl;

  cout << "pred_boxes: " << rcnn_outputs.pred_boxes.toString() << " "
       << rcnn_outputs.pred_boxes.sizes() << endl;
  cout << "scores: " << rcnn_outputs.scores.toString() << " "
       << rcnn_outputs.scores.sizes() << endl;
  cout << "pred_classes: " << rcnn_outputs.pred_classes.toString() << " "
       << rcnn_outputs.pred_classes.sizes() << endl;
  cout << "pred_masks: " << rcnn_outputs.pred_masks.toString() << " "
       << rcnn_outputs.pred_masks.sizes() << endl;

  cout << rcnn_outputs.pred_boxes << endl;
  
  cout << "Inference completed successfully!" << endl;
  return 0;
}
