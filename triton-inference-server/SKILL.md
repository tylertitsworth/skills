---
name: triton-inference-server
description: >
  Serve ML/DL models with NVIDIA Triton Inference Server — multi-framework, multi-model
  serving with dynamic batching and model ensembles. Use when: (1) Setting up a model
  repository with config.pbtxt, (2) Serving ONNX, TensorRT, PyTorch, TensorFlow, or Python
  models, (3) Configuring dynamic batching for throughput, (4) Building model ensembles
  (pre/post-processing pipelines), (5) Deploying on Kubernetes with autoscaling,
  (6) Configuring model instances and GPU assignment, (7) Using the HTTP/gRPC client,
  (8) Debugging model loading, inference errors, or performance issues. Note: for LLM
  serving, prefer vLLM. This skill focuses on traditional ML/DL model serving.
---

# Triton Inference Server

NVIDIA Triton is a multi-framework inference server supporting ONNX, TensorRT, PyTorch, TensorFlow, Python, and more. Optimized for GPU inference with dynamic batching, model ensembles, and concurrent model execution.

> **For LLM serving**: Use the `vllm` skill instead. This skill covers traditional ML/DL model serving (vision, NLP classification, embeddings, custom pipelines).

## Quick Start

Container image: `nvcr.io/nvidia/tritonserver:24.12-py3`
Command: `tritonserver --model-repository=/models`

**Ports:**
- `8000` — HTTP/REST
- `8001` — gRPC
- `8002` — Prometheus metrics

## Model Repository

The model repository is a filesystem layout Triton reads on startup:

```
model_repository/
├── image_classifier/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
├── text_embedder/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt
└── preprocessor/
    ├── config.pbtxt
    └── 1/
        └── model.py
```

**Structure per model:**
```
<model-name>/
├── config.pbtxt          # model configuration (required unless auto-complete)
├── 1/                    # version 1
│   └── model.<ext>       # model file
├── 2/                    # version 2 (optional)
│   └── model.<ext>
└── labels.txt            # optional labels file
```

### Model File Names by Backend

| Backend | File Name | Extension |
|---------|-----------|-----------|
| ONNX Runtime | `model.onnx` | `.onnx` |
| TensorRT | `model.plan` | `.plan` |
| PyTorch (TorchScript) | `model.pt` | `.pt` |
| TensorFlow SavedModel | `model.savedmodel/` | directory |
| Python | `model.py` | `.py` |
| OpenVINO | `model.xml` + `model.bin` | `.xml` |

## Model Configuration (config.pbtxt)

### ONNX Model

```protobuf
name: "image_classifier"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}
```

### TensorRT Model

```protobuf
name: "detection_model"
platform: "tensorrt_plan"
max_batch_size: 32

input [
  {
    name: "images"
    data_type: TYPE_FP16
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "detections"
    data_type: TYPE_FP32
    dims: [ 100, 6 ]  # [x1, y1, x2, y2, score, class]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

### PyTorch (TorchScript) Model

```protobuf
name: "text_embedder"
platform: "pytorch_libtorch"
max_batch_size: 128

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]  # variable length
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]
output [
  {
    name: "embeddings"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]
```

### Python Backend

For custom pre/post-processing or models not supported by other backends:

```protobuf
name: "preprocessor"
backend: "python"
max_batch_size: 64

input [
  {
    name: "raw_image"
    data_type: TYPE_UINT8
    dims: [ -1, -1, 3 ]  # variable HxW
  }
]
output [
  {
    name: "processed_image"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
```

**Python model implementation (`model.py`):**

```python
import triton_python_backend_utils as pb_utils
import numpy as np
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        # Load any resources here

    def execute(self, requests):
        responses = []
        for request in requests:
            raw_image = pb_utils.get_input_tensor_by_name(request, "raw_image").as_numpy()

            # Custom preprocessing
            processed = self.preprocess(raw_image)

            output_tensor = pb_utils.Tensor("processed_image", processed.astype(np.float32))
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        return responses

    def preprocess(self, image):
        # Resize, normalize, etc.
        ...

    def finalize(self):
        pass
```

## Dynamic Batching

Accumulates individual requests into batches for GPU efficiency:

```protobuf
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32, 64 ]
  max_queue_delay_microseconds: 100  # max wait time to fill a batch
}
```

**Tuning:**
- Lower `max_queue_delay_microseconds` → lower latency, smaller batches
- Higher → better throughput, higher latency
- `preferred_batch_size` should match sizes your model runs efficiently at

## Instance Groups

Control how many model copies run and on which GPUs:

```protobuf
# 2 instances on GPU 0
instance_group [
  { count: 2, kind: KIND_GPU, gpus: [ 0 ] }
]

# 1 instance each on GPU 0 and GPU 1
instance_group [
  { count: 1, kind: KIND_GPU, gpus: [ 0 ] },
  { count: 1, kind: KIND_GPU, gpus: [ 1 ] }
]

# CPU inference
instance_group [
  { count: 4, kind: KIND_CPU }
]
```

## Model Ensembles

Chain models together (e.g., preprocessing → inference → postprocessing):

```protobuf
name: "image_pipeline"
platform: "ensemble"
max_batch_size: 64

input [
  { name: "raw_image", data_type: TYPE_UINT8, dims: [ -1, -1, 3 ] }
]
output [
  { name: "class_label", data_type: TYPE_STRING, dims: [ 1 ] }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocessor"
      model_version: -1
      input_map { key: "raw_image", value: "raw_image" }
      output_map { key: "processed_image", value: "preprocessed" }
    },
    {
      model_name: "image_classifier"
      model_version: -1
      input_map { key: "input", value: "preprocessed" }
      output_map { key: "output", value: "logits" }
    },
    {
      model_name: "postprocessor"
      model_version: -1
      input_map { key: "logits", value: "logits" }
      output_map { key: "class_label", value: "class_label" }
    }
  ]
}
```

## Client Usage

### Python Client

```bash
pip install tritonclient[all]
```

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# Check server health
assert client.is_server_live()
assert client.is_model_ready("image_classifier")

# Prepare input
image = np.random.rand(1, 3, 224, 224).astype(np.float32)
inputs = [httpclient.InferInput("input", image.shape, "FP32")]
inputs[0].set_data_from_numpy(image)

outputs = [httpclient.InferRequestedOutput("output")]

# Inference
result = client.infer("image_classifier", inputs, outputs=outputs)
predictions = result.as_numpy("output")
print(f"Top class: {predictions.argmax()}")
```

### gRPC Client

```python
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="localhost:8001")
# Same API pattern as HTTP client
```

### Async / Streaming

```python
import tritonclient.http as httpclient
from functools import partial

def callback(result, error):
    if error:
        print(f"Error: {error}")
    else:
        print(f"Result: {result.as_numpy('output').shape}")

client.async_infer("image_classifier", inputs, outputs=outputs,
                    callback=partial(callback))
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: triton
  template:
    metadata:
      labels:
        app: triton
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8002"
    spec:
      containers:
        - name: triton
          image: nvcr.io/nvidia/tritonserver:24.12-py3
          args:
            - tritonserver
            - --model-repository=/models
            - --strict-model-config=false
            - --log-verbose=0
          ports:
            - containerPort: 8000  # HTTP
            - containerPort: 8001  # gRPC
            - containerPort: 8002  # Metrics
          resources:
            limits:
              nvidia.com/gpu: "1"
            requests:
              cpu: "4"
              memory: "16Gi"
          readinessProbe:
            httpGet:
              path: /v2/health/ready
              port: 8000
            initialDelaySeconds: 30
          livenessProbe:
            httpGet:
              path: /v2/health/live
              port: 8000
          volumeMounts:
            - name: model-repo
              mountPath: /models
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: model-repo
          persistentVolumeClaim:
            claimName: model-repo-pvc
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 4Gi
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

## Model Management

```bash
# Load/unload models at runtime (if --model-control-mode=explicit)
curl -X POST localhost:8000/v2/repository/models/my_model/load
curl -X POST localhost:8000/v2/repository/models/my_model/unload

# Get model metadata
curl localhost:8000/v2/models/image_classifier

# Model statistics
curl localhost:8000/v2/models/image_classifier/stats
```

## Debugging

See `references/troubleshooting.md` for:
- Model loading failures
- Shape mismatches and data type errors
- Performance tuning (batch size, instances, concurrency)
- Python backend issues
- Ensemble pipeline debugging

## Reference

- [Triton docs](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Triton GitHub](https://github.com/triton-inference-server/server)
- [Model Analyzer](https://github.com/triton-inference-server/model_analyzer)
- `references/troubleshooting.md` — common errors and fixes
