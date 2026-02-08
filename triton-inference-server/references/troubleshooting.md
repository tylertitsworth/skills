# Triton Inference Server Troubleshooting

## Model Loading Failures

### "failed to load model"

Check server logs (`tritonserver` stdout):

```bash
# Increase verbosity
tritonserver --model-repository=/models --log-verbose=1
```

**Common causes:**

1. **Wrong platform/backend**: Ensure `config.pbtxt` platform matches the model file:
   - ONNX → `platform: "onnxruntime_onnx"`
   - TensorRT → `platform: "tensorrt_plan"`
   - PyTorch → `platform: "pytorch_libtorch"`
   - Python → `backend: "python"`

2. **Model file not found**: Must follow naming convention:
   ```
   model_name/1/model.onnx   # ONNX
   model_name/1/model.plan   # TensorRT
   model_name/1/model.pt     # PyTorch
   model_name/1/model.py     # Python
   ```

3. **Missing dependencies**: Python backend models need deps in the container:
   ```bash
   # Add to Dockerfile or install at runtime
   pip install transformers torchvision
   ```

### "no version available"

Model directory must contain at least one version subdirectory:
```
my_model/
├── config.pbtxt
└── 1/              # ← must exist with model file inside
    └── model.onnx
```

### TensorRT plan incompatible

TensorRT plans are GPU-specific. A plan built on A100 won't load on T4:
```bash
# Rebuild for target GPU
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16
```

## Shape Mismatches

### "input ... unexpected shape"

Input tensor shape doesn't match `config.pbtxt`:

```protobuf
# config.pbtxt says:
input [ { name: "input", dims: [ 3, 224, 224 ] } ]

# But client sends shape [224, 224, 3] — wrong order!
```

**Fix**: Ensure client tensor shape matches config dims. With `max_batch_size > 0`, don't include the batch dimension in config (Triton adds it):
```protobuf
max_batch_size: 32
input [ { name: "input", dims: [ 3, 224, 224 ] } ]
# Client should send shape: [batch, 3, 224, 224]
```

### Variable-length inputs

Use `-1` for variable dimensions:
```protobuf
input [ { name: "input_ids", dims: [ -1 ] } ]  # variable sequence length
```

### Data type mismatch

```protobuf
# Config says FP32 but client sends FP16
input [ { name: "input", data_type: TYPE_FP32, dims: [...] } ]
```

Ensure client matches:
```python
image = image.astype(np.float32)  # match TYPE_FP32
```

## Performance Tuning

### Low throughput

1. **Enable dynamic batching**:
   ```protobuf
   dynamic_batching {
     preferred_batch_size: [ 8, 16, 32 ]
     max_queue_delay_microseconds: 200
   }
   ```

2. **Increase model instances**:
   ```protobuf
   instance_group [ { count: 2, kind: KIND_GPU } ]
   ```

3. **Use TensorRT** instead of ONNX for GPU models — typically 2-5x faster

4. **Use FP16/INT8 quantization** when converting to TensorRT:
   ```bash
   trtexec --onnx=model.onnx --saveEngine=model.plan --fp16
   ```

### High latency

1. **Reduce `max_queue_delay_microseconds`** (smaller batches, lower latency):
   ```protobuf
   dynamic_batching { max_queue_delay_microseconds: 50 }
   ```

2. **Pin model to specific GPU** to avoid contention:
   ```protobuf
   instance_group [ { count: 1, kind: KIND_GPU, gpus: [ 0 ] } ]
   ```

3. **Use gRPC** instead of HTTP for lower overhead

### Monitoring performance

```bash
# Prometheus metrics
curl localhost:8002/metrics

# Key metrics:
# nv_inference_request_success    — successful requests
# nv_inference_request_failure    — failed requests
# nv_inference_queue_duration_us  — time in queue (batching delay)
# nv_inference_compute_infer_duration_us — inference compute time
# nv_gpu_utilization              — GPU utilization
```

## Python Backend Issues

### "ModuleNotFoundError" in model.py

Dependencies missing from the Triton container:

```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.12-py3
RUN pip install transformers pillow scikit-learn
```

### Python model crashes silently

Add error handling and logging:
```python
class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # ... your logic
                responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))
        return responses
```

### Python backend is slow

1. Minimize Python overhead — do heavy compute in NumPy/PyTorch, not pure Python
2. Use `initialize()` to load models/resources once, not per-request
3. Consider converting to ONNX/TensorRT if possible

## Ensemble Issues

### "ensemble step failed"

Check that intermediate tensor names match between steps:
```protobuf
# Step 1 output_map key must match step 2 input_map value
step [ {
    model_name: "step1"
    output_map { key: "output", value: "intermediate" }
}, {
    model_name: "step2"
    input_map { key: "input", value: "intermediate" }  # must match "intermediate"
} ]
```

### Shape mismatch between ensemble steps

Ensure output shape of step N matches input shape of step N+1. Check each model's config independently first.

## Kubernetes Issues

### Triton OOMKilled

```yaml
resources:
  requests:
    memory: "16Gi"
  limits:
    memory: "32Gi"
```

Also mount `/dev/shm` for shared memory:
```yaml
volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 4Gi
```

### Models not loading from PVC

Ensure the PVC is mounted and contains the correct directory structure. Check permissions:
```bash
kubectl exec -it triton-pod -- ls -la /models/
```
