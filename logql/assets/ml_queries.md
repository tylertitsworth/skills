# LogQL Queries for ML Infrastructure

Ready-to-use LogQL queries for monitoring ML training and inference workloads.

## Training Job Monitoring

### Training loss extraction (PyTorch/HF Trainer format)
```logql
{namespace="ml-training", app="training-job"}
  |= "loss"
  | json
  | unwrap loss
  | avg_over_time(loss[5m]) by (pod)
```

### OOM kills in training pods
```logql
{namespace="ml-training"}
  |~ "(?i)(out of memory|OOM|CUDA error|RuntimeError.*memory)"
  | label_format severity="critical"
```

### NCCL errors across training jobs
```logql
{namespace="ml-training"}
  |~ "(?i)(NCCL|nccl).*(error|timeout|failed)"
  | line_format "{{.pod}}: {{__line__}}"
```

### Training throughput (tokens/sec or samples/sec)
```logql
{namespace="ml-training", app="megatron"}
  |= "throughput"
  | pattern "<_> throughput=<throughput> <_>"
  | unwrap throughput
  | avg_over_time(throughput[5m]) by (pod)
```

### Checkpoint save events
```logql
count_over_time(
  {namespace="ml-training"} |= "Saving checkpoint" [1h]
) by (pod)
```

## Inference Monitoring

### vLLM request errors
```logql
{app="vllm-server"}
  |~ "(?i)(error|exception|failed)"
  != "health"
  | rate([5m])
```

### vLLM request latency from logs
```logql
{app="vllm-server"}
  |= "Processed request"
  | pattern "<_> Processed request <request_id>: prompt: <prompt_tokens> tokens, generation: <gen_tokens> tokens, <latency> s"
  | unwrap latency
  | quantile_over_time(0.95, latency[5m])
```

### Ollama model load events
```logql
{app="ollama"}
  |= "loading model"
  | rate([1h])
```

## KubeRay Cluster Monitoring

### Ray head GCS failures
```logql
{namespace=~"ray-.*", container="ray-head"}
  |~ "(?i)(gcs.*fail|gcs.*error|gcs.*restart)"
```

### Ray worker connection issues
```logql
{namespace=~"ray-.*", container=~"ray-worker.*"}
  |~ "(?i)(connection refused|cannot connect|timeout.*head)"
```

### Ray job submission and completion
```logql
{namespace=~"ray-.*"}
  |~ "(Job submitted|Job .* succeeded|Job .* failed)"
  | pattern "<_> Job <job_id> <status>"
```

## Kueue Queue Monitoring

### Workload admission events
```logql
{namespace="kueue-system", container="kueue-controller"}
  |= "admitted"
  | rate([5m])
```

### Workload preemption events
```logql
{namespace="kueue-system", container="kueue-controller"}
  |~ "(?i)preempt"
```

### Queue quota exceeded
```logql
{namespace="kueue-system", container="kueue-controller"}
  |~ "(?i)(quota exceeded|insufficient quota|cannot admit)"
```

## Flyte Workflow Monitoring

### Failed Flyte tasks
```logql
{namespace="flyte", container="flytepropeller"}
  |~ "(?i)(task.*failed|execution.*error)"
  | rate([5m])
```

### Flyte task retries
```logql
count_over_time(
  {namespace="flyte", container="flytepropeller"} |= "retrying" [1h]
) by (workflow)
```
