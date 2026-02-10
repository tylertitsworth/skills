---
name: kuberay
description: >
  Deploy and manage Ray clusters on Kubernetes using the KubeRay operator. Use when:
  (1) Deploying RayCluster resources (head/worker config, GPU scheduling, autoscaling,
  multiple worker groups, heterogeneous hardware), (2) Submitting RayJob workloads
  (submission modes, runtime envs, retries, existing clusters), (3) Managing RayService
  for model serving (zero-downtime upgrades, autoscaling, HA), (4) Configuring operator
  Helm chart (RBAC, namespace scoping, feature gates, leader election), (5) GPU configuration
  (multi-tenancy, NVIDIA_VISIBLE_DEVICES, tolerations, node selectors), (6) Autoscaling
  (in-tree autoscaler, autoscaler v2, idle timeout, upscaling modes), (7) GCS fault tolerance
  with Redis for head pod recovery, (8) TLS authentication between Ray nodes,
  (9) Observability (dashboard, Prometheus, status conditions, events),
  (10) Integrating with Kueue, Ingress controllers, Volcano, MCAD,
  (11) Debugging KubeRay issues (init containers, GCS failures, autoscaler, version compat).
---

# KubeRay

Kubernetes operator for Ray. Provides CRDs for running distributed Ray workloads natively on K8s.

**Docs:** https://docs.ray.io/en/latest/cluster/kubernetes/index.html  
**GitHub:** https://github.com/ray-project/kuberay  
**Operator:** v1.5.1 | **Ray:** 2.53.0 | **API:** `ray.io/v1`

## CRDs

| CRD | Purpose | Lifecycle |
|---|---|---|
| **RayCluster** | Long-running Ray cluster (head + worker groups) | Manual or autoscaled |
| **RayJob** | One-shot: creates cluster, submits job, optionally cleans up | Ephemeral |
| **RayService** | Ray Serve with zero-downtime upgrades | Long-running serving |

**When to use which:**
- **RayJob** for batch/training — new cluster per job, auto-cleanup, cost-efficient
- **RayCluster** for interactive/dev — persistent, no startup latency per job
- **RayService** for model serving — managed upgrades, HA, traffic routing

## Installation

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 1.5.1 \
  --namespace kuberay-system --create-namespace
```

### Key Helm Values

```yaml
image:
  repository: quay.io/kuberay/operator
  tag: v1.5.1

# Namespace scoping
watchNamespace: []                    # empty = all namespaces
singleNamespaceInstall: false         # true = Role instead of ClusterRole

# RBAC
rbacEnable: true
crNamespacedRbacEnable: true          # false for GitOps tools like ArgoCD

# Feature gates
featureGates:
- name: RayClusterStatusConditions
  enabled: true

# Operator tuning
reconcileConcurrency: 1               # increase for many CRs
batchScheduler: ""                     # "volcano" or "yunikorn"

# Leader election (for HA)
leaderElection:
  enabled: true
```

Verify: `kubectl get pods -n kuberay-system`

## RayCluster

### GPU Cluster Example

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: gpu-cluster
spec:
  rayVersion: "2.53.0"
  enableInTreeAutoscaling: true
  autoscalerOptions:
    upscalingMode: Default
    idleTimeoutSeconds: 60
    resources:
      limits:
        cpu: "1"
        memory: 1Gi
  headGroupSpec:
    serviceType: ClusterIP             # ClusterIP | NodePort | LoadBalancer
    rayStartParams:
      dashboard-host: "0.0.0.0"
      num-cpus: "0"                    # prevent workloads on head
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray-ml:2.53.0-gpu
          resources:
            limits:
              cpu: "4"
              memory: 16Gi
            requests:
              cpu: "4"
              memory: 16Gi
          ports:
          - containerPort: 6379
            name: gcs
          - containerPort: 8265
            name: dashboard
          - containerPort: 10001
            name: client
          - containerPort: 8000
            name: serve
          env:
          - name: NVIDIA_VISIBLE_DEVICES
            value: void                # head doesn't need GPU access
  workerGroupSpecs:
  - groupName: gpu-a100
    replicas: 2
    minReplicas: 0
    maxReplicas: 8
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray-ml:2.53.0-gpu
          resources:
            limits:
              cpu: "8"
              memory: 64Gi
              nvidia.com/gpu: "1"
            requests:
              cpu: "8"
              memory: 64Gi
              nvidia.com/gpu: "1"
        tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
        nodeSelector:
          nvidia.com/gpu.product: A100
  - groupName: cpu-workers            # heterogeneous: CPU-only group
    replicas: 4
    minReplicas: 0
    maxReplicas: 16
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray-ml:2.53.0
          resources:
            limits:
              cpu: "16"
              memory: 64Gi
            requests:
              cpu: "16"
              memory: 64Gi
          env:
          - name: NVIDIA_VISIBLE_DEVICES
            value: void
```

### Configuration Best Practices

**Pod sizing:**
- Size each Ray pod to fill one K8s node (fewer large pods > many small)
- Set memory and GPU requests = limits (KubeRay ignores memory/GPU requests, uses limits)
- CPU: set requests only (no limits) to avoid throttling; KubeRay uses requests if limits absent
- KubeRay rounds CPU to nearest integer for Ray resource accounting

**Head pod:**
- Set `num-cpus: "0"` to prevent workloads on head
- Set `dashboard-host: "0.0.0.0"` to expose dashboard
- Set `NVIDIA_VISIBLE_DEVICES: void` if head is on GPU node but shouldn't use GPUs

**Worker groups:**
- All `rayStartParams` values must be strings
- Use same Ray image + version across head and all workers (same Python version too)
- Multiple worker groups for heterogeneous hardware (GPU types, spot vs on-demand)
- Use `nodeSelector` and `tolerations` to target specific node pools

**Custom Ray resources:**
```yaml
rayStartParams:
  resources: '"{\"TPU\": 4, \"custom_resource\": 1}"'  # JSON string of custom resources
```

### Volume Mounts

Mount shared storage for checkpoints, datasets, etc.:

```yaml
template:
  spec:
    containers:
    - name: ray-worker
      volumeMounts:
      - name: shared-data
        mountPath: /mnt/data
    volumes:
    - name: shared-data
      persistentVolumeClaim:
        claimName: shared-pvc
```

### Head Service

KubeRay auto-creates `<cluster>-head-svc` with ports:

| Port | Name | Purpose |
|---|---|---|
| 6379 | gcs | Global Control Store |
| 8265 | dashboard | Ray Dashboard + Jobs API |
| 10001 | client | Ray client connections |
| 8000 | serve | Ray Serve HTTP endpoint |

Override `serviceType` in `headGroupSpec`: `ClusterIP` (default), `NodePort`, `LoadBalancer`.

## RayJob

RayJob manages two things: a **RayCluster** and a **submitter** that calls `ray job submit` to run your code on that cluster. The submitter is NOT your workload — it's a lightweight pod that submits and monitors.

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: training-job
spec:
  entrypoint: python /home/ray/train.py --epochs 10
  runtimeEnvYAML: |
    pip:
      - torch==2.5.0
      - transformers
    env_vars:
      WANDB_API_KEY: "secret"
    working_dir: "https://github.com/org/repo/archive/main.zip"
  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 300
  activeDeadlineSeconds: 7200          # max total runtime
  backoffLimit: 2                      # retries (each = new cluster)
  submissionMode: K8sJobMode           # see submission modes below
  suspend: false                       # true for Kueue integration
  rayClusterSpec:
    rayVersion: "2.53.0"
    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray-ml:2.53.0-gpu
            resources:
              limits:
                cpu: "4"
                memory: 16Gi
    workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 4
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray-ml:2.53.0-gpu
            resources:
              limits:
                cpu: "8"
                memory: 64Gi
                nvidia.com/gpu: "1"
```

### Submission Modes

| Mode | How It Works | When to Use |
|---|---|---|
| `K8sJobMode` (default) | Creates a K8s Job pod that runs `ray job submit` | Most reliable. Works with Kueue. |
| `HTTPMode` | Operator sends HTTP POST to Ray Dashboard directly | No extra pod. Dashboard must be reachable from operator. |
| `SidecarMode` | Injects submitter container into head pod | No extra pod. Cannot use `clusterSelector`. Head restart must be `Never`. |
| `InteractiveMode` (alpha) | Waits for user to submit via `kubectl ray` plugin | Jupyter/notebook workflows. |

In K8sJobMode, the submitter pod gets two injected env vars: `RAY_DASHBOARD_ADDRESS` and `RAY_JOB_SUBMISSION_ID`.

### Key Fields

| Field | Purpose |
|---|---|
| `entrypoint` | Command passed to `ray job submit` |
| `runtimeEnvYAML` | pip packages, env vars, working_dir, py_modules |
| `shutdownAfterJobFinishes` | Delete RayCluster on completion |
| `ttlSecondsAfterFinished` | Delay before cleanup |
| `activeDeadlineSeconds` | Max runtime before `DeadlineExceeded` failure |
| `backoffLimit` | Full retries (each = new cluster). Different from `submitterConfig.backoffLimit` (submitter pod retries). |
| `submissionMode` | See table above |
| `suspend` | Set `true` for Kueue (Kueue controls unsuspension) |
| `clusterSelector` | Use existing RayCluster instead of creating one |
| `entrypointNumCpus/Gpus` | Reserve head resources for driver script |

For full RayJob details (lifecycle, deletion strategies, submitter customization, troubleshooting), see `references/rayjob.md`.

### Using Existing Clusters

Skip cluster creation — submit to a running RayCluster:

```yaml
spec:
  clusterSelector:
    ray.io/cluster: my-existing-cluster
  # Do NOT include rayClusterSpec
```

## Autoscaling

Three levels of autoscaling work together:
1. **Ray Serve** auto-scales replicas (actors) based on request load
2. **Ray Autoscaler** scales Ray worker pods based on logical resource demand
3. **K8s Cluster Autoscaler** provisions new nodes for pending pods

### Configuration

```yaml
spec:
  enableInTreeAutoscaling: true
  autoscalerOptions:
    upscalingMode: Default           # Default | Aggressive | Conservative
    idleTimeoutSeconds: 60           # seconds before removing idle workers
    resources:
      limits:
        cpu: "500m"
        memory: 512Mi
```

| Mode | Behavior |
|---|---|
| `Default` | Scale up to meet demand, conservative bin-packing |
| `Aggressive` | Scale up faster, less bin-packing |
| `Conservative` | Scale up more slowly |

**Key behavior:** The autoscaler monitors *logical* resource demands (from `@ray.remote` decorators), not physical utilization. If a task requests more resources than any single worker provides, the autoscaler won't scale.

**Autoscaler V2** (alpha, Ray ≥ 2.10): Improved observability and stability. Enable via KubeRay feature gate.

## GCS Fault Tolerance

Without GCS FT, head pod failure kills the entire cluster. Enable with external Redis:

```yaml
spec:
  headGroupSpec:
    rayStartParams:
      redis-password: "${REDIS_PASSWORD}"
    template:
      metadata:
        annotations:
          ray.io/ft-enabled: "true"
      spec:
        containers:
        - name: ray-head
          env:
          - name: RAY_REDIS_ADDRESS
            value: "redis:6379"
          - name: RAY_gcs_rpc_server_reconnect_timeout_s
            value: "120"             # worker reconnect timeout (default 60s)
```

With GCS FT: workers continue serving during head recovery, cluster state persists in Redis.

## TLS Authentication

Encrypt gRPC between Ray nodes. Performance overhead is significant for small workloads.

```yaml
# Mount CA cert + per-pod certs, then set env vars:
env:
- name: RAY_USE_TLS
  value: "1"
- name: RAY_TLS_SERVER_CERT
  value: "/etc/ray/tls/tls.crt"
- name: RAY_TLS_SERVER_KEY
  value: "/etc/ray/tls/tls.key"
- name: RAY_TLS_CA_CERT
  value: "/etc/ray/tls/ca.crt"
```

See KubeRay sample: `ray-operator/config/samples/ray-cluster.tls.yaml`

## Kueue Integration

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: queued-training
  labels:
    kueue.x-k8s.io/queue-name: user-queue
spec:
  suspend: true    # Kueue controls unsuspension
  # ... rest of spec
```

Also works with RayCluster (set `spec.suspend: true`).

## Observability

### Ray Dashboard

```bash
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
kubectl port-forward $HEAD_POD 8265:8265
# Open http://localhost:8265
```

### Status Conditions (feature gate: RayClusterStatusConditions)

| Condition | True When |
|---|---|
| `RayClusterProvisioned` | All pods reached ready at least once |
| `HeadPodReady` | Head pod is currently ready |
| `RayClusterReplicaFailure` | Reconciliation error (failed create/delete pod) |

RayService conditions: `Ready` (serving traffic), `UpgradeInProgress` (pending cluster exists).

### Prometheus Metrics

Head pod exposes metrics on port 8080. Configure ServiceMonitor to scrape.

```bash
kubectl exec -it $HEAD_POD -- ray status       # cluster resources
kubectl exec -it $HEAD_POD -- ray list actors   # actor states
kubectl exec -it $HEAD_POD -- ray summary actors
```

### Kubernetes Events

```bash
kubectl describe raycluster <name>   # events: CreatedService, CreatedHeadPod, CreatedWorkerPod
kubectl describe rayjob <name>       # events: job submission, completion, failure
```

## kubectl ray Plugin

The KubeRay kubectl plugin (beta, v1.3.0+) simplifies common cluster management tasks.

### Installation

```bash
# Via Krew (recommended)
kubectl krew install ray

# Or download binary
curl -LO https://github.com/ray-project/kuberay/releases/download/v1.5.1/kubectl-ray_v1.5.1_linux_amd64.tar.gz
tar -xvf kubectl-ray_v1.5.1_linux_amd64.tar.gz
cp kubectl-ray ~/.local/bin/
```

### Cluster Management

```bash
# Create a RayCluster without YAML
kubectl ray create cluster my-cluster \
  --worker-replicas 4 \
  --worker-gpu 1 \
  --worker-memory 32Gi \
  --ray-version 2.53.0

# Add a worker group
kubectl ray create workergroup gpu-group --ray-cluster my-cluster \
  --worker-gpu 1 --worker-memory 64Gi --worker-replicas 2

# List clusters and nodes
kubectl ray get cluster
kubectl ray get workergroup --ray-cluster my-cluster
kubectl ray get nodes

# Scale a worker group
kubectl ray scale cluster my-cluster --worker-group gpu-group --replicas 8

# Delete
kubectl ray delete my-cluster
```

### Sessions, Logs, and Job Submission

```bash
# Port-forward to Ray dashboard + client port
kubectl ray session my-cluster
# → Ray Dashboard: http://localhost:8265
# → Ray Interactive Client: http://localhost:10001

# Download all logs from a cluster
kubectl ray log my-cluster
# Creates ./my-cluster/ directory with head + worker logs

# Submit a job (wraps ray job submit with auto port-forward)
kubectl ray job submit --ray-cluster my-cluster -- python train.py --epochs 10

# Submit with working directory
kubectl ray job submit --ray-cluster my-cluster \
  --working-dir ./src -- python train.py

# Submit with an ephemeral cluster (no existing cluster needed)
kubectl ray job submit -- python train.py
```

### When to Use kubectl ray vs Raw kubectl

| Task | kubectl ray | kubectl |
|---|---|---|
| Quick cluster creation | ✅ `create cluster` (no YAML) | Need full RayCluster YAML |
| Interactive sessions | ✅ `session` (auto port-forward) | Manual `port-forward` |
| Log collection | ✅ `log` (downloads all nodes) | Manual per-pod `kubectl logs` |
| Job submission | ✅ `job submit` (auto-forward) | Create RayJob YAML |
| Complex configs (TLS, GCS FT) | Use kubectl apply | ✅ Full YAML control |

## Key kubectl Commands

```bash
# List all Ray resources
kubectl get rayclusters,rayjobs,rayservices -A

# Ray pods
kubectl get pods -l ray.io/is-ray-node=yes
kubectl get pods -l ray.io/node-type=head
kubectl get pods -l ray.io/node-type=worker

# Head pod logs
kubectl logs $HEAD_POD -c ray-head

# Autoscaler logs (sidecar)
kubectl logs $HEAD_POD -c autoscaler

# Worker init container (if stuck)
kubectl logs <worker-pod> -c wait-gcs-ready

# Operator logs
kubectl logs -n kuberay-system deploy/kuberay-operator

# Ray internal logs
kubectl exec -it $HEAD_POD -- ls /tmp/ray/session_latest/logs/
kubectl exec -it $HEAD_POD -- cat /tmp/ray/session_latest/logs/gcs_server.out
```

## RayService

For Ray Serve deployments with zero-downtime upgrades, see `references/rayservice.md`.

## Troubleshooting

For debugging common KubeRay issues, see `references/troubleshooting.md`.

## Cross-References

- [kueue](../kueue/) — Queue and gang-schedule Ray workloads
- [ray-core](../ray-core/) — Ray programming model
- [ray-train](../ray-train/) — Distributed training on Ray clusters
- [ray-serve](../ray-serve/) — Model serving on Ray clusters
- [ray-data](../ray-data/) — Data processing on Ray clusters

## Reference

- [KubeRay docs](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [KubeRay GitHub](https://github.com/ray-project/kuberay)
- [RayJob docs](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayjob-quick-start.html)
- `references/rayjob.md` — RayJob submitter deep dive
- `references/troubleshooting.md` — common KubeRay issues

- `assets/architecture.md` — Mermaid architecture diagrams
