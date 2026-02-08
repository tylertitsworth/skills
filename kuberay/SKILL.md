---
name: kuberay
description: >
  Deploy and manage Ray clusters on Kubernetes using the KubeRay operator. Use when:
  (1) Deploying RayCluster resources (head/worker config, GPU scheduling, autoscaling),
  (2) Submitting RayJob or RayService workloads declaratively via K8s manifests,
  (3) Managing Ray cluster lifecycle (scaling, upgrades, rolling updates),
  (4) Debugging Ray on K8s issues (pods pending, init container stuck, GCS failures,
  autoscaler not scaling, head node crashes),
  (5) Integrating KubeRay with Kueue for job queueing,
  (6) Configuring Ray Dashboard, Prometheus metrics, and observability.
---

# KubeRay

Kubernetes operator for Ray. Provides CRDs for running distributed Ray workloads natively on Kubernetes.

**Docs:** https://docs.ray.io/en/latest/cluster/kubernetes/index.html
**GitHub:** https://github.com/ray-project/kuberay
**Operator version:** v1.5.1 | **Ray version:** 2.53.0
**API version:** `ray.io/v1`

## CRDs

| CRD | Purpose |
|---|---|
| **RayCluster** | Long-running Ray cluster (head + worker groups) |
| **RayJob** | One-shot: creates RayCluster, submits job, optionally cleans up |
| **RayService** | Ray Serve deployments with zero-downtime upgrades |

## Installation

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 1.5.1 \
  --namespace kuberay-system --create-namespace
```

Verify:

```bash
kubectl get pods -n kuberay-system
```

## RayCluster

Minimal GPU cluster:

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: gpu-cluster
spec:
  rayVersion: "2.53.0"
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
      num-cpus: "0"              # prevent workloads on head
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray-ml:2.53.0
          resources:
            limits:
              cpu: "4"
              memory: 8Gi
            requests:
              cpu: "4"
              memory: 8Gi
          ports:
          - containerPort: 6379
            name: gcs
          - containerPort: 8265
            name: dashboard
          - containerPort: 10001
            name: client
  workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 2
    minReplicas: 1
    maxReplicas: 8
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray-ml:2.53.0
          resources:
            limits:
              cpu: "8"
              memory: 32Gi
              nvidia.com/gpu: "1"
            requests:
              cpu: "8"
              memory: 32Gi
              nvidia.com/gpu: "1"
        tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

### Key Configuration Notes

- Set `num-cpus: "0"` on head to reserve it for cluster control only
- Set `dashboard-host: "0.0.0.0"` on head to expose the dashboard
- Size each Ray pod to fill one K8s node for best performance (fewer large pods > many small)
- Use the same Ray image + version across head and all worker groups
- KubeRay auto-detects CPU/memory/GPU from container resource limits
- Set memory and GPU requests equal to limits (KubeRay ignores memory/GPU requests)
- All `rayStartParams` values must be strings

## RayJob

Creates a RayCluster, submits a Ray job, and optionally cleans up:

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: training-job
spec:
  entrypoint: python /home/ray/train.py
  runtimeEnvYAML: |
    pip:
      - torch==2.1.0
    working_dir: "https://github.com/org/repo/archive/main.zip"
  rayClusterSpec:
    rayVersion: "2.53.0"
    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray-ml:2.53.0
            resources:
              limits:
                cpu: "4"
                memory: 8Gi
              requests:
                cpu: "4"
                memory: 8Gi
    workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 4
      rayStartParams: {}
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray-ml:2.53.0
            resources:
              limits:
                cpu: "8"
                memory: 32Gi
                nvidia.com/gpu: "1"
              requests:
                cpu: "8"
                memory: 32Gi
                nvidia.com/gpu: "1"
  shutdownAfterJobFinishes: true
  ttlSecondsAfterFinished: 300
  backoffLimit: 2
  submissionMode: K8sJobMode
```

### RayJob Key Fields

| Field | Purpose |
|---|---|
| `entrypoint` | Command to run (passed to `ray job submit`) |
| `runtimeEnvYAML` | Pip packages, env vars, working_dir |
| `shutdownAfterJobFinishes` | Delete RayCluster when job completes |
| `ttlSecondsAfterFinished` | Delay before cleanup (seconds) |
| `backoffLimit` | Retries before marking failed (each retry = new cluster) |
| `submissionMode` | `K8sJobMode` (default), `HTTPMode`, `InteractiveMode`, `SidecarMode` |
| `suspend` | Set `true` for Kueue integration (Kueue controls unsuspension) |
| `clusterSelector` | Use an existing RayCluster instead of creating one |

## RayService

For Ray Serve deployments, see `references/rayservice.md`.

## Autoscaling

Enable autoscaling in the RayCluster spec:

```yaml
spec:
  enableInTreeAutoscaling: true
  autoscalerOptions:
    upscalingMode: Default        # Default, Aggressive, Conservative
    idleTimeoutSeconds: 60
    resources:
      limits:
        cpu: "1"
        memory: 1Gi
```

Set `minReplicas` and `maxReplicas` on each worker group. The autoscaler runs as a sidecar on the head pod.

**How it works:** Ray Autoscaler monitors logical resource demands (from `@ray.remote` decorators), not physical utilization. It scales workers up when tasks/actors are queued, and down when workers are idle.

**Common pitfall:** If a task requests more resources than any single worker can provide, the autoscaler won't scale up. Ensure task resource requests fit within a single worker pod.

## Kueue Integration

Add the queue label and set `suspend: true`:

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: queued-training
  labels:
    kueue.x-k8s.io/queue-name: user-queue
spec:
  suspend: true
  # ... rest of RayJob spec
```

Kueue manages the `suspend` field — do not manually toggle it when using Kueue.

## Observability

```bash
# Ray Dashboard (port-forward)
kubectl port-forward svc/<cluster>-head-svc 8265:8265
# Open http://localhost:8265

# Ray status from head pod
kubectl exec -it <head-pod> -- ray status

# Cluster resource usage
kubectl exec -it <head-pod> -- ray status | grep -A20 "Resources"
```

KubeRay exposes Prometheus metrics on the head pod (port 8080). Configure a ServiceMonitor to scrape them.

## Key kubectl Commands

```bash
# List Ray resources
kubectl get rayclusters,rayjobs,rayservices -A

# Check cluster status
kubectl describe raycluster <name>

# View Ray pods
kubectl get pods -l ray.io/is-ray-node=yes

# Head pod logs
kubectl logs <head-pod> -c ray-head

# Worker init container logs (if stuck)
kubectl logs <worker-pod> -c wait-gcs-ready

# Operator logs
kubectl logs -n kuberay-system deploy/kuberay-operator
```

## Troubleshooting

For debugging common KubeRay issues, see `references/troubleshooting.md`.
