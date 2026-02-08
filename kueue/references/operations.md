# Kueue Operations & Deployment

Comprehensive guide for installing, configuring, monitoring, and operating Kueue in production.

## Installation Methods

### kubectl (Static Manifests)

```bash
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.0/manifests.yaml
kubectl wait deploy/kueue-controller-manager -nkueue-system --for=condition=available --timeout=5m
```

### Helm

```bash
helm install kueue oci://registry.k8s.io/kueue/charts/kueue \
  --version=0.16.0 \
  --namespace kueue-system \
  --create-namespace \
  --wait --timeout 300s
```

#### Key Helm Values

```yaml
# values.yaml
controllerManager:
  manager:
    resources:
      limits:
        cpu: 500m
        memory: 512Mi
      requests:
        cpu: 250m
        memory: 256Mi
  replicas: 1          # HA: set to 2+ with leader election

# Feature gates
featureGates:
  TopologyAwareScheduling: true
  MultiKueue: true
  AdmissionFairSharing: true

# Metrics
enablePrometheus: true

# Custom KueueConfiguration (see below)
managerConfig:
  controllerManagerConfigYaml: |
    apiVersion: config.kueue.x-k8s.io/v1beta2
    kind: Configuration
    # ... (see KueueConfiguration section)
```

### Uninstall

```bash
# kubectl
kubectl delete -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.0/manifests.yaml

# Helm
helm uninstall kueue --namespace kueue-system
```

## KueueConfiguration Reference

The controller reads its config from the `kueue-manager-config` ConfigMap in `kueue-system`. Full reference:

```yaml
apiVersion: config.kueue.x-k8s.io/v1beta2
kind: Configuration
namespace: kueue-system

health:
  healthProbeBindAddress: :8081

metrics:
  bindAddress: :8443
  enableClusterQueueResources: true    # emit per-CQ resource metrics

webhook:
  port: 9443

# Auto-manage jobs without queue-name label (when manageJobsWithoutQueueName is true,
# Kueue manages ALL jobs — use with caution)
manageJobsWithoutQueueName: false

# Internal TLS cert management (disable if using cert-manager)
internalCertManagement:
  enable: true
  webhookServiceName: kueue-webhook-service
  webhookSecretName: kueue-webhook-server-cert

# All-or-nothing: evict jobs whose pods don't become ready
waitForPodsReady:
  enable: true
  timeout: 10m
  requeuingStrategy:
    timestamp: Creation     # or Eviction
    backoffBaseSeconds: 60
    backoffMaxSeconds: 3600
    backoffLimitCount: 10   # max re-queues before deactivation

# Fair sharing (preemption-based)
fairSharing:
  enable: true
  preemptionStrategies:
  - LessThanOrEqualToFinalShare
  - LessThanInitialShare

# Admission fair sharing (usage-based ordering)
admissionFairSharing:
  usageHalfLifeTime: "168h"       # usage decays by half every 7 days
  usageSamplingInterval: "5m"
  resourceWeights:
    cpu: 1.0
    memory: 1.0
    nvidia.com/gpu: 10.0           # weight GPU usage higher

# Queue visibility (pending workloads API)
queueVisibility:
  clusterQueues:
    maxCount: 4000                 # max pending workloads exposed per CQ

# Integrations
integrations:
  frameworks:
  - "batch/job"
  - "jobset.x-k8s.io/jobset"
  - "ray.io/rayjob"
  - "ray.io/raycluster"
  - "kubeflow.org/pytorchjob"
  - "kubeflow.org/mpijob"
  - "kubeflow.org/tfjob"
  - "pod"
  # externalFrameworks for custom integrations (v0.7+)
  labelKeysToCopy:
  - "app.kubernetes.io/name"       # copy these labels from jobs to workloads

# Client connection tuning
clientConnection:
  qps: 50
  burst: 100

# Controller concurrency
controller:
  groupKindConcurrency:
    Job.batch: 5
    Workload.kueue.x-k8s.io: 5
    ClusterQueue.kueue.x-k8s.io: 1

# Resources configuration (DRA device class mappings)
resources:
  excludeResourcePrefixes: []
  transformations: []
```

## Feature Gates

### Alpha Features (disabled by default)

| Gate | Since | Description |
|---|---|---|
| `ElasticJobsViaWorkloadSlices` | v0.13 | Dynamic scaling without suspension |
| `ExternalFrameworks` | v0.7 | Custom job type integration |

### Beta Features (enabled by default)

| Gate | Since | Description |
|---|---|---|
| `TopologyAwareScheduling` | v0.14 | GPU-aware pod placement |
| `MultiKueue` | v0.9 | Multi-cluster job dispatching |
| `AdmissionFairSharing` | v0.15 | Usage-based workload ordering |
| `ProvisioningACC` | GA v0.14 | Cluster-autoscaler integration |

Enable/disable in KueueConfiguration or Helm:

```yaml
# In ConfigMap
featureGates:
  TopologyAwareScheduling: true
  ElasticJobsViaWorkloadSlices: true

# Or via Helm values
featureGates:
  TopologyAwareScheduling: true
```

## Metrics & Monitoring

### Enable Prometheus Scraping

```bash
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.0/prometheus.yaml
```

This creates a ServiceMonitor for prometheus-operator. Metrics are served on `:8443`.

### Key Metrics

| Metric | Type | Description |
|---|---|---|
| `kueue_admitted_active_workloads` | Gauge | Active admitted workloads per CQ |
| `kueue_pending_workloads` | Gauge | Pending workloads per CQ |
| `kueue_admission_wait_time_seconds` | Histogram | Time from creation to admission per CQ |
| `kueue_admission_checks_wait_time_seconds` | Histogram | Time from quota reservation to admission |
| `kueue_evicted_workloads_total` | Counter | Evictions per CQ by reason (Preempted, PodsReadyTimeout, etc.) |
| `kueue_admission_attempts_total` | Counter | Admission attempts (success/inadmissible) |
| `kueue_admission_attempt_duration_seconds` | Histogram | Admission cycle latency |
| `kueue_cluster_queue_status` | Gauge | CQ status (pending/active/terminated) |
| `kueue_cluster_queue_weighted_share` | Gauge | Fair sharing weighted share per CQ |

Enable per-CQ resource usage metrics with `metrics.enableClusterQueueResources: true` in Configuration.

### Useful PromQL

```promql
# Queue saturation (ratio of admitted to nominal GPU quota)
kueue_cluster_queue_nominal_quota{resource="nvidia.com/gpu"}
  - kueue_cluster_queue_resource_reservation{resource="nvidia.com/gpu"}

# Admission latency p99
histogram_quantile(0.99, rate(kueue_admission_wait_time_seconds_bucket[5m]))

# Preemption rate
rate(kueue_evicted_workloads_total{reason="Preempted"}[1h])
```

### Visibility API

Query pending workloads without watching all Workload objects:

```bash
# Pending workloads for a ClusterQueue
kubectl get --raw "/apis/visibility.kueue.x-k8s.io/v1beta1/clusterqueues/<cq>/pendingworkloads"

# Requires API Priority and Fairness config for production use
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.0/visibility-apf.yaml
```

## MultiKueue (Multi-Cluster Dispatching)

Dispatch jobs from a manager cluster to multiple worker clusters. Beta since v0.9.

### Architecture

- **Manager cluster**: runs MultiKueue admission check controller, holds ClusterQueues with total quota
- **Worker clusters**: standalone Kueue installations, receive dispatched jobs
- Manager quota should ≈ sum of all worker quotas

### Setup

#### 1. Worker Cluster Preparation

Each worker needs:
- Kueue installed independently
- Matching namespaces and LocalQueues
- A ServiceAccount with RBAC for MultiKueue (create/delete/get/list/watch on Jobs, Workloads, etc.)

Generate worker kubeconfig:

```bash
# On worker cluster — creates SA + RBAC + kubeconfig file
wget https://raw.githubusercontent.com/kubernetes-sigs/kueue/main/hack/create-multikueue-kubeconfig.sh
chmod +x create-multikueue-kubeconfig.sh
./create-multikueue-kubeconfig.sh worker1.kubeconfig
```

#### 2. Manager Cluster Configuration

```bash
# Store worker kubeconfig as secret
kubectl create secret generic worker1-secret -n kueue-system \
  --from-file=kubeconfig=worker1.kubeconfig
```

```yaml
# MultiKueue admission check
apiVersion: kueue.x-k8s.io/v1beta2
kind: AdmissionCheck
metadata:
  name: multikueue-check
spec:
  controllerName: kueue.x-k8s.io/multikueue
  parameters:
    apiGroup: kueue.x-k8s.io
    kind: MultiKueueConfig
    name: multikueue-config
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: MultiKueueConfig
metadata:
  name: multikueue-config
spec:
  clusters:
  - name: worker1
    kubeConfig:
      locationType: Secret
      location: worker1-secret
---
# ClusterQueue with MultiKueue
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: multi-cluster-queue
spec:
  namespaceSelector: {}
  admissionChecksStrategy:
    admissionChecks:
    - name: multikueue-check
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: default-flavor
      resources:
      - name: "cpu"
        nominalQuota: 100    # sum of all workers
      - name: "memory"
        nominalQuota: 400Gi
      - name: "nvidia.com/gpu"
        nominalQuota: 32
```

### Dispatching Algorithms (v0.16+)

| Algorithm | Behavior |
|---|---|
| **AllAtOnce** (default) | Copy workload to all workers simultaneously |
| **Incremental** | Nominate ≤3 workers at a time, expand every 5 min if not admitted |
| **External** | Delegate nomination to external controller via `status.nominatedClusterNames` |

### Supported MultiKueue Job Types

Jobs, JobSets, Kubeflow jobs (PyTorchJob, MPIJob, TFJob, etc.), RayJobs, RayClusters, Deployments, StatefulSets, plain Pods, AppWrappers.

### Limitations

- Manager cluster cannot be a worker for itself
- Worker namespaces and LocalQueues must mirror manager's

## Upgrades

### In-Place Upgrade

```bash
# kubectl — just re-apply newer manifests
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.0/manifests.yaml

# Helm
helm upgrade kueue oci://registry.k8s.io/kueue/charts/kueue \
  --version=0.16.0 --namespace kueue-system --wait
```

Running workloads are NOT evicted during controller upgrade. The controller resumes managing them after restart.

### CRD Compatibility

- CRDs are forward-compatible within a minor version
- Always upgrade CRDs before the controller
- `kubectl apply --server-side` handles CRD updates (server-side apply required)

## RBAC

Standard roles for batch administrators and users:

```yaml
# Batch admin — manage queues and quotas
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kueue-batch-admin
rules:
- apiGroups: ["kueue.x-k8s.io"]
  resources: ["clusterqueues", "resourceflavors", "workloadpriorityclasses", "admissionchecks"]
  verbs: ["*"]
- apiGroups: ["kueue.x-k8s.io"]
  resources: ["localqueues"]
  verbs: ["*"]
---
# Batch user — submit jobs and view queue status
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kueue-batch-user
rules:
- apiGroups: ["kueue.x-k8s.io"]
  resources: ["localqueues"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["kueue.x-k8s.io"]
  resources: ["workloads"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["kueue.x-k8s.io"]
  resources: ["workloadpriorityclasses"]
  verbs: ["get", "list"]
```

## Certificate Management

Kueue uses internal cert management by default. For cert-manager:

1. Disable internal certs in Configuration: `internalCertManagement.enable: false`
2. Create a cert-manager Certificate for the webhook service
3. Annotate the webhook with `cert-manager.io/inject-ca-from`

See: https://kueue.sigs.k8s.io/docs/tasks/manage/productization/cert_manager/

## Controller Tuning

| Parameter | Default | Tuning Advice |
|---|---|---|
| `clientConnection.qps` | 15 | Increase for large clusters (50-100) |
| `clientConnection.burst` | 20 | 2-3x QPS |
| `controller.groupKindConcurrency` | 1 per kind | Increase for high workload throughput |
| `waitForPodsReady.timeout` | — | Set to catch stuck jobs (e.g., `10m`) |
| `queueVisibility.maxCount` | 10 | Up to 4000 for monitoring dashboards |
