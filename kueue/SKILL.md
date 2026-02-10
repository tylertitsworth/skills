---
name: kueue
description: >
  Manage Kueue, the Kubernetes-native job queueing system for batch, ML, and serving workloads.
  Use when: (1) Installing and configuring Kueue (Helm, kubectl, feature gates, KueueConfiguration),
  (2) Setting up ClusterQueues, LocalQueues, ResourceFlavors, and WorkloadPriorityClasses,
  (3) Submitting or managing workloads (Jobs, JobSets, RayJobs, RayClusters, RayServices,
  PyTorchJobs, MPIJobs, TFJobs, PaddleJobs, XGBoostJobs, TrainJobs, Deployments, StatefulSets,
  LeaderWorkerSets, plain Pods, pod groups), (4) Configuring fair sharing (preemption-based and
  admission-based), cohorts, borrowing/lending limits, and preemption policies, (5) Setting up
  Topology Aware Scheduling (TAS) for GPU pod placement optimization, (6) Configuring MultiKueue
  for multi-cluster job dispatching, (7) Using AdmissionChecks and ProvisioningRequests for
  cluster-autoscaler integration, (8) Debugging queueing issues (pending workloads, quota
  exhaustion, preemption, admission failures), (9) Monitoring Kueue metrics and visibility API,
  (10) Elastic workloads, dynamic reclaim, and partial admission.
---

# Kueue

Kubernetes-native job queueing system. Manages quotas and decides when workloads should wait, start, or be preempted.

**Docs:** https://kueue.sigs.k8s.io/docs/  
**GitHub:** https://github.com/kubernetes-sigs/kueue  
**Version:** v0.16.0 | **Requires:** Kubernetes ≥ 1.29

## Core API Objects

| Object | Scope | Purpose |
|---|---|---|
| **ResourceFlavor** | Cluster | Maps to node types (GPU models, spot/on-demand, architectures). Optional `topologyName` for TAS. |
| **ClusterQueue** | Cluster | Defines resource quotas per flavor, fair sharing, preemption, admission checks |
| **LocalQueue** | Namespace | Tenant-facing queue pointing to a ClusterQueue |
| **WorkloadPriorityClass** | Cluster | Priority for queue ordering (independent of pod priority) |
| **Workload** | Namespace | Unit of admission — auto-created for each job |
| **Topology** | Cluster | Hierarchical node topology for TAS (block → rack → node) |
| **AdmissionCheck** | Cluster | Gate admission on external signals (provisioning, MultiKueue) |

**Flow:** Job → LocalQueue → ClusterQueue → quota reservation → admission checks → admission → pods created.

## Installation & Operations

See `references/operations.md` for comprehensive deployment, configuration, Helm values, metrics, upgrades, MultiKueue setup, and feature gates.

Quick install:

```bash
# kubectl
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.0/manifests.yaml

# Helm
helm install kueue oci://registry.k8s.io/kueue/charts/kueue \
  --version=0.16.0 --namespace kueue-system --create-namespace --wait
```

## Minimal Setup

```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: default-flavor
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: cluster-queue
spec:
  namespaceSelector: {}
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: default-flavor
      resources:
      - name: "cpu"
        nominalQuota: 40
      - name: "memory"
        nominalQuota: 256Gi
      - name: "nvidia.com/gpu"
        nominalQuota: 8
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: LocalQueue
metadata:
  name: user-queue
  namespace: default
spec:
  clusterQueue: cluster-queue
```

## Submitting Workloads

Label any supported job with `kueue.x-k8s.io/queue-name`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job
  labels:
    kueue.x-k8s.io/queue-name: user-queue
    kueue.x-k8s.io/priority-class: high-priority
spec:
  suspend: true  # Kueue unsuspends on admission
  parallelism: 4
  completions: 4
  template:
    spec:
      containers:
      - name: trainer
        image: training:latest
        resources:
          requests:
            cpu: "2"
            memory: 8Gi
            nvidia.com/gpu: "1"
      restartPolicy: Never
```

### Supported Integrations

**Batch:** Job, JobSet, RayJob, RayCluster, PyTorchJob, TFJob, MPIJob, PaddleJob, XGBoostJob, TrainJob (Trainer v2), plain Pods, pod groups  
**Serving:** Deployment, StatefulSet, LeaderWorkerSet, RayService

Enable integrations in the KueueConfiguration:

```yaml
integrations:
  frameworks:
  - "batch/job"
  - "jobset.x-k8s.io/jobset"
  - "ray.io/rayjob"
  - "ray.io/raycluster"
  - "kubeflow.org/pytorchjob"
  - "kubeflow.org/mpijob"
  - "pod"
```

### Pod Groups (Plain Pods)

Group multiple pods as a single workload using the group label:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: worker-0
  labels:
    kueue.x-k8s.io/queue-name: user-queue
  annotations:
    kueue.x-k8s.io/pod-group-name: my-training
    kueue.x-k8s.io/pod-group-total-count: "4"
spec:
  containers:
  - name: worker
    image: training:latest
    resources:
      requests:
        nvidia.com/gpu: "1"
```

## ClusterQueue Configuration

### Resource Groups

Resources in the same group are assigned the same flavor (e.g., GPU + CPU + memory on the same node type):

```yaml
spec:
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: gpu-a100
      resources:
      - name: "cpu"
        nominalQuota: 64
      - name: "memory"
        nominalQuota: 512Gi
      - name: "nvidia.com/gpu"
        nominalQuota: 8
        borrowingLimit: 4    # max borrow from cohort
        lendingLimit: 2      # max lend to cohort
    - name: gpu-t4           # fallback flavor
      resources:
      - name: "cpu"
        nominalQuota: 32
      - name: "memory"
        nominalQuota: 128Gi
      - name: "nvidia.com/gpu"
        nominalQuota: 4
```

Kueue tries flavors in order — A100 first, T4 as fallback.

### Namespace Selector

Restrict which namespaces can submit to a ClusterQueue:

```yaml
spec:
  namespaceSelector:
    matchLabels:
      kubernetes.io/metadata.name: team-a  # single namespace
  # OR
  namespaceSelector:
    matchLabels:
      research-cohort: ml-team             # custom label on multiple namespaces
  # OR
  namespaceSelector: {}                     # all namespaces
```

### Queueing Strategy

| Strategy | Behavior |
|---|---|
| `BestEffortFIFO` (default) | Priority-ordered, but smaller jobs can skip ahead if larger ones don't fit |
| `StrictFIFO` | Strict ordering — head-of-line blocks even if smaller jobs fit |

### Cohorts and Borrowing

ClusterQueues in the same cohort share unused quota. See `references/multi-tenant.md` for full examples.

- `nominalQuota` — guaranteed resources
- `borrowingLimit` — max resources this queue can borrow
- `lendingLimit` — max resources this queue lends out

### Flavor Fungibility

Controls behavior when preferred flavor is full:

```yaml
spec:
  flavorFungibility:
    whenCanBorrow: Borrow       # or TryNextFlavor
    whenCanPreempt: TryNextFlavor  # or Preempt
```

### Stop Policy

Pause a ClusterQueue without deleting it:

```yaml
spec:
  stopPolicy: HoldAndDrain  # or Hold, None
```

- `Hold` — stop admitting new workloads, keep admitted ones running
- `HoldAndDrain` — stop admitting and evict all admitted workloads

## Preemption

Configure in `.spec.preemption`:

```yaml
spec:
  preemption:
    withinClusterQueue: LowerPriority          # Never | LowerPriority | LowerOrNewerEqualPriority
    reclaimWithinCohort: Any                    # Never | Any | LowerPriority | LowerOrNewerEqualPriority
    borrowWithinCohort:
      policy: LowerPriority                    # Never | LowerPriority | LowerOrNewerEqualPriority
      maxPriorityThreshold: 100                # only preempt workloads at or below this priority
```

**Preemption order:** Borrowing workloads in cohort → lowest priority → most recently admitted.

### Fair Sharing (Preemption-Based)

Enables DRF-based preemption across cohorts. Enable globally in KueueConfiguration:

```yaml
fairSharing:
  enable: true
  preemptionStrategies:
  - LessThanOrEqualToFinalShare   # preempt if preemptor share ≤ target share after preemption
  - LessThanInitialShare          # fallback: preempt if preemptor share < target share before
```

Per-ClusterQueue weight:

```yaml
spec:
  fairSharing:
    weight: "2"  # this queue counts as half-usage relative to weight-1 queues
```

### Admission Fair Sharing (Usage-Based)

Orders workloads by historical LocalQueue resource consumption. Beta in v0.15.

Enable in KueueConfiguration:

```yaml
admissionFairSharing:
  usageHalfLifeTime: "168h"
  usageSamplingInterval: "5m"
  resourceWeights:
    cpu: 2.0
    memory: 1.0
```

Enable per-ClusterQueue:

```yaml
spec:
  admissionScope:
    admissionMode: UsageBasedAdmissionFairSharing
```

Per-LocalQueue weight:

```yaml
spec:
  fairSharing:
    weight: "2"
```

Track usage: `kubectl get lq <name> -o jsonpath='{.status.fairSharing}'`

## Topology Aware Scheduling (TAS)

Optimizes pod placement for network throughput. Beta in v0.14. Critical for distributed training.

### Setup

```yaml
# 1. Define topology hierarchy
apiVersion: kueue.x-k8s.io/v1beta2
kind: Topology
metadata:
  name: gpu-topology
spec:
  levels:
  - nodeLabel: "topology.kubernetes.io/zone"       # block level
  - nodeLabel: "kubernetes.io/hostname"             # node level (required as lowest for hot-swap)
---
# 2. Reference from ResourceFlavor
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: tas-gpu
spec:
  nodeLabels:
    node-pool: gpu-nodes
  topologyName: gpu-topology
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
```

### User Annotations (PodTemplate level)

- `kueue.x-k8s.io/podset-required-topology: <level>` — all pods MUST be in the same topology domain
- `kueue.x-k8s.io/podset-preferred-topology: <level>` — best-effort, falls back to wider domains
- `kueue.x-k8s.io/podset-unconstrained-topology: ""` — TAS capacity accounting, no placement constraint
- `kueue.x-k8s.io/podset-group-name: <name>` — group multiple PodSets into same domain

Example — 8-GPU training requiring same host:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: distributed-training
  labels:
    kueue.x-k8s.io/queue-name: user-queue
spec:
  parallelism: 8
  completions: 8
  completionMode: Indexed
  template:
    metadata:
      annotations:
        kueue.x-k8s.io/podset-required-topology: "kubernetes.io/hostname"
    spec:
      containers:
      - name: trainer
        image: training:latest
        resources:
          requests:
            nvidia.com/gpu: "1"
```

**Hot-swap:** When lowest topology level is hostname, TAS supports automatic node replacement on failure without evicting the entire workload. Beta in v0.14.

## AdmissionChecks

External gates that must pass before a workload is admitted.

### ProvisioningRequest (Cluster Autoscaler)

Ensures physical GPU capacity exists before admission. GA in v0.14.

```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: ProvisioningRequestConfig
metadata:
  name: gpu-provisioning
spec:
  provisioningClassName: check-capacity.autoscaling.x-k8s.io
  managedResources:
  - nvidia.com/gpu
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: AdmissionCheck
metadata:
  name: gpu-capacity-check
spec:
  controllerName: kueue.x-k8s.io/provisioning-request
  parameters:
    apiGroup: kueue.x-k8s.io
    kind: ProvisioningRequestConfig
    name: gpu-provisioning
---
# Reference from ClusterQueue
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: autoscaled-queue
spec:
  admissionChecks:
  - gpu-capacity-check
  resourceGroups:
  - coveredResources: ["nvidia.com/gpu", "cpu", "memory"]
    flavors:
    - name: gpu-a100
      resources:
      - name: "nvidia.com/gpu"
        nominalQuota: 16
      - name: "cpu"
        nominalQuota: 64
      - name: "memory"
        nominalQuota: 512Gi
```

Flow: quota reserved → ProvisioningRequest created → cluster-autoscaler scales nodes → `Provisioned=true` → workload admitted.

Per-flavor admission checks: use `admissionChecksStrategy.admissionChecks[].onFlavors` to scope checks to specific flavors.

## Elastic Workloads

Dynamic scaling without suspension. Alpha in v0.13. Feature gate: `ElasticJobsViaWorkloadSlices: true`. Annotate job with `kueue.x-k8s.io/elastic-job: "true"` — scale up creates a new Workload Slice, scale down updates existing workload.

## Dynamic Reclaim

Admitted workloads release unused quota early via `status.reclaimablePods`. Count can only increase while quota is held.

## WorkloadPriorityClass

```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: WorkloadPriorityClass
metadata:
  name: high-priority
value: 10000
description: "Production training jobs"
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: WorkloadPriorityClass
metadata:
  name: low-priority
value: 100
description: "Experimental jobs"
```

Apply via label: `kueue.x-k8s.io/priority-class: high-priority`

## Feature Compatibility & Gotchas

Not all Kueue features compose freely. Key incompatibilities:

| Feature A | Feature B | Status | Notes |
|---|---|---|---|
| Elastic Workloads | TAS | ❌ Incompatible | Elastic workloads do not support Topology Aware Scheduling |
| Elastic Workloads | MultiKueue | ❌ Incompatible | Elastic workloads cannot be dispatched across clusters |
| TAS | Pod affinities/anti-affinities | ⚠️ Ignored | TAS overrides kube-scheduler placement; pod affinities are silently ignored |
| TAS | `podset-required-topology` | ⚠️ May fail | If ClusterAutoscaler cannot provision nodes matching required topology |
| MultiKueue | Manager as worker | ❌ Unsupported | Manager cluster cannot also be its own worker |
| Preemption (borrowing) | Fair sharing | ⚠️ Interaction | Preempting ClusterQueue above nominal quota behaves differently with `Classic` vs `Fair` algorithms |
| All-or-nothing (WaitForPodsReady) | Tight quotas | ⚠️ Deadlock risk | Two large jobs can deadlock if physical capacity < configured quota; use `requeuingStrategy` with backoff |
| LimitRanges | Workload admission | ⚠️ Inadmissible | If adjusted resource values violate namespace LimitRanges, workload is marked Inadmissible |
| Zero resource requests | Missing CQ resource | ⚠️ Blocks admission | Requesting "0" of a resource not defined in ClusterQueue blocks the workload |

### Upgrade Gotchas

- **CRD API changes between versions** — major/minor upgrades often change CRD schemas. You must update CRDs before the controller, and **all existing workloads may need to be drained** because old Workload objects may not be compatible with new CRD versions. Always read the release notes.
- **Feature gate graduation** — alpha features become beta (enabled by default) on upgrade, potentially changing behavior. Check `featureGates` in your KueueConfiguration.
- **Webhook certificate rotation** — Kueue uses internal cert management by default. If switching to cert-manager mid-life, plan for downtime.
- **Kubernetes version requirements** — Kueue v0.14+ requires Kubernetes 1.29+. Upgrade K8s first.

### General Gotchas

- **Workload names are generated** — don't rely on workload naming conventions; use labels/ownerReferences to find the workload for a job.
- **Quota is virtual** — Kueue quotas don't enforce actual resource limits. A ClusterQueue with 100 GPUs will admit 100 GPUs of work even if the cluster only has 80 physical GPUs. Use `WaitForPodsReady` to handle the gap.
- **Preemption is not instant** — preempted workloads are set to `spec.active: false` (suspension). The owning controller must actually stop the pods. If the controller doesn't support suspension, the workload hangs.
- **Borrowing requires a cohort** — `borrowingLimit` on a ClusterQueue does nothing without a cohort containing other ClusterQueues with available quota.

## Workload Lifecycle

- `spec.active: false` — deactivates workload (evicts if admitted)
- `spec.maximumExecutionTimeSeconds: N` — auto-evict after N seconds

## kueuectl CLI Plugin

Install the kubectl-kueue plugin for streamlined Kueue management:

```bash
# Install via Krew (recommended)
kubectl krew install kueue

# Or download binary directly
curl -Lo ./kubectl-kueue https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.0/kubectl-kueue-linux-amd64
chmod +x ./kubectl-kueue && sudo mv ./kubectl-kueue /usr/local/bin/kubectl-kueue

# Optional alias
alias kueuectl="kubectl kueue"
```

### Commands

```bash
# List resources
kubectl kueue list clusterqueues
kubectl kueue list localqueues -n my-namespace
kubectl kueue list workloads -n my-namespace

# Create resources
kubectl kueue create clusterqueue my-cq \
  --nominal-quota cpu=10,memory=64Gi,nvidia.com/gpu=8 \
  --cohort default-cohort
kubectl kueue create localqueue my-lq -n my-namespace \
  --clusterqueue my-cq
kubectl kueue create resourceflavor gpu-flavor \
  --node-labels cloud.provider.com/accelerator=nvidia-a100

# Resume suspended workloads
kubectl kueue resume workload my-workload -n my-namespace
kubectl kueue resume localqueue my-lq -n my-namespace
kubectl kueue resume clusterqueue my-cq

# Stop (suspend) workloads/queues
kubectl kueue stop workload my-workload -n my-namespace
kubectl kueue stop localqueue my-lq -n my-namespace
kubectl kueue stop clusterqueue my-cq

# Passthrough — standard kubectl commands with Kueue context
kubectl kueue get clusterqueue my-cq -o yaml
kubectl kueue describe workload my-workload -n my-namespace

# Version
kubectl kueue version
```

### When to Use kueuectl vs kubectl

| Task | kueuectl | kubectl |
|---|---|---|
| Create CQ with quota inline | ✅ `create clusterqueue --nominal-quota` | ❌ Need full YAML |
| Resume/stop workloads | ✅ `resume workload` / `stop workload` | Manual patch of `spec.active` |
| List with Kueue-aware formatting | ✅ `list workloads` | Generic `get workloads` |
| View pending workloads | Use kubectl visibility API | `get --raw /apis/visibility...` |
| Complex YAML resources | Use kubectl apply | Use kubectl apply |

## Key kubectl Commands

```bash
# List all Kueue objects
kubectl get clusterqueues,localqueues,resourceflavors,workloads,workloadpriorityclasses -A

# ClusterQueue status and usage
kubectl describe clusterqueue <name>
kubectl get cq <name> -o jsonpath='{.status.flavorsReservation}'

# Workload for a job
JOB_UID=$(kubectl get job -n <ns> <name> -o jsonpath='{.metadata.uid}')
kubectl get workloads -n <ns> -l "kueue.x-k8s.io/job-uid=$JOB_UID"

# Pending workloads (visibility API)
kubectl get --raw "/apis/visibility.kueue.x-k8s.io/v1beta1/clusterqueues/<cq>/pendingworkloads"

# Fair sharing status
kubectl get cq <name> -o jsonpath='{.status.fairSharing}'

# Controller logs
kubectl logs -n kueue-system deploy/kueue-controller-manager --tail=200
```

## Multi-Tenant Setup

For cohorts, borrowing, preemption policies, and ResourceFlavor examples, see `references/multi-tenant.md`.

## Troubleshooting

For debugging pending workloads, admission failures, and preemption, see `references/troubleshooting.md`.

## Operations & Deployment

For installation options, Helm configuration, KueueConfiguration reference, metrics, feature gates, upgrades, and MultiKueue setup, see `references/operations.md`.

## Cross-References

- [kuberay](../kuberay/) — Gang scheduling RayJob/RayCluster workloads with Kueue
- [aws-fsx](../aws-fsx/) — FSx storage for queued training jobs

## Reference

- [Kueue docs](https://kueue.sigs.k8s.io/docs/)
- [Kueue GitHub](https://github.com/kubernetes-sigs/kueue)
- `references/operations.md` — deployment, Helm, metrics, MultiKueue
- `references/multi-tenant.md` — multi-tenancy patterns
- `references/troubleshooting.md` — common issues and fixes
- `assets/clusterqueue.yaml` — complete ClusterQueue + ResourceFlavor + LocalQueue example with GPU quotas, borrowing, and fair sharing
- `assets/tas-training-job.yaml` — Topology-Aware Scheduling Job for distributed GPU training

- `assets/architecture.md` — Mermaid architecture diagrams
