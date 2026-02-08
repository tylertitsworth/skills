# Multi-Tenant Kueue Setup

Fair sharing, cohorts, borrowing, and preemption for multi-team GPU clusters.

## Table of Contents

- [ResourceFlavors for heterogeneous hardware](#resourceflavors-for-heterogeneous-hardware)
- [Cohorts and borrowing](#cohorts-and-borrowing)
- [Preemption policies](#preemption-policies)
- [Complete multi-team example](#complete-multi-team-example)

## ResourceFlavors for Heterogeneous Hardware

Map flavors to node types using labels and tolerations:

```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: gpu-a100
spec:
  nodeLabels:
    nvidia.com/gpu.product: A100
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: gpu-t4
spec:
  nodeLabels:
    nvidia.com/gpu.product: T4
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: spot
spec:
  nodeLabels:
    cloud.google.com/gke-provisioning: spot
  tolerations:
  - key: cloud.google.com/gke-spot
    operator: Equal
    value: "true"
    effect: NoSchedule
```

Kueue injects `nodeSelector` and `tolerations` into admitted pods automatically.

## Cohorts and Borrowing

ClusterQueues in the same **cohort** can borrow unused quota from each other.

- `nominalQuota`: guaranteed resources for this queue
- `borrowingLimit`: max resources this queue can borrow from cohort peers
- `lendingLimit`: max resources this queue lends to cohort peers

```yaml
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: team-a-cq
spec:
  cohort: research-cohort
  namespaceSelector:
    matchLabels:
      kubernetes.io/metadata.name: team-a
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: gpu-a100
      resources:
      - name: "cpu"
        nominalQuota: 20
      - name: "memory"
        nominalQuota: 128Gi
      - name: "nvidia.com/gpu"
        nominalQuota: 4
        borrowingLimit: 4    # can borrow up to 4 more GPUs
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: team-b-cq
spec:
  cohort: research-cohort
  namespaceSelector:
    matchLabels:
      kubernetes.io/metadata.name: team-b
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: gpu-a100
      resources:
      - name: "cpu"
        nominalQuota: 20
      - name: "memory"
        nominalQuota: 128Gi
      - name: "nvidia.com/gpu"
        nominalQuota: 4
        borrowingLimit: 4
```

When team-a is idle, team-b can use up to 8 GPUs total (4 nominal + 4 borrowed).

## Preemption Policies

Set in `.spec.preemption` on ClusterQueue:

```yaml
spec:
  preemption:
    withinClusterQueue: LowerPriority
    reclaimWithinCohort: Any
    borrowWithinCohort:
      policy: LowerPriority
      maxPriorityThreshold: 100
```

| Field | Values | Meaning |
|---|---|---|
| `withinClusterQueue` | `Never`, `LowerPriority`, `LowerOrNewerEqualPriority` | Preempt workloads in the same queue |
| `reclaimWithinCohort` | `Never`, `Any`, `LowerPriority`, `LowerOrNewerEqualPriority` | Reclaim lent quota from cohort peers |
| `borrowWithinCohort.policy` | `Never`, `LowerPriority`, `LowerOrNewerEqualPriority` | Preempt in cohort when borrowing |

**Fair sharing** is enabled by setting `fair-sharing` in the Kueue configuration. It uses DRF (Dominant Resource Fairness) to balance usage across queues in a cohort.

## Queueing Strategy

Set on ClusterQueue `.spec.queueingStrategy`:

| Strategy | Behavior |
|---|---|
| `BestEffortFIFO` (default) | Higher-priority first, but smaller jobs can skip ahead if larger ones don't fit |
| `StrictFIFO` | Strict ordering — older jobs block newer ones even if newer ones fit |

## Flavor Fungibility

Control how Kueue tries alternative flavors when the preferred one is full:

```yaml
spec:
  flavorFungibility:
    whenCanBorrow: Borrow    # or TryNextFlavor
    whenCanPreempt: Preempt  # or TryNextFlavor
```

## Complete Multi-Team Example

A two-team setup sharing an 8-GPU A100 cluster with borrowing and preemption:

```yaml
# ResourceFlavor
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: gpu-a100
spec:
  nodeLabels:
    nvidia.com/gpu.product: A100
---
# Team A — 4 guaranteed GPUs, can borrow 4 more
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: team-a-cq
spec:
  cohort: ml-teams
  queueingStrategy: BestEffortFIFO
  namespaceSelector:
    matchLabels:
      kubernetes.io/metadata.name: team-a
  preemption:
    withinClusterQueue: LowerPriority
    reclaimWithinCohort: LowerPriority
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: gpu-a100
      resources:
      - name: "cpu"
        nominalQuota: 32
      - name: "memory"
        nominalQuota: 128Gi
      - name: "nvidia.com/gpu"
        nominalQuota: 4
        borrowingLimit: 4
---
# Team B — same config
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: team-b-cq
spec:
  cohort: ml-teams
  queueingStrategy: BestEffortFIFO
  namespaceSelector:
    matchLabels:
      kubernetes.io/metadata.name: team-b
  preemption:
    withinClusterQueue: LowerPriority
    reclaimWithinCohort: LowerPriority
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: gpu-a100
      resources:
      - name: "cpu"
        nominalQuota: 32
      - name: "memory"
        nominalQuota: 128Gi
      - name: "nvidia.com/gpu"
        nominalQuota: 4
        borrowingLimit: 4
---
# LocalQueues
apiVersion: kueue.x-k8s.io/v1beta2
kind: LocalQueue
metadata:
  name: training-queue
  namespace: team-a
spec:
  clusterQueue: team-a-cq
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: LocalQueue
metadata:
  name: training-queue
  namespace: team-b
spec:
  clusterQueue: team-b-cq
```
