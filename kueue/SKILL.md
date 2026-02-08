---
name: kueue
description: >
  Manage Kueue, the Kubernetes-native job queueing system for batch and ML workloads.
  Use when: (1) Setting up Kueue (install, ClusterQueues, LocalQueues, ResourceFlavors,
  WorkloadPriorityClasses), (2) Submitting or managing batch workloads through Kueue
  (Jobs, RayJobs, PyTorchJobs, JobSets), (3) Debugging queueing issues (pending workloads,
  quota exhaustion, preemption, admission failures), (4) Configuring fair sharing, cohorts,
  borrowing limits, and preemption policies, (5) Monitoring queue utilization and pending
  workloads, (6) Integrating Kueue with KubeRay, Kubeflow Training Operator, or JobSet.
---

# Kueue

Kubernetes-native job queueing system. Manages quotas and decides when workloads should wait, start, or be preempted.

**Docs:** https://kueue.sigs.k8s.io/docs/
**GitHub:** https://github.com/kubernetes-sigs/kueue
**Current version:** v0.16.0 (requires Kubernetes ≥ 1.29)

## Core Concepts

Kueue has four primary API objects:

| Object | Scope | Purpose |
|---|---|---|
| **ResourceFlavor** | Cluster | Maps to node types (GPU models, spot vs on-demand, architectures) |
| **ClusterQueue** | Cluster | Defines resource quotas per flavor, fair sharing, preemption rules |
| **LocalQueue** | Namespace | Tenant-facing queue that points to a ClusterQueue |
| **WorkloadPriorityClass** | Cluster | Priority values for queue ordering (independent of pod priority) |

**Flow:** Job → LocalQueue → ClusterQueue → admission → pods created.

Kueue creates a **Workload** object for each job to track admission status.

## Installation

```bash
# kubectl
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.0/manifests.yaml
kubectl wait deploy/kueue-controller-manager -nkueue-system --for=condition=available --timeout=5m

# Helm
helm install kueue oci://registry.k8s.io/kueue/charts/kueue \
  --version=0.16.0 --namespace kueue-system --create-namespace --wait

# Prometheus metrics (optional)
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.0/prometheus.yaml
```

## Minimal Setup

A single-queue setup for a small cluster:

```yaml
# 1. ResourceFlavor (default, no node selectors)
apiVersion: kueue.x-k8s.io/v1beta2
kind: ResourceFlavor
metadata:
  name: default-flavor
---
# 2. ClusterQueue
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
# 3. LocalQueue (per namespace)
apiVersion: kueue.x-k8s.io/v1beta2
kind: LocalQueue
metadata:
  name: user-queue
  namespace: default
spec:
  clusterQueue: cluster-queue
```

## Submitting Workloads

Add the `kueue.x-k8s.io/queue-name` label to any supported job type:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job
  labels:
    kueue.x-k8s.io/queue-name: user-queue
    kueue.x-k8s.io/priority-class: high-priority  # optional
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

**Key:** Set `suspend: true` — Kueue controls when the job starts. For Kubeflow jobs and RayJobs, Kueue handles suspension automatically via the integration.

### Supported Job Types

| Kind | API Group | Notes |
|---|---|---|
| Job | batch/v1 | Native support, partial admission available |
| JobSet | jobset.x-k8s.io | Multi-template jobs |
| RayJob | ray.io | Via KubeRay operator |
| RayCluster | ray.io | Via KubeRay operator |
| PyTorchJob | kubeflow.org | Via Training Operator |
| PaddleJob | kubeflow.org | Via Training Operator |
| TFJob | kubeflow.org | Via Training Operator |
| XGBoostJob | kubeflow.org | Via Training Operator |
| MPIJob | kubeflow.org | Via Training Operator |
| Deployment | apps/v1 | For serving workloads |
| StatefulSet | apps/v1 | For stateful serving |

## Multi-Tenant Setup with Fair Sharing

For multi-team GPU clusters, see `references/multi-tenant.md`.

## Troubleshooting

For debugging pending workloads, preemption, and admission failures, see `references/troubleshooting.md`.

## WorkloadPriorityClass

Define queue priority independently from pod scheduling priority:

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
description: "Experimental and dev jobs"
```

Apply to jobs via label: `kueue.x-k8s.io/priority-class: high-priority`

## Key kubectl Commands

```bash
# List all Kueue objects
kubectl get clusterqueues
kubectl get localqueues -A
kubectl get resourceflavors
kubectl get workloads -A
kubectl get workloadpriorityclass

# Check ClusterQueue status and usage
kubectl describe clusterqueue <name>

# Check workload admission status
kubectl get workload -n <ns> <name> -o yaml

# Find the workload for a job
JOB_UID=$(kubectl get job -n <ns> <name> -o jsonpath='{.metadata.uid}')
kubectl get workloads -n <ns> -l "kueue.x-k8s.io/job-uid=$JOB_UID"

# Pending workloads visibility API
kubectl get --raw "/apis/visibility.kueue.x-k8s.io/v1beta1/clusterqueues/<cq>/pendingworkloads"

# Check Kueue controller logs
kubectl logs -n kueue-system deploy/kueue-controller-manager
```
