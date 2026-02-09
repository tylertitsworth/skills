# RayJob Deep Dive

RayJob is the primary way to run batch workloads (training, preprocessing, evaluation) on Kubernetes with Ray. It manages two things: a RayCluster and a job submission mechanism.

## Architecture

```
RayJob CR ──→ KubeRay Operator
                  │
                  ├──→ Creates RayCluster (head + workers)
                  │         │
                  │         └──→ Waits for cluster Ready
                  │
                  └──→ Creates Submitter (K8s Job / HTTP / Sidecar)
                            │
                            └──→ `ray job submit --address=http://$HEAD:8265 -- $entrypoint`
                                      │
                                      └──→ Ray job runs on cluster
                                               │
                                               └──→ On completion: cleanup per policy
```

## The Submitter

The **submitter** is the mechanism that submits your Ray job to the RayCluster. It is NOT your actual workload — it's a lightweight process that calls `ray job submit`. Your actual code runs inside the Ray cluster (on the head or workers).

### Submission Modes

| Mode | How It Works | When to Use |
|---|---|---|
| `K8sJobMode` (default) | KubeRay creates a K8s Job that runs `ray job submit` | Most reliable. Works with Kueue. Use this unless you have a reason not to. |
| `HTTPMode` | KubeRay operator sends HTTP POST to Ray Dashboard API directly | Simpler (no submitter pod). Requires dashboard reachable from operator. |
| `SidecarMode` | KubeRay injects a sidecar container into the head pod | Avoids extra pod. Cannot use `clusterSelector` or `submitterPodTemplate`. Requires head restart policy `Never`. |
| `InteractiveMode` (alpha) | KubeRay waits — user submits manually via `kubectl ray` plugin | Jupyter/notebook workflows. User controls when job starts. |

### K8sJobMode Details

The default mode creates a submitter Kubernetes Job:

1. KubeRay creates the RayCluster
2. Waits for the head pod to be Ready
3. Creates a K8s Job with a pod that runs:
   ```
   ray job submit \
     --address=http://$RAY_DASHBOARD_ADDRESS \
     --submission-id=$RAY_JOB_SUBMISSION_ID \
     -- $entrypoint
   ```
4. The submitter pod has two injected env vars:
   - `RAY_DASHBOARD_ADDRESS` — `<head-svc>:<dashboard-port>`
   - `RAY_JOB_SUBMISSION_ID` — the tracked job ID

The submitter pod is lightweight — it just submits and monitors. Your actual code runs on the Ray cluster.

### Customizing the Submitter

```yaml
spec:
  submissionMode: K8sJobMode
  submitterPodTemplate:
    spec:
      containers:
      - name: ray-job-submitter
        image: rayproject/ray:2.53.0   # must have ray[default] installed
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
      serviceAccountName: ray-submitter
      nodeSelector:
        node-role: utility             # run on utility nodes, not GPU nodes
  submitterConfig:
    backoffLimit: 3                    # submitter retries (not job retries)
```

**Important:** `submitterConfig.backoffLimit` controls submitter pod retries (e.g., if the dashboard is temporarily unreachable). This is different from `spec.backoffLimit` which controls full RayJob retries (each retry = new cluster).

## Lifecycle

### State Machine

```
New → Initializing → Running → Succeeded/Failed
                        │
                        └──→ (if activeDeadlineSeconds exceeded) → Failed (DeadlineExceeded)
```

### Status Fields

```bash
kubectl get rayjob my-job -o jsonpath='{.status}' | jq
```

| Status Field | Meaning |
|---|---|
| `jobStatus` | Ray job status: `PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `STOPPED` |
| `jobDeploymentStatus` | RayJob lifecycle: `Initializing`, `Running`, `Complete`, `Failed`, `Suspended` |
| `startTime` | When job started |
| `endTime` | When job completed |
| `message` | Error details on failure |
| `rayClusterName` | Name of created RayCluster |

### Retry Behavior

```yaml
spec:
  backoffLimit: 3   # retry up to 3 times
```

Each retry **creates a completely new RayCluster**. This means:
- Fresh cluster state (no leftover processes)
- All data in Ray object store is lost
- Use persistent storage (S3/GCS/PVC) for checkpoints

## Entrypoint Resources

Reserve resources on the head node for the driver script:

```yaml
spec:
  entrypointNumCpus: 1
  entrypointNumGpus: 0
  entrypointResources: '{"custom_resource": 1}'
```

Without this, the driver competes with tasks for head resources. For training scripts that use `ray.init()` and schedule work to workers, reserve at least 1 CPU for the driver.

## Runtime Environments

```yaml
spec:
  runtimeEnvYAML: |
    pip:
      - torch==2.5.0
      - transformers==4.46.0
    conda:
      dependencies:
        - pytorch::pytorch=2.5.0
    env_vars:
      WANDB_API_KEY: "secret"
      HF_TOKEN: "hf_..."
    working_dir: "s3://my-bucket/code/v1.zip"
    py_modules:
      - "s3://my-bucket/modules/utils.zip"
    excludes:
      - "*.pyc"
      - "__pycache__"
```

**Gotcha:** Runtime environments install packages on every worker at startup. For large dependency sets, bake them into the container image instead.

## Using Existing Clusters

Skip cluster creation — submit to a long-running RayCluster:

```yaml
spec:
  clusterSelector:
    ray.io/cluster: my-persistent-cluster
  submissionMode: K8sJobMode
  entrypoint: python train.py
  # No rayClusterSpec needed
  # shutdownAfterJobFinishes should be false (don't kill shared cluster)
```

## Deletion Strategies (v1.5.1+, alpha)

Rules-based cleanup replaces `shutdownAfterJobFinishes`:

```yaml
spec:
  deletionStrategy:
    deletionRules:
    # On success: delete workers immediately, cluster after 5min, self after 1hr
    - policy: DeleteWorkers
      condition:
        jobStatus: SUCCEEDED
    - policy: DeleteCluster
      condition:
        jobStatus: SUCCEEDED
        ttlSeconds: 300
    - policy: DeleteSelf
      condition:
        jobStatus: SUCCEEDED
        ttlSeconds: 3600
    # On failure: keep everything for 1hr for debugging
    - policy: DeleteCluster
      condition:
        jobStatus: FAILED
        ttlSeconds: 3600
```

Requires feature gate: `RayJobDeletionPolicy: true`

**Cannot use with** `shutdownAfterJobFinishes` — they are mutually exclusive.

## Kueue Integration

```yaml
metadata:
  labels:
    kueue.x-k8s.io/queue-name: user-queue
spec:
  suspend: true        # Kueue controls unsuspension
  submissionMode: K8sJobMode
```

When Kueue admits the workload, it sets `suspend: false`, triggering cluster creation and job submission.

**Warning:** Don't manually toggle `suspend` if using Kueue — Kueue manages this field.

## Troubleshooting

### Job stuck in Initializing

```bash
# Check if cluster is ready
kubectl get raycluster -l ray.io/rayjob=<rayjob-name>
kubectl describe raycluster <name>   # look for pending pods

# Check if submitter was created (K8sJobMode)
kubectl get jobs -l ray.io/rayjob=<rayjob-name>
```

Common causes: insufficient resources for cluster, image pull errors, PVC mount failures.

### Job submitted but FAILED

```bash
# Get Ray job logs
kubectl exec <head-pod> -- ray job logs <job-id>

# Or via dashboard
kubectl port-forward svc/<head-svc> 8265:8265
# Open http://localhost:8265, navigate to Jobs
```

### Submitter pod CrashLoopBackOff

The submitter can't reach the dashboard. Check:
```bash
# Verify head service exists and has endpoints
kubectl get svc <head-svc>
kubectl get endpoints <head-svc>

# Check submitter logs
kubectl logs job/<rayjob-name>-submitter
```

### activeDeadlineSeconds exceeded

The entire RayJob (cluster creation + job execution) took too long. Increase `activeDeadlineSeconds` or optimize the workload.
