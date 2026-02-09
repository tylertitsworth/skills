# KubeRay Troubleshooting

Common issues when running Ray on Kubernetes with KubeRay.

## Table of Contents

- [Worker pods stuck in Init:0/1](#worker-pods-stuck-in-init01)
- [Head pod crashes or restarts](#head-pod-crashes-or-restarts)
- [Autoscaler not scaling up](#autoscaler-not-scaling-up)
- [Autoscaler not scaling down](#autoscaler-not-scaling-down)
- [Pods pending (not scheduled)](#pods-pending-not-scheduled)
- [RayJob stuck or failed](#rayjob-stuck-or-failed)
- [GCS fault tolerance](#gcs-fault-tolerance)
- [Version compatibility](#version-compatibility)
- [Operator issues](#operator-issues)

## Worker Pods Stuck in Init:0/1

KubeRay injects an init container into every worker pod that waits for the GCS server on the head pod to be ready.

**Common causes:**

1. **Head GCS server failed** — Check head pod logs:
   ```bash
   kubectl logs <head-pod> -c ray-head
   kubectl exec -it <head-pod> -- cat /tmp/ray/session_latest/logs/gcs_server.out
   ```

2. **`ray` not in PATH** — The init container runs `ray health-check`. Ensure the Ray image has `ray` on PATH.

3. **Wrong cluster domain** — Default is `cluster.local`. If your cluster uses a different domain, set `CLUSTER_DOMAIN` env var on the KubeRay operator:
   ```bash
   # Check your cluster domain
   kubectl exec -it <any-pod> -- cat /etc/resolv.conf
   ```

4. **Init container resource/security conflicts** — The init container inherits `SecurityContext`, `Env`, `VolumeMounts`, and `Resources` from the worker template. This can cause deadlocks in some cases.

**If stuck >2 minutes**, logs are printed to the worker pod:
```bash
kubectl logs <worker-pod> -c wait-gcs-ready
```

**Disable init container injection** (advanced):
Set `ENABLE_INIT_CONTAINER_INJECTION=false` on the KubeRay operator deployment.

## Head Pod Crashes or Restarts

```bash
kubectl describe pod <head-pod>
kubectl logs <head-pod> -c ray-head --previous
```

**Common causes:**
- **OOM killed** — Increase memory limits. Check `kubectl describe pod` for `OOMKilled`.
- **GCS crash** — Check `/tmp/ray/session_latest/logs/gcs_server.err` in the head container.
- **Liveness probe failure** — The head probe hits the dashboard agent. If the agent is overloaded (e.g., many concurrent jobs), it can timeout.

**Impact without GCS fault tolerance:** If the head dies, all workers attempt to reconnect. If reconnect fails after `RAY_gcs_rpc_server_reconnect_timeout_s` (default 60s), workers die too.

## Autoscaler Not Scaling Up

```bash
# Check autoscaler logs (sidecar on head pod)
kubectl logs <head-pod> -c autoscaler

# Check Ray resource demands
kubectl exec -it <head-pod> -- ray status
```

**Common causes:**

1. **Task requires more resources than any worker can provide** — If a task needs 2 GPUs but workers have 1 GPU each, the autoscaler won't scale. Fix the task's resource request or increase worker resources.

2. **maxReplicas reached** — Check `workerGroupSpecs[].maxReplicas`.

3. **Autoscaler not enabled** — Verify `enableInTreeAutoscaling: true` in the RayCluster spec.

4. **Cluster autoscaler not provisioning nodes** — The Ray autoscaler creates pods, but if the K8s cluster doesn't have capacity, pods stay Pending. Check cluster-autoscaler logs.

## Autoscaler Not Scaling Down

- **Idle timeout** — Workers must be idle for `idleTimeoutSeconds` (default 60s) before removal.
- **Detached actors** — Detached actors keep workers alive. Kill unused detached actors.
- **Object store references** — Workers holding object references won't be removed.

## Pods Pending (Not Scheduled)

This is a Kubernetes scheduling issue, not a KubeRay issue:

```bash
kubectl describe pod <pending-pod>
# Look at Events for scheduling failures
```

**Common causes:**
- Insufficient CPU/memory/GPU on cluster nodes
- Taints without matching tolerations
- Node affinity/selector mismatch
- PVC binding issues

## RayJob Stuck or Failed

```bash
kubectl describe rayjob <name>
kubectl get rayjob <name> -o yaml
```

**Check status fields:**
- `jobStatus`: Ray job status (PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED)
- `jobDeploymentStatus`: Kubernetes-level status (Initializing, Running, Complete, Failed, Suspended)

**Common issues:**

1. **Submitter job failed** — Check the submitter pod:
   ```bash
   kubectl get pods | grep <rayjob-name>
   kubectl logs <rayjob-submitter-pod>
   ```

2. **activeDeadlineSeconds exceeded** — Job took too long, marked as Failed with `DeadlineExceeded`.

3. **Ray cluster never became ready** — Check RayCluster status and pod events.

4. **entrypoint script error** — Check Ray job logs via dashboard or:
   ```bash
   kubectl exec -it <head-pod> -- ray job logs <job-id>
   ```

## GCS Fault Tolerance

Without GCS FT, head failure kills the entire cluster. Enable it with Redis:

```yaml
spec:
  headGroupSpec:
    rayStartParams:
      redis-password: "your-password"
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
```

With GCS FT enabled:
- Workers continue serving during head recovery
- State is persisted to Redis
- Head pod restart recovers cluster state

## GPU Multi-Tenancy

Pods without `nvidia.com/gpu` in resource limits may still see all GPUs on the node (`NVIDIA_VISIBLE_DEVICES` defaults to `all`). Fix:

```yaml
env:
- name: NVIDIA_VISIBLE_DEVICES
  value: void    # hides all GPUs from this pod
```

Set this on head pods and CPU-only worker groups when running on GPU nodes.

## Version Compatibility

| KubeRay | Minimum Ray | Notes |
|---|---|---|
| v1.5.x | 2.8.0+ | Recommended: 2.52.0+ for auth support |
| v1.3.0+ | 2.8.0+ | CPU uses requests if limits absent |
| v1.1.x | 2.8.0+ | |
| v1.0.0 | 1.10+ | |

**Known issues:**
- Ray 2.11.0–2.37.0: dashboard agent hangs when jobs created, causing liveness probe failures. Avoid these versions.
- KubeRay ≥ 1.1.0 requires `wget` in Ray container image.
- Only `replicas` field changes are supported at runtime. Other RayCluster field changes may not take effect — recreate the resource.

## Operator Issues

```bash
# Operator logs
kubectl logs -n kuberay-system deploy/kuberay-operator --tail=200

# CRDs installed?
kubectl get crd | grep ray.io

# Reconcile performance (many CRs)
# Increase concurrency:
helm upgrade kuberay-operator kuberay/kuberay-operator --set reconcileConcurrency=10
```

**Changes to RayCluster CR not taking effect:** Only `replicas` field changes are supported at runtime. Other field changes may not take effect — recreate the resource instead.
