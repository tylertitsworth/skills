# Kueue Troubleshooting

Debugging pending workloads, admission failures, and preemption issues.

## Table of Contents

- [Find the Workload for a Job](#find-the-workload-for-a-job)
- [Why is my job pending?](#why-is-my-job-pending)
- [Common admission failure reasons](#common-admission-failure-reasons)
- [Preemption debugging](#preemption-debugging)
- [Controller diagnostics](#controller-diagnostics)

## Find the Workload for a Job

Kueue creates a Workload object for each job. Find it:

```bash
# From job events
kubectl describe job -n <ns> <job-name> | grep CreatedWorkload

# By job UID label
JOB_UID=$(kubectl get job -n <ns> <job-name> -o jsonpath='{.metadata.uid}')
kubectl get workloads -n <ns> -l "kueue.x-k8s.io/job-uid=$JOB_UID"

# By name pattern (job-<name>-<hash>)
kubectl get workloads -n <ns> | grep <job-name>
```

Then inspect:

```bash
kubectl get workload -n <ns> <workload-name> -o yaml
```

## Why Is My Job Pending?

Check the Workload `.status.conditions`:

### Insufficient quota

```yaml
status:
  conditions:
  - message: "couldn't assign flavors to pod set main: insufficient quota for
      cpu in flavor default-flavor in ClusterQueue"
    reason: Pending
    status: "False"
    type: QuotaReserved
```

**Fix:** Either wait for running jobs to complete, increase `nominalQuota` in the ClusterQueue, or enable borrowing in a cohort.

### Missing resource definition

If the ClusterQueue doesn't define a resource the job requests (e.g., job requests `nvidia.com/gpu` but ClusterQueue only defines `cpu` and `memory`), admission fails silently.

**Fix:** Add the missing resource to `coveredResources` and the flavor's `resources` list.

### LocalQueue doesn't exist or is misconfigured

```yaml
status:
  conditions:
  - message: "LocalQueue user-queue doesn't exist"
    reason: Inadmissible
    status: "False"
    type: QuotaReserved
```

**Fix:** Create the LocalQueue in the job's namespace, or fix the `kueue.x-k8s.io/queue-name` label.

### ClusterQueue is stopped

```yaml
status:
  conditions:
  - message: "ClusterQueue cluster-queue is inactive"
    reason: Inadmissible
    status: "False"
    type: QuotaReserved
```

**Fix:** Check `kubectl get clusterqueue <name> -o yaml` — look for `.spec.stopPolicy`. Remove it or set to `None`.

### Blocked by StrictFIFO

With `StrictFIFO` queueing, a large job at the head of the queue blocks all jobs behind it, even if smaller jobs could fit.

**Fix:** Switch to `BestEffortFIFO` or increase quota so the head-of-line job can be admitted.

### Pending admission checks

```yaml
status:
  admissionChecks:
  - name: dws-prov
    state: Pending
```

**Fix:** Check the admission check controller (ProvisioningRequest, MultiKueue, etc.) for why the check hasn't passed.

## Common Admission Failure Reasons

| Symptom | Check | Fix |
|---|---|---|
| Job stays suspended forever | `kubectl get workload -n <ns>` — no workload exists | Verify `kueue.x-k8s.io/queue-name` label is set |
| Workload exists but no conditions | StrictFIFO, not at head of queue | Wait, or switch to BestEffortFIFO |
| `insufficient quota` | ClusterQueue quota exhausted | Wait, increase quota, add borrowing, or enable preemption |
| `Inadmissible` | LocalQueue/ClusterQueue misconfigured | Check queue names, namespace selectors |
| Job admitted but pods not scheduling | Pod-level issue, not Kueue | Check `kubectl describe pod` for scheduler events |
| Workload admitted then evicted | Preemption occurred | Check Workload events for preemption reason |

## Preemption Debugging

When a higher-priority workload arrives and preempts:

```bash
# Check workload events
kubectl describe workload -n <ns> <name>

# Look for eviction conditions
kubectl get workload -n <ns> <name> -o jsonpath='{.status.conditions[?(@.type=="Evicted")]}'
```

Preempted workloads get re-queued automatically. The Workload status shows:

```yaml
conditions:
- message: "Preempted to accommodate a higher priority Workload"
  reason: Preempted
  status: "True"
  type: Evicted
```

### Check what's using quota

```bash
# See all admitted workloads in a ClusterQueue
kubectl get workloads -A -o custom-columns=\
  NAME:.metadata.name,\
  NS:.metadata.namespace,\
  QUEUE:.spec.queueName,\
  ADMITTED:.status.conditions[0].status,\
  AGE:.metadata.creationTimestamp

# ClusterQueue usage summary
kubectl describe clusterqueue <name>
# Look at status.flavorsReservation and status.admittedWorkloads
```

## Controller Diagnostics

```bash
# Controller logs
kubectl logs -n kueue-system deploy/kueue-controller-manager --tail=100

# Controller health
kubectl get deploy -n kueue-system

# Webhook issues (admission errors on job creation)
kubectl logs -n kueue-system deploy/kueue-controller-manager | grep -i webhook

# Kueue CRDs installed?
kubectl get crd | grep kueue
```

### Common controller issues

| Issue | Symptom | Fix |
|---|---|---|
| Controller not running | No workloads created for jobs | `kubectl get deploy -n kueue-system`, check pod logs |
| Webhook cert expired | Jobs fail to create with admission errors | Restart controller or reconfigure cert management |
| CRDs missing | `error: the server doesn't have a resource type "clusterqueues"` | Re-install Kueue manifests |
| Version mismatch | Unexpected behavior after upgrade | Ensure CRDs and controller are same version |
