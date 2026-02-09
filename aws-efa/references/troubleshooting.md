# EFA Troubleshooting

## Diagnostic Commands

```bash
# Check EFA device plugin is running
kubectl get ds -n kube-system aws-efa-k8s-device-plugin-daemonset

# Check EFA resources on nodes
kubectl describe node <node> | grep -A2 "vpc.amazonaws.com/efa"

# Check NCCL sees EFA (run inside training pod)
# Set NCCL_DEBUG=INFO in pod env, then check logs for:
#   "NET/OFI Selected Provider is efa"

# Verify libfabric detects EFA (inside pod)
fi_info -p efa

# Check EFA interfaces on the host (via node shell)
# Look for rdma devices
ls /dev/infiniband/
```

## Common Issues

### NCCL Falls Back to TCP/Sockets

**Symptoms**: `NCCL INFO NET/OFI Selected Provider is sockets` or `tcp` in logs. Bandwidth ~10-25 GB/s instead of expected 50-400 GB/s.

**Causes and fixes**:

| Cause | Fix |
|-------|-----|
| EFA interfaces not allocated to pod | Ensure pod requests `vpc.amazonaws.com/efa` in resources |
| Partial EFA allocation | Request ALL EFA interfaces (32 on p5, 4 on p4d) |
| Security group blocking | Verify self-referencing SG rule (all protocols, all ports) |
| aws-ofi-nccl not installed in container | Use NGC or AWS DLC base images |
| libfabric EFA provider missing | Verify `fi_info -p efa` works inside container |
| Wrong container base image | Use images with EFA support pre-installed |

### Pod Stuck Pending for EFA Resources

**Symptoms**: Pod in `Pending` state, events show `Insufficient vpc.amazonaws.com/efa`.

**Causes**:
- Node doesn't have EFA resources → check device plugin DaemonSet
- Another pod already consumed all EFA interfaces → only one training job per node
- Node pool scaled to 0 → trigger scale-up by creating the pod

### Low Bandwidth in NCCL Tests

**Symptoms**: NCCL all-reduce perf test shows bandwidth well below expected.

| Check | Expected | Fix if Wrong |
|-------|----------|-------------|
| Placement group | All nodes in same cluster placement group | Recreate nodegroup with placement group |
| Availability Zone | All nodes in same AZ | Single AZ in nodegroup config |
| EFA count | 32 on p5, 4 on p4d | Request all EFA interfaces in pod spec |
| Huge Pages | Allocated and mounted | Check `hugepages-2Mi` in resources + volume mount |
| Security group | Self-referencing rule | Add inbound/outbound all-traffic from self |

### NCCL Timeout During Training

**Symptoms**: `NCCL WARN ... Timeout` or training hangs.

**Common causes**:
1. **Nodes in different AZs** — placement group ensures same AZ; verify all workers are co-located
2. **Security group issue** — EFA traffic blocked between nodes
3. **Straggler node** — one node slower than others; check GPU health with `nvidia-smi`
4. **OOM on one worker** — check for OOMKilled in pod events
5. **Shared memory too small** — mount `/dev/shm` with sufficient size:
   ```yaml
   volumes:
     - name: shm
       emptyDir:
         medium: Memory
         sizeLimit: "64Gi"
   volumeMounts:
     - name: shm
       mountPath: /dev/shm
   ```

### EFA Device Plugin Not Starting

**Symptoms**: DaemonSet pods in CrashLoopBackOff or not scheduled.

| Cause | Fix |
|-------|-----|
| Non-EFA instance type | Plugin only runs on EFA-capable instances; use `nodeSelector` |
| Old plugin version for new instance | p6-b200 requires device plugin ≥ v0.5.6 |
| Incompatible AMI | Ensure EKS GPU AMI (AL2 or AL2023 NVIDIA variant) |

### GPUDirect RDMA Not Working

**Symptoms**: EFA is selected but performance is lower than expected; NCCL falls back to non-GDR path.

**Check**: Look for `NET/OFI GDR` messages in NCCL debug output.

| Cause | Fix |
|-------|-----|
| nvidia_peermem not loaded | Check `lsmod | grep nvidia_peermem` on node; use EKS GPU AMI |
| Old EFA driver | Update EKS node AMI to latest version |
| CUDA version mismatch | Ensure container CUDA version matches host driver compatibility |

## NCCL Debug Output Reference

Key log lines to look for with `NCCL_DEBUG=INFO`:

```
# Good — EFA active
NCCL INFO NET/OFI Using aws-ofi-nccl 1.x.x
NCCL INFO NET/OFI Selected Provider is efa
NCCL INFO NET/OFI Running on ... platform, ... domain
NCCL INFO Channel 00/32 : 0[0] -> 1[0] [send] via NET/OFI/0

# Bad — TCP fallback
NCCL INFO NET/Socket : Using [eth0]
NCCL INFO Channel 00/01 : 0[0] -> 1[0] [send] via NET/Socket/0

# GPUDirect RDMA active
NCCL INFO NET/OFI GDR : RDMA enabled for GPU 0

# Topology detection
NCCL INFO Trees [0] ... (shows tree topology for collectives)
```

## Running NCCL Tests

To validate EFA setup, run the standard NCCL all-reduce perf test. Use the Kubeflow MPI Operator with the `public.ecr.aws/hpc-cloud/nccl-tests:latest` image. Each worker should request all GPUs, all EFA interfaces, and Huge Pages for the instance type.

Key test parameters:
- `-b 8 -e 16G -f 2 -g 1`: Test from 8 bytes to 16 GB, doubling each step, 1 GPU per process
- `-n 100`: Number of iterations per message size
- `-w 50`: Warmup iterations

Compare results against expected bus bandwidth for your instance type (see main SKILL.md).
