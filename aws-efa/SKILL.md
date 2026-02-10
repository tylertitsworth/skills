---
name: aws-efa
description: AWS Elastic Fabric Adapter (EFA) — architecture, SRD protocol, GPUDirect RDMA, NCCL integration, EKS node configuration, and environment tuning. Use when understanding or configuring EFA for distributed GPU training on EKS, troubleshooting multi-node communication, tuning NCCL over EFA, or configuring EFA for disaggregated serving KV cache transfer.
---

# AWS Elastic Fabric Adapter (EFA)

## What EFA Is

EFA is a network interface that combines a standard Elastic Network Adapter (ENA) with an **OS-bypass** interface using the AWS Scalable Reliable Datagram (SRD) protocol. The OS-bypass path allows applications (via libfabric) to communicate directly with the network hardware, skipping the kernel network stack entirely.

**On GPU instances, EFA enables GPUDirect RDMA**: NCCL collective operations move data directly between GPU memory across nodes without copying through CPU memory.

### Protocol Stack

```
Application (NCCL)
    ↓
aws-ofi-nccl plugin (NCCL → libfabric translation)
    ↓
libfabric (EFA provider)
    ↓
EFA OS-bypass hardware interface
    ↓
SRD protocol (AWS Scalable Reliable Datagram)
    ↓
AWS network fabric
```

**SRD** is a custom AWS transport protocol optimized for HPC/ML. It provides:
- Reliable delivery with minimal overhead
- Multi-path routing across the AWS network (unlike TCP which uses single paths)
- Packet spraying across multiple network paths for higher aggregate throughput
- Built-in congestion control tuned for collective operations

### EFA vs ENA vs EFA-Only

| Interface Type | IP Stack | OS-Bypass | Use |
|---------------|----------|-----------|-----|
| ENA | Yes | No | Standard networking (pods, services) |
| EFA | Yes | Yes | Primary interface (network card 0) — both IP + RDMA |
| EFA-only | No | Yes | Additional interfaces (cards 1+) — RDMA traffic only |

**EFA-only interfaces** (available on p5, p5e, trn2) have no IP address — they carry only SRD/RDMA traffic with lower overhead. Network card 0 must always be a full EFA (with ENA) for standard connectivity.

## Instance Types and Topology

| Instance | GPUs | EFA Interfaces | Aggregate Bandwidth | NVSwitch |
|----------|------|---------------|---------------------|----------|
| p4d.24xlarge | 8× A100 40GB | 4 | 400 Gbps | Yes (intra-node) |
| p4de.24xlarge | 8× A100 80GB | 4 | 400 Gbps | Yes |
| p5.48xlarge | 8× H100 80GB | 32 | 3200 Gbps | Yes |
| p5e.48xlarge | 8× H200 141GB | 32 | 3200 Gbps | Yes |
| p5en.48xlarge | 8× H200 141GB | 32 | 3200 Gbps | Yes |
| p6-b200.48xlarge | 8× B200 | 32 | 3200 Gbps | Yes |
| trn1.32xlarge | 16× Trainium | 8 | 800 Gbps | N/A |
| trn2.48xlarge | 16× Trainium2 | 16 | 1600 Gbps | N/A |

### Topology: How NCCL Uses EFA

Within a node, GPUs communicate via NVSwitch (NVLink). Across nodes, NCCL maps EFA interfaces to GPUs:

- **p4d** (4 EFA, 8 GPU): Each EFA serves 2 GPUs. NCCL uses 4 network channels.
- **p5** (32 EFA, 8 GPU): Each GPU gets 4 dedicated EFA interfaces. NCCL uses up to 32 channels.

NCCL auto-detects this topology. The number of channels (`NCCL_MIN_NCHANNELS`) defaults based on EFA count — don't override unless benchmarking shows improvement.

## How GPUDirect RDMA Works on EFA

1. GPU allocates memory for send/receive buffers
2. NCCL calls aws-ofi-nccl, which calls libfabric
3. Libfabric's EFA provider registers GPU memory with the EFA device
4. EFA hardware DMA-reads directly from GPU memory (no CPU copy)
5. SRD packets are sent across the AWS fabric
6. Remote EFA hardware DMA-writes directly into remote GPU memory

**Requirements for GPUDirect RDMA**:
- NVIDIA peer memory module loaded (included in EKS GPU AMIs)
- EFA driver with GPUDirect support (included in EKS AMIs ≥ 2023)
- Huge Pages allocated for EFA internal buffers

## EKS Node Configuration

### eksctl

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: ml-cluster
  region: us-west-2
  version: "1.31"
iam:
  withOIDC: true
availabilityZones: ["us-west-2a", "us-west-2b"]
managedNodeGroups:
  - name: gpu-efa
    instanceType: p5.48xlarge
    minSize: 0
    desiredCapacity: 2
    maxSize: 8
    availabilityZones: ["us-west-2a"]    # Single AZ for placement group
    volumeSize: 500
    privateNetworking: true
    efaEnabled: true                      # Handles SG, placement group, device plugin
```

When `efaEnabled: true`, eksctl automatically:
1. Creates an **EFA security group** (all traffic between members)
2. Creates a **cluster placement group** (co-locates instances)
3. Deploys the **AWS EFA device plugin** DaemonSet
4. Deploys the **NVIDIA device plugin** (Amazon Linux 2)
5. Configures all EFA interfaces on the launch template

### Terraform

```hcl
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"
  cluster_name = "ml-cluster"
  enable_efa_support = true
  eks_managed_node_groups = {
    gpu-efa = {
      ami_type           = "AL2023_x86_64_NVIDIA"
      instance_types     = ["p5.48xlarge"]
      enable_efa_support = true
      subnet_ids         = [module.vpc.private_subnets[0]]  # Single AZ
    }
  }
}
```

## EFA Device Plugin

The AWS EFA Kubernetes device plugin exposes `vpc.amazonaws.com/efa` as a schedulable resource.

```bash
# Verify
kubectl get ds -n kube-system aws-efa-k8s-device-plugin-daemonset
kubectl describe node <node> | grep "vpc.amazonaws.com/efa"
# → vpc.amazonaws.com/efa: 32   (p5.48xlarge)
```

For p6-b200 instances, use device plugin **v0.5.6 or later**.

### Pod Resource Requests

When using Ray Train via RayJob/RayCluster CRDs, the worker pod template should request all EFA interfaces:

```yaml
resources:
  requests:
    nvidia.com/gpu: "8"
    vpc.amazonaws.com/efa: "32"        # All 32 on p5
    hugepages-2Mi: "5120Mi"
    memory: "128Gi"
  limits:
    nvidia.com/gpu: "8"
    vpc.amazonaws.com/efa: "32"
    hugepages-2Mi: "5120Mi"
    memory: "128Gi"
```

**Always request all EFA interfaces** — partial allocation causes suboptimal NCCL topology mapping and degraded performance.

## Security Groups

EFA requires a security group allowing **all traffic between members**:

```
Inbound:  All protocols, all ports, source = self (same SG)
Outbound: All protocols, all ports, destination = self (same SG)
```

This is separate from the EKS cluster security group. Both must be attached to EFA nodes. eksctl and the Terraform EKS module create this automatically with `efaEnabled`/`enable_efa_support`.

## Placement Groups

All EFA nodes must be in a **cluster placement group** in a **single Availability Zone**. This ensures instances are physically co-located on the same network spine for lowest latency.

```bash
aws ec2 create-placement-group --group-name ml-pg --strategy cluster
```

**Capacity limitations**: Cluster placement groups can fail to launch if insufficient capacity exists in the AZ. Use `minSize: 0` and scale up on demand.

## Huge Pages

EFA requires 2MiB Huge Pages for internal buffers. EKS GPU AMIs pre-allocate **5128 × 2MiB** (~10 GiB).

Pods must request Huge Pages and mount the hugepages volume:

```yaml
resources:
  requests:
    hugepages-2Mi: "5120Mi"
  limits:
    hugepages-2Mi: "5120Mi"
volumes:
  - name: hugepages
    emptyDir:
      medium: HugePages
volumeMounts:
  - name: hugepages
    mountPath: /dev/hugepages
```

For Bottlerocket nodes, configure via settings:

```yaml
bottlerocket:
  settings:
    kernel:
      sysctl:
        "vm.nr_hugepages": "5128"
```

## NCCL over EFA: aws-ofi-nccl

The **aws-ofi-nccl** plugin bridges NCCL and libfabric. It's pre-installed in:
- NVIDIA NGC containers (`nvcr.io/nvidia/pytorch:*`)
- AWS Deep Learning Containers (`763104351884.dkr.ecr.*.amazonaws.com/pytorch-training:*`)

### Environment Variables

Modern software stacks (aws-ofi-nccl ≥ 1.7.0, libfabric ≥ 1.18.0) require **minimal configuration** — most legacy env vars are auto-detected:

| Variable | Status | Notes |
|----------|--------|-------|
| `FI_PROVIDER=efa` | **Not needed** | Auto-detected by libfabric |
| `FI_EFA_USE_DEVICE_RDMA=1` | **Not needed** (libfabric ≥ 1.18.0) | Was needed for older stacks; harmless to set |
| `FI_EFA_FORK_SAFE=1` | **Not needed** (aws-ofi-nccl ≥ 1.7.0) | Legacy |
| `NCCL_MIN_NCHANNELS` | Leave at default | Auto-set based on NIC count (8 for p4d, higher for p5) |
| `NCCL_ALGO` | Optional | Override collective algorithm (Tree, Ring, CollnetDirect, CollnetChain, NVLS) |
| `NCCL_PROTO` | Optional | Override protocol (Simple, LL, LL128) |
| `NCCL_CROSS_NIC` | `0` (default) | Set to `1` to allow cross-NIC communication patterns |
| `NCCL_NET_GDR_LEVEL` | Auto | GPUDirect RDMA level; auto-detected |
| `NCCL_DEBUG` | `WARN` (default) | Set to `INFO` for debugging, `TRACE` for verbose |
| `NCCL_TOPO_DUMP_FILE` | Not set | Dump detected topology to file for inspection |
| `NCCL_SOCKET_IFNAME` | Auto | Network interface for OOB (out-of-band) communication |

### Container Image Requirements

Your training container must include:
1. **NCCL** (typically bundled with PyTorch/CUDA)
2. **aws-ofi-nccl** plugin
3. **libfabric** with EFA provider
4. **EFA installer components** (efa_installer or individual packages)

AWS Deep Learning Containers and NGC PyTorch containers include all of these. If building a custom image:

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.12-py3
# EFA components already included in NGC containers

# Or for custom builds:
# RUN apt-get update && apt-get install -y libfabric-dev
# RUN git clone https://github.com/aws/aws-ofi-nccl && cd aws-ofi-nccl && ...
```

### Verifying EFA is Active

In NCCL debug output (`NCCL_DEBUG=INFO`), look for:

```
NCCL INFO NET/OFI Using aws-ofi-nccl ...
NCCL INFO NET/OFI Selected Provider is efa
```

**Bad signs** — EFA not being used:
- `Selected Provider is sockets` or `tcp` → EFA driver not available
- `No EFA devices found` → Device plugin not running or EFA interfaces not allocated
- Bandwidth in NCCL all-reduce test ≪ expected → Check placement group, security group

### Expected Bandwidth

| Instance | Message Size | Expected Bus BW | Notes |
|----------|-------------|-----------------|-------|
| p4d.24xlarge | 1 GB+ | ~50-80 GB/s | 4 EFA × 100 Gbps |
| p5.48xlarge | 1 GB+ | ~300-400 GB/s | 32 EFA × 100 Gbps |

If observed bandwidth is ~10-25 GB/s (TCP-level), EFA is not being used correctly.

## EFA-Only Interfaces

For p5/p5e/trn2, interfaces on network cards 1+ can be **EFA-only** (no IP stack):

- Lower overhead — no TCP/IP processing on RDMA paths
- All bandwidth dedicated to collective operations
- Requires: VPC CNI ≥ 1.18.5, Amazon Linux 2 AMI ≥ v20240928
- Cannot be configured via eksctl — requires custom launch template with `InterfaceType: efa-only`

## EFA for Disaggregated Serving (KV Cache Transfer)

Disaggregated prefill-decode (PD) serving requires high-bandwidth, low-latency KV cache transfer between prefill and decode instances. EFA provides the RDMA transport that makes this practical at scale.

### Why EFA Matters for PD

KV cache transfer size scales with model and sequence length. For Llama 70B at 8K context in bf16, a single request's KV cache is **~5 GB**. At production request rates, this demands sustained multi-GB/s network throughput that TCP cannot deliver reliably.

| Transport | Bandwidth | Latency | Production? |
|---|---|---|---|
| TCP/Ethernet | ~10-12 GB/s (100 GbE) | ~100-500 μs | Marginal |
| EFA + UCX (RDMA) | ~50-80 GB/s (p4d), ~300+ GB/s (p5) | ~5-20 μs | ✅ |
| EFA + GPUDirect RDMA | Same bandwidth, zero-copy | ~2-10 μs | ✅ Best |

### UCX over EFA Configuration

vLLM's NixlConnector uses UCX as the default transport. UCX auto-detects EFA via libfabric:

```bash
# Environment for vLLM prefill/decode instances on EFA nodes
export UCX_TLS=all                    # Let UCX select best transport (will use EFA)
export UCX_NET_DEVICES=all            # Use all available network devices
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600

vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"cuda"}'
```

Setting `kv_buffer_device="cuda"` enables GPUDirect RDMA — KV cache transfers directly between GPU memories without CPU staging.

### Kubernetes Pod Configuration

For PD serving pods on EFA nodes, request EFA interfaces just like training pods:

```yaml
# Both prefill and decode pods need EFA access
containers:
- name: vllm
  env:
  - name: UCX_TLS
    value: "all"
  - name: UCX_NET_DEVICES
    value: "all"
  - name: VLLM_NIXL_SIDE_CHANNEL_PORT
    value: "5600"
  resources:
    requests:
      nvidia.com/gpu: "4"
      vpc.amazonaws.com/efa: "32"       # Request all EFA interfaces
      hugepages-2Mi: "5120Mi"
    limits:
      nvidia.com/gpu: "4"
      vpc.amazonaws.com/efa: "32"
      hugepages-2Mi: "5120Mi"
  volumeMounts:
  - name: hugepages
    mountPath: /dev/hugepages
```

### MOE + PD on EFA

MOE models with expert parallelism generate two concurrent network traffic patterns over EFA:
1. **All2all expert routing** — tokens routed between GPUs for expert computation
2. **KV cache transfer** — PD KV cache between prefill and decode instances

On p5 instances with 32 EFA interfaces, bandwidth is sufficient for both. On p4d (4 EFA), contention is possible — monitor with NCCL debug logging.

### Verifying KV Transfer Uses EFA

Set `UCX_LOG_LEVEL=info` to confirm UCX selects EFA:

```bash
UCX_LOG_LEVEL=info vllm serve ... 2>&1 | grep -i "efa\|transport"
# Should show: "using transport: efa" or similar
```

If UCX falls back to TCP (`using transport: tcp`), check:
1. EFA device plugin is running and EFA interfaces are allocated to the pod
2. `vpc.amazonaws.com/efa` resource is requested in pod spec
3. Security group allows all traffic between PD pods
4. Placement group co-locates prefill and decode nodes

## Cross-References

- [aws-fsx](../aws-fsx/) — FSx storage for training data on EFA-enabled nodes
- [pytorch](../pytorch/) — PyTorch distributed training over EFA
- [fsdp](../fsdp/) — FSDP distributed training using EFA for all-reduce
- [megatron-lm](../megatron-lm/) — Megatron-LM multi-node training over EFA
- [ray-train](../ray-train/) — Ray Train distributed jobs on EFA-enabled clusters
- [vllm](../vllm/) — vLLM disaggregated serving using EFA for KV cache transfer
- [ray-serve](../ray-serve/) — Ray Serve PD serving on EFA-enabled clusters

## Reference

- [EFA docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [EFA EKS best practices](https://aws.github.io/aws-eks-best-practices/networking/efa/)
- [aws-ofi-nccl GitHub](https://github.com/aws/aws-ofi-nccl)
- `references/troubleshooting.md` — NCCL debug, diagnostics, bandwidth expectations
- `scripts/check_efa.sh` — verify EFA device availability, libfabric provider, GPU, and NCCL config on a node
- `assets/architecture.md` — EFA network stack, GPUDirect RDMA flow, and EKS multi-node topology diagrams
