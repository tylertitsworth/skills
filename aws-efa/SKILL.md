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

## Communication Backends: Libfabric vs UCX

Two communication libraries sit between applications and EFA hardware. Understanding their differences matters for configuring NCCL (training), NixlConnector (disaggregated serving), and DeepEP (MOE all2all).

### Libfabric (OFI)

Libfabric is the **native** communication library for EFA. The `aws-ofi-nccl` plugin bridges NCCL → libfabric → EFA.

| Aspect | Details |
|---|---|
| **EFA integration** | Native — EFA provider is maintained by AWS in the libfabric tree |
| **GPUDirect RDMA** | Supported via `FI_EFA_USE_DEVICE_RDMA` (auto-enabled on libfabric ≥ 1.18.0) |
| **Used by** | NCCL (via aws-ofi-nccl), training frameworks (PyTorch DDP, FSDP, Megatron-LM) |
| **Transport selection** | Automatic — libfabric detects EFA provider. No manual transport config needed. |
| **Key env vars** | `FI_PROVIDER` (auto), `FI_EFA_USE_DEVICE_RDMA` (auto ≥ 1.18.0), `FI_EFA_FORK_SAFE` (auto ≥ aws-ofi-nccl 1.7.0) |
| **Multi-path** | SRD protocol provides built-in multi-path routing across AWS fabric |

### UCX (Unified Communication X)

UCX is a general-purpose communication library supporting multiple transports. It's the **default backend for NIXL/NixlConnector** in vLLM disaggregated serving.

| Aspect | Details |
|---|---|
| **EFA integration** | Via libfabric shim — UCX calls libfabric's EFA provider internally |
| **GPUDirect RDMA** | Supported when `kv_buffer_device="cuda"` and EFA GPUDirect is available |
| **Used by** | vLLM NixlConnector, NIXL library, some MPI implementations |
| **Transport selection** | `UCX_TLS=all` auto-selects best available (RDMA > shared memory > TCP). Can specify: `rc,ud,sm` for InfiniBand, or let auto-detection find EFA via libfabric. |
| **Key env vars** | `UCX_TLS` (transport selection), `UCX_NET_DEVICES` (NIC selection), `UCX_LOG_LEVEL` (debugging) |
| **Multi-path** | Relies on underlying provider — gets SRD multi-path when using EFA |

### Which Backend for What

| Workload | Backend | Why |
|---|---|---|
| **NCCL collectives** (training, all-reduce) | Libfabric (via aws-ofi-nccl) | Native EFA support, mature, AWS-optimized |
| **KV cache transfer** (NixlConnector PD) | UCX (via NIXL) | NIXL uses UCX as default; async send/recv model fits PD pattern |
| **MOE all2all** (DeepEP, pplx) | Libfabric or UCX | Depends on backend: `deepep_*` uses its own transport; `pplx` uses NCCL → libfabric |
| **Custom connectors** | Either | P2pNcclConnector uses NCCL (→ libfabric); LMCache uses NIXL (→ UCX) |

### Key Difference: NCCL env vars don't apply to NixlConnector

When using NixlConnector for PD serving, `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, etc. have **no effect**. Configure UCX variables instead:

| PD Serving (NixlConnector) | Training (NCCL) |
|---|---|
| `UCX_TLS=all` | `FI_PROVIDER=efa` (auto) |
| `UCX_NET_DEVICES=all` | `NCCL_SOCKET_IFNAME` (auto) |
| `UCX_LOG_LEVEL=info` (debug) | `NCCL_DEBUG=INFO` (debug) |

## EFA vs InfiniBand for Disaggregated Serving and MOE

EFA and InfiniBand both provide RDMA for KV cache transfer and expert all2all, but differ fundamentally in architecture:

### Protocol and Routing

| | EFA (SRD) | InfiniBand (IB) |
|---|---|---|
| **Protocol** | Scalable Reliable Datagram — connectionless, multi-path | Queue Pair — connection-oriented, single-path per QP |
| **Routing** | Packet spraying across AWS fabric — automatic load balancing | Subnet-managed routing — deterministic paths, manual ECMP for multi-path |
| **Congestion control** | Built into SRD, tuned for ML collectives | Hardware-based (ECN), requires switch configuration |
| **Topology** | Flat — any instance can reach any other at full bandwidth (within placement group) | Fat-tree or rail-optimized — bandwidth depends on switch tiers |

### Implications for PD Serving

| Concern | EFA | InfiniBand |
|---|---|---|
| **KV transfer bandwidth** | Consistent — SRD multi-path prevents hotspots. p5: 3200 Gbps aggregate. | High peak, but single-QP transfers use one path. Multiple QPs or RDMA-CM needed for multi-path. |
| **Tail latency** | Lower variance — packet spraying smooths bursts | Lower absolute minimum latency, but higher variance under contention |
| **Scaling** | Easier — no subnet manager, no switch config. Placement groups handle locality. | Requires subnet manager, careful topology planning, switch firmware management |
| **GPUDirect RDMA** | Supported on p4d/p5/p5e (NVIDIA peer memory + EFA driver) | Native support on all RDMA NICs (ConnectX-6/7) |
| **NixlConnector** | UCX auto-detects EFA via libfabric. `UCX_TLS=all` works. | UCX native IB support. `UCX_TLS=rc` for reliable connected. `UCX_NET_DEVICES=mlx5_0:1`. |

### Implications for MOE All2All

| Concern | EFA | InfiniBand |
|---|---|---|
| **All2all pattern** | Good — SRD handles many-to-many well due to packet spraying | Good — but all2all generates N² flows; fat-tree bandwidth must support it |
| **DeepEP backends** | `deepep_high_throughput` and `deepep_low_latency` work over EFA via NCCL → libfabric | Native IB support; `deepep_low_latency` benefits from IB's lower absolute latency |
| **pplx backend** | Works via NCCL → aws-ofi-nccl → libfabric → EFA | Works via NCCL → IB verbs directly |
| **Expert migration (EPLB)** | Fast — equal bandwidth to all nodes in placement group | Depends on topology — migration between nodes on different switches may be slower |

### When EFA Wins

- **Cloud-native deployments** — no switch management, no subnet manager, automatic scaling
- **Large clusters** — SRD multi-path scales better than fat-tree at 100+ nodes
- **Mixed workloads** — PD + MOE + training can coexist on EFA without careful traffic engineering

### When InfiniBand Wins

- **Absolute lowest latency** — IB RDMA has ~1-2 μs vs EFA's ~3-5 μs for small messages
- **On-prem** — IB is the standard; EFA is AWS-only
- **Established tooling** — IB has decades of RDMA ecosystem (perftest, ibstat, opensm)

## Concurrent Traffic Patterns: PD + MOE on EFA

MOE models with disaggregated serving generate overlapping network traffic:

| Traffic Type | Pattern | Bandwidth Need | EFA Interfaces Used |
|---|---|---|---|
| **KV cache transfer** (PD) | Point-to-point: prefill → decode | High burst (GB-scale per request) | UCX selects available EFA NICs |
| **All2all** (MOE expert routing) | Many-to-many: all GPUs exchange tokens | Sustained, proportional to batch × experts | NCCL maps NICs to GPUs |
| **NCCL collectives** (attention layers) | AllReduce within TP group | Moderate | Shared with all2all |

### Bandwidth Planning

| Instance | Aggregate BW | KV Transfer Budget | All2All Budget | Sufficient? |
|---|---|---|---|---|
| p4d (4 EFA, 400 Gbps) | ~50 GB/s | ~15 GB/s | ~35 GB/s | ⚠️ Tight for large MOE + PD |
| p5 (32 EFA, 3200 Gbps) | ~400 GB/s | ~50 GB/s | ~350 GB/s | ✅ Comfortable |
| p5e (32 EFA, 3200 Gbps) | ~400 GB/s | ~50 GB/s | ~350 GB/s | ✅ Comfortable |

On p4d, if running MOE + PD simultaneously, KV transfer and all2all contend for the same 4 EFA interfaces. Monitor with `NCCL_DEBUG=INFO` and `UCX_LOG_LEVEL=info` to identify bottlenecks. Consider dedicating specific EFA interfaces to each traffic type via `UCX_NET_DEVICES` and `NCCL_IB_HCA` if contention is observed.

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
