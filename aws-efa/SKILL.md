---
name: aws-efa
description: AWS Elastic Fabric Adapter (EFA) on EKS — device plugin, node group configuration, security groups, placement groups, Huge Pages, and NCCL integration. Use when setting up EFA-enabled GPU nodes on EKS for distributed training, configuring the EFA device plugin, or troubleshooting multi-node NCCL communication.
---

# AWS EFA on EKS

## What EFA Does

Elastic Fabric Adapter (EFA) is a network interface that provides OS-bypass hardware via the AWS Scalable Reliable Datagram (SRD) protocol. On GPU instances, EFA enables GPUDirect RDMA — NCCL collective operations communicate directly between GPUs across nodes without going through the CPU.

**Result**: Multi-node distributed training scales near-linearly instead of being bottlenecked by TCP networking.

## Instance Types

| Instance | GPUs | EFA Interfaces | Network Bandwidth | GPU Type |
|----------|------|---------------|-------------------|----------|
| p4d.24xlarge | 8× A100 40GB | 4 | 400 Gbps | NVIDIA A100 |
| p4de.24xlarge | 8× A100 80GB | 4 | 400 Gbps | NVIDIA A100 |
| p5.48xlarge | 8× H100 80GB | 32 | 3200 Gbps | NVIDIA H100 |
| p5e.48xlarge | 8× H200 141GB | 32 | 3200 Gbps | NVIDIA H200 |
| p5en.48xlarge | 8× H200 141GB | 32 | 3200 Gbps | NVIDIA H200 |
| p6-b200.48xlarge | 8× B200 | 32 | 3200 Gbps | NVIDIA B200 |
| trn1.32xlarge | 16× Trainium | 8 | 800 Gbps | AWS Trainium |
| trn2.48xlarge | 16× Trainium2 | 16 | 1600 Gbps | AWS Trainium2 |

Each EFA interface occupies one network card. Network card 0 is always a standard ENA+EFA interface; additional interfaces can be **EFA-only** (no IP, lower overhead).

## EKS Node Group Setup

### eksctl (Recommended)

```yaml
# efa-cluster.yaml
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
    availabilityZones: ["us-west-2a"]   # Single AZ for placement group
    volumeSize: 500
    privateNetworking: true
    efaEnabled: true                     # Enables EFA interfaces + device plugin
    # amiFamily: Bottlerocket           # Optional: Bottlerocket ≥ 1.28.0
```

```bash
eksctl create nodegroup -f efa-cluster.yaml
```

When `efaEnabled: true`, eksctl automatically:
1. Creates an **EFA-enabled security group** (allows all traffic within the group)
2. Creates a **cluster placement group** (co-locates instances for lowest latency)
3. Installs the **NVIDIA device plugin** (for Amazon Linux 2 AMIs)
4. Deploys the **AWS EFA Kubernetes device plugin** DaemonSet
5. Configures all available EFA interfaces on the launch template

### Bottlerocket

Bottlerocket ≥ 1.28.0 supports EFA natively. Configure Huge Pages via settings:

```yaml
managedNodeGroups:
  - name: gpu-efa-bottlerocket
    instanceType: p5.48xlarge
    efaEnabled: true
    amiFamily: Bottlerocket
    bottlerocket:
      enableAdminContainer: true
      settings:
        kernel:
          sysctl:
            "vm.nr_hugepages": "5128"    # 5128 × 2MiB = ~10GiB
```

### Terraform

```hcl
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "ml-cluster"
  cluster_version = "1.31"

  enable_efa_support = true    # Adds EFA security group rules

  eks_managed_node_groups = {
    gpu-efa = {
      ami_type       = "AL2023_x86_64_NVIDIA"
      instance_types = ["p5.48xlarge"]
      min_size       = 0
      desired_size   = 2
      max_size       = 8

      enable_efa_support = true   # Exposes all EFA interfaces

      subnet_ids = [module.vpc.private_subnets[0]]  # Single AZ
    }
  }
}
```

## EFA Device Plugin

The AWS EFA Kubernetes device plugin exposes `vpc.amazonaws.com/efa` as a schedulable resource. It's automatically installed by eksctl when `efaEnabled: true`.

### Manual Installation

```bash
kubectl apply -f https://raw.githubusercontent.com/aws-samples/aws-efa-eks/main/manifest/efa-k8s-device-plugin.yml
```

For p6-b200 instances, use device plugin **v0.5.6 or later**.

### Verify Installation

```bash
# Check DaemonSet
kubectl get ds -n kube-system aws-efa-k8s-device-plugin-daemonset

# Check node resources
kubectl describe node <node-name> | grep -A5 "Allocatable"
# Should show:
#   vpc.amazonaws.com/efa: 32    (for p5.48xlarge)
```

## Requesting EFA in Pod Specs

```yaml
resources:
  requests:
    nvidia.com/gpu: "8"
    vpc.amazonaws.com/efa: "32"       # All 32 EFA interfaces on p5
    hugepages-2Mi: "5120Mi"           # Huge Pages for EFA
    memory: "128Gi"
  limits:
    nvidia.com/gpu: "8"
    vpc.amazonaws.com/efa: "32"
    hugepages-2Mi: "5120Mi"
    memory: "128Gi"
```

**Request all EFA interfaces** — partial allocation causes suboptimal NCCL topology mapping.

## Huge Pages

EFA requires Huge Pages for its internal buffers. Amazon Linux 2 EFA AMIs pre-allocate **5128 × 2MiB** Huge Pages.

```yaml
# In pod spec
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

## Security Groups

EFA requires a security group that allows **all traffic between members of the same security group**:

```yaml
# This is what eksctl creates automatically with efaEnabled: true
SecurityGroupIngress:
  - IpProtocol: "-1"      # All protocols
    FromPort: 0
    ToPort: 65535
    SourceSecurityGroupId: !Ref EFASecurityGroup   # Self-referencing

SecurityGroupEgress:
  - IpProtocol: "-1"
    FromPort: 0
    ToPort: 65535
    DestinationSecurityGroupId: !Ref EFASecurityGroup
```

This is separate from the EKS cluster security group. Both the EFA SG and the cluster SG must be attached to the nodes.

## Placement Groups

All EFA-enabled nodes should be in a **cluster placement group** for lowest latency. eksctl creates this automatically. For Terraform/manual setup:

```bash
aws ec2 create-placement-group \
  --group-name ml-training-pg \
  --strategy cluster
```

Reference the placement group in the node group's launch template. All instances must be in the **same Availability Zone** (cluster placement group requirement).

## EFA-Only Interfaces

For p5/p5e/trn2 instances, interfaces beyond network card 0 can be **EFA-only** (no IP stack, no ENA). Benefits:
- Lower overhead — no TCP/IP processing
- All bandwidth dedicated to RDMA/SRD traffic

Requirements:
- VPC CNI ≥ 1.18.5
- Amazon Linux 2 AMI ≥ v20240928
- Custom launch template with `InterfaceType: efa-only` (network card 0 must remain standard EFA+ENA)

eksctl does not currently support EFA-only interfaces — use a custom launch template.

## NCCL Integration

EFA uses the **aws-ofi-nccl** plugin to bridge NCCL and libfabric (EFA's communication library). This plugin is pre-installed in NVIDIA NGC containers and AWS Deep Learning Containers.

### Environment Variables

Modern aws-ofi-nccl (≥ 1.7.0) with libfabric (≥ 1.18.0) requires minimal configuration. Most legacy env vars are no longer needed:

| Variable | Needed? | Notes |
|----------|---------|-------|
| `FI_PROVIDER=efa` | **No** (auto-detected) | Only set if multiple providers present |
| `FI_EFA_USE_DEVICE_RDMA=1` | **No** (≥ libfabric 1.18.0) | Not harmful to set, but unnecessary on p4/p5 |
| `FI_EFA_FORK_SAFE=1` | **No** (≥ aws-ofi-nccl 1.7.0) | Legacy — was needed for older versions |
| `NCCL_ALGO` | Optional | Override collective algorithm selection |
| `NCCL_PROTO` | Optional | Override protocol selection |
| `NCCL_MIN_NCHANNELS` | Default | Leave at default (8 for 4-NIC, higher for 32-NIC) |
| `NCCL_DEBUG=INFO` | For debugging | Verbose NCCL logging |
| `NCCL_TOPO_DUMP_FILE` | For debugging | Dump detected topology |

**Key point**: On modern software stacks (DLC, NGC containers from 2024+), EFA+NCCL works out of the box with zero env var configuration.

### NCCL Test (Validation)

Run the standard NCCL all-reduce test to validate EFA is working:

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: nccl-test
spec:
  slotsPerWorker: 8
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
            - name: launcher
              image: public.ecr.aws/hpc-cloud/nccl-tests:latest
              env:
                - name: PATH
                  value: "/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH"
                - name: LD_LIBRARY_PATH
                  value: "/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib"
              command: ["mpirun"]
              args:
                - "--allow-run-as-root"
                - "-np"
                - "16"                    # 2 nodes × 8 GPUs
                - "--bind-to"
                - "none"
                - "-x"
                - "NCCL_DEBUG=INFO"
                - "/opt/nccl-tests/build/all_reduce_perf"
                - "-b"
                - "8"
                - "-e"
                - "16G"
                - "-f"
                - "2"
                - "-g"
                - "1"
    Worker:
      replicas: 2
      template:
        spec:
          containers:
            - name: worker
              image: public.ecr.aws/hpc-cloud/nccl-tests:latest
              resources:
                requests:
                  nvidia.com/gpu: "8"
                  vpc.amazonaws.com/efa: "32"
                  hugepages-2Mi: "5120Mi"
                  memory: "128Gi"
                limits:
                  nvidia.com/gpu: "8"
                  vpc.amazonaws.com/efa: "32"
                  hugepages-2Mi: "5120Mi"
                  memory: "128Gi"
              volumeMounts:
                - name: hugepages
                  mountPath: /dev/hugepages
          volumes:
            - name: hugepages
              emptyDir:
                medium: HugePages
```

**Expected output**: For p5.48xlarge (32 EFA), expect **~400-900 GB/s** bus bandwidth for large message all-reduce (varies by message size). If you see bandwidth similar to TCP (~10-25 GB/s), EFA is not being used.

### Verifying EFA Usage

In NCCL debug output, look for:

```
NCCL INFO NET/OFI Using aws-ofi-nccl ...
NCCL INFO NET/OFI Selected Provider is efa
```

If you see `Selected Provider is sockets` or `tcp`, EFA is not active.

## Training Job Pattern

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: distributed-training
spec:
  slotsPerWorker: 8
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
            - name: launcher
              image: my-training-image:latest
              command: ["mpirun"]
              args:
                - "--allow-run-as-root"
                - "-np"
                - "32"
                - "--bind-to"
                - "none"
                - "python"
                - "train.py"
    Worker:
      replicas: 4
      template:
        spec:
          nodeSelector:
            node.kubernetes.io/instance-type: p5.48xlarge
          containers:
            - name: worker
              image: my-training-image:latest
              resources:
                requests:
                  nvidia.com/gpu: "8"
                  vpc.amazonaws.com/efa: "32"
                  hugepages-2Mi: "5120Mi"
                  memory: "256Gi"
                limits:
                  nvidia.com/gpu: "8"
                  vpc.amazonaws.com/efa: "32"
                  hugepages-2Mi: "5120Mi"
                  memory: "256Gi"
              volumeMounts:
                - name: hugepages
                  mountPath: /dev/hugepages
                - name: shm
                  mountPath: /dev/shm
          volumes:
            - name: hugepages
              emptyDir:
                medium: HugePages
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "64Gi"   # Shared memory for NCCL
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `vpc.amazonaws.com/efa: 0` on node | Device plugin not running | Check `aws-efa-k8s-device-plugin-daemonset` DaemonSet |
| NCCL falls back to TCP | Security group blocking EFA traffic | Ensure self-referencing SG rule on all protocols |
| Low bandwidth in NCCL test | Nodes not in placement group | Verify cluster placement group, single AZ |
| Pod stuck Pending for EFA | Insufficient EFA resources | Check node allocatable; don't partially allocate |
| Huge Pages allocation failure | Not pre-allocated on node | Check AMI version; Bottlerocket needs `vm.nr_hugepages` in settings |
| NCCL timeout during training | Nodes in different AZs | All EFA nodes must be same AZ for placement group |
| "No EFA devices found" | Instance type doesn't support EFA | Verify instance type with `aws ec2 describe-instance-types --filters Name=network-info.efa-supported,Values=true` |
| p6-b200 EFA not working | Old device plugin | Upgrade EFA device plugin to ≥ v0.5.6 |
