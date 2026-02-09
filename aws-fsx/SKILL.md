---
name: aws-fsx
description: Amazon FSx for Lustre on EKS — CSI driver, StorageClass configuration, S3 data repository integration, and ML storage patterns. Use when configuring FSx for Lustre storage on EKS for training data, checkpoints, or shared model weights.
---

# AWS FSx for Lustre on EKS

Amazon FSx for Lustre is a fully managed high-performance parallel filesystem. Key properties for ML workloads:
- **Throughput**: Up to hundreds of GB/s aggregate
- **Latency**: Sub-millisecond
- **S3 integration**: Native data repository — lazy-loads S3 objects as Lustre files
- **Access mode**: ReadWriteMany (multiple pods across nodes)
- **Protocol**: Lustre (POSIX-compatible)

### CSI Driver Installation

Install as EKS add-on (recommended) or via Helm:

```bash
# EKS add-on (requires EKS Pod Identity agent)
aws eks create-addon \
  --cluster-name my-cluster \
  --addon-name aws-fsx-csi-driver \
  --service-account-role-arn arn:aws:iam::ACCOUNT:role/FsxCsiDriverRole

# Or via kubectl
kubectl apply -k "github.com/kubernetes-sigs/aws-fsx-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.8"
```

#### IAM Policy

The CSI driver needs permissions to create/delete FSx filesystems. Attach this policy to the driver's service account role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "fsx:CreateFileSystem",
        "fsx:DeleteFileSystem",
        "fsx:DescribeFileSystems",
        "fsx:TagResource",
        "fsx:UpdateFileSystem"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLocation",
        "s3:ListBucket",
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": ["arn:aws:s3:::ml-*", "arn:aws:s3:::ml-*/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["iam:CreateServiceLinkedRole"],
      "Resource": "arn:aws:iam::*:role/aws-service-role/fsx.amazonaws.com/*",
      "Condition": {
        "StringLike": {"iam:AWSServiceName": "fsx.amazonaws.com"}
      }
    }
  ]
}
```

### StorageClass Parameters

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fsx-lustre-persistent
provisioner: fsx.csi.aws.com
parameters:
  subnetId: subnet-0123456789abcdef0
  securityGroupIds: sg-0123456789abcdef0
  deploymentType: PERSISTENT_2
  storageType: SSD
  perUnitStorageThroughput: "250"
  dataCompressionType: LZ4
  autoImportPolicy: NEW_CHANGED_DELETED
  s3ImportPath: s3://ml-training-data
  s3ExportPath: s3://ml-training-data/exports
  fileSystemTypeVersion: "2.15"
mountOptions:
  - flock
reclaimPolicy: Delete
volumeBindingMode: Immediate
```

#### All StorageClass Parameters

| Parameter | Values | Default | Effect |
|-----------|--------|---------|--------|
| `subnetId` | subnet ID | — | **Required.** Subnet for FSx ENI (must be in same VPC as EKS nodes) |
| `securityGroupIds` | comma-separated SG IDs | — | Security groups for FSx ENI (port 988 inbound required) |
| `deploymentType` | `SCRATCH_1`, `SCRATCH_2`, `PERSISTENT_1`, `PERSISTENT_2` | `SCRATCH_1` | Filesystem durability and performance tier |
| `storageType` | `SSD`, `HDD` | `SSD` | Storage media (HDD only with PERSISTENT_1) |
| `perUnitStorageThroughput` | `125`, `250`, `500`, `1000` | — | MB/s per TiB (PERSISTENT only). 1000 requires SSD + PERSISTENT_2 |
| `dataCompressionType` | `NONE`, `LZ4` | `NONE` | Transparent compression |
| `autoImportPolicy` | `NONE`, `NEW`, `NEW_CHANGED`, `NEW_CHANGED_DELETED` | `NONE` | Auto-import from S3 when files change |
| `s3ImportPath` | `s3://bucket/prefix` | — | S3 data repository to import from |
| `s3ExportPath` | `s3://bucket/prefix` | — | S3 path for exports (same bucket as import) |
| `fileSystemTypeVersion` | `2.10`, `2.12`, `2.15` | `2.15` | Lustre version |
| `extraTags` | `key1=val1,key2=val2` | — | Additional AWS tags |

#### Deployment Types

| Type | Durability | Performance | Min Size | Throughput | Use Case |
|------|-----------|-------------|----------|-----------|----------|
| `SCRATCH_1` | None (no replication) | Burst: 200 MB/s/TiB | 1.2 TiB | 200 MB/s/TiB | Short-lived training jobs |
| `SCRATCH_2` | None | Burst: 200 MB/s/TiB | 1.2 TiB | 200 MB/s/TiB | Better networking than SCRATCH_1 |
| `PERSISTENT_1` | Within-AZ replication | Configurable | 1.2 TiB | 50-200 MB/s/TiB | Long-running workloads |
| `PERSISTENT_2` | Within-AZ replication | Configurable | 1.2 TiB | 125-1000 MB/s/TiB | Production, highest throughput |

**Minimum filesystem sizes**: 1.2 TiB, 2.4 TiB, or increments of 2.4 TiB (SCRATCH) / 2.4 TiB (PERSISTENT_2 SSD).

### Dynamic Provisioning with S3

The primary ML use case: FSx Lustre acts as a high-performance cache for S3 training data.

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fsx-lustre-s3
provisioner: fsx.csi.aws.com
parameters:
  subnetId: subnet-0123456789abcdef0
  securityGroupIds: sg-0123456789abcdef0
  deploymentType: SCRATCH_2
  s3ImportPath: s3://ml-training-data/imagenet
  autoImportPolicy: NEW_CHANGED
  dataCompressionType: LZ4
reclaimPolicy: Delete
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data
spec:
  accessModes: [ReadWriteMany]
  storageClassName: fsx-lustre-s3
  resources:
    requests:
      storage: 1200Gi
```

**Data flow**: S3 objects appear as files on the Lustre filesystem. On first access, data is lazy-loaded from S3 (transparent to the application). With `autoImportPolicy`, new/changed S3 objects are automatically reflected.

### Static Provisioning (Pre-existing Filesystem)

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: fsx-pv
spec:
  capacity:
    storage: 1200Gi
  accessModes: [ReadWriteMany]
  mountOptions:
    - flock
  csi:
    driver: fsx.csi.aws.com
    volumeHandle: fs-0123456789abcdef0
    volumeAttributes:
      dnsname: fs-0123456789abcdef0.fsx.us-west-2.amazonaws.com
      mountname: fsx
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: fsx-claim
spec:
  accessModes: [ReadWriteMany]
  storageClassName: ""
  resources:
    requests:
      storage: 1200Gi
  volumeName: fsx-pv
```

### Training Job Example

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: training-job
spec:
  template:
    spec:
      containers:
        - name: trainer
          image: my-training-image:latest
          resources:
            requests:
              nvidia.com/gpu: "8"
              memory: "64Gi"
            limits:
              nvidia.com/gpu: "8"
          volumeMounts:
            - name: training-data
              mountPath: /data
              readOnly: true
            - name: checkpoints
              mountPath: /checkpoints
      volumes:
        - name: training-data
          persistentVolumeClaim:
            claimName: training-data     # FSx Lustre backed by S3
        - name: checkpoints
          persistentVolumeClaim:
            claimName: checkpoint-storage # Separate PVC for writes
      restartPolicy: Never
```

### Multi-Pod Access

FSx for Lustre supports `ReadWriteMany` — multiple pods across multiple nodes can mount the same filesystem simultaneously. This is critical for:
- **Data-parallel training**: All workers read from the same dataset
- **Shared checkpoints**: Coordinator writes, all workers can read
- **Model serving**: Multiple inference pods load the same weights

### Networking Requirements

- FSx ENI must be in a subnet reachable from EKS worker nodes (same VPC or peered)
- Security group must allow **inbound TCP port 988** (Lustre protocol)
- For best performance, place EKS nodes and FSx filesystem in the **same Availability Zone**

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| PVC stuck in Pending | Missing subnet/SG in StorageClass | Verify `subnetId` and `securityGroupIds` |
| Mount timeout | Security group blocking port 988 | Add inbound TCP 988 rule |
| Slow first reads from S3 | Lazy loading on first access | Pre-warm with `lfs hsm_restore` or prefetch |
| CSI driver pods not running | Missing IAM permissions | Check Pod Identity / IRSA role |
| "filesystem not found" on static PV | Wrong `volumeHandle` or `dnsname` | Verify filesystem ID and DNS name in AWS console |

## Cross-References

- [aws-efa](../aws-efa/) — EFA networking for the GPU nodes consuming FSx storage
- [ray-train](../ray-train/) — Distributed training jobs that consume FSx-backed PVCs
- [kueue](../kueue/) — Queue training jobs that mount FSx volumes
- [megatron-lm](../megatron-lm/) — Large-scale training with shared checkpoint storage
