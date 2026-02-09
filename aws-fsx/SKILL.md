---
name: aws-fsx
description: Amazon FSx for Lustre — architecture, performance model, striping, S3 data repository, metadata IOPS, and EKS integration. Use when understanding FSx for Lustre internals, tuning performance for ML workloads, configuring S3 data repositories, or troubleshooting Lustre filesystem behavior on EKS.
---

# AWS FSx for Lustre

## Architecture

An FSx for Lustre filesystem consists of:

- **Metadata Targets (MDTs)**: Store file metadata — names, timestamps, permissions, directory structure, file layouts. Hosted on metadata servers (MDS).
- **Object Storage Targets (OSTs)**: Store actual file data. Each OST is backed by a disk (SSD or HDD). Files are striped across OSTs for parallel I/O.
- **File servers**: In-memory cache layer in front of OSTs. Hot data is served from cache (network-limited), cold data from disk.

```
Client (pod) ─── Lustre client ─── File servers ─── OSTs (data)
                                      │                └── SSD or HDD disks
                                      └── MDT (metadata)
```

Every client mounts the full filesystem and communicates directly with the file servers hosting the relevant OSTs — no single server bottleneck.

### Read/Write Paths

- **Cached read**: Client → file server in-memory/SSD cache → network-limited throughput
- **Uncached read**: Client → file server → disk → limited by lower of network and disk throughput
- **Write**: Client → file server → disk → limited by lower of network and disk throughput
- **S3 lazy load**: First access to an S3-backed file triggers a fetch from S3 → stored on OSTs → subsequent reads from OSTs

### Client Throughput Limits

| Client Network Interface | Max Throughput per Client |
|-------------------------|-------------------------|
| Standard ENA | 100 Gbps (5 Gbps per OST) |
| EFA | 700 Gbps |
| EFA with GPUDirect Storage (GDS) | 1200 Gbps |

For filesystems with >10 GB/s throughput capacity, AWS recommends EFA-enabled clients. GDS allows GPU memory to read/write directly to Lustre storage without CPU copies.

## Deployment Types

| Type | Durability | Throughput | Min Size | Use Case |
|------|-----------|------------|----------|----------|
| `SCRATCH_1` | None | 200 MB/s per TiB (burst) | 1.2 TiB | Ephemeral training data |
| `SCRATCH_2` | None | 200 MB/s per TiB (burst) | 1.2 TiB | Better networking than SCRATCH_1 |
| `PERSISTENT_1` | In-AZ replication | 50–200 MB/s per TiB | 1.2 TiB | Longer-lived workloads |
| `PERSISTENT_2` | In-AZ replication | 125–1000 MB/s per TiB | 1.2 TiB | Production, highest throughput |

**SCRATCH** filesystems have no data replication — data is lost if hardware fails. Ideal for training jobs where data is re-derivable from S3.

**PERSISTENT_2** supports `perUnitStorageThroughput` of 125, 250, 500, or 1000 MB/s per TiB. 1000 MB/s requires SSD storage.

**Storage sizing**: Minimum 1.2 TiB, then increments of 2.4 TiB. Throughput scales linearly with size.

## Striping

Lustre splits files into chunks distributed across multiple OSTs. This is the primary performance lever for large file I/O.

### Default Progressive File Layout (PFL)

Filesystems created after August 2023 use a 4-component PFL:

| File Size | Stripe Count | Effect |
|-----------|-------------|--------|
| ≤ 100 MiB | 1 | Single OST, no overhead |
| 100 MiB – 10 GiB | 8 | Parallel I/O across 8 OSTs |
| 10 GiB – 100 GiB | 16 | Higher parallelism |
| > 100 GiB | 32 | Maximum parallelism |

### Custom Striping

```bash
# Set stripe count on a directory (applies to new files)
lfs setstripe -c 16 /mount/training-data/

# Set PFL for a directory
lfs setstripe -E 100M -c 1 -E 10G -c 8 -E 100G -c 16 -E -1 -c 32 /mount/data/

# Check file layout
lfs getstripe /mount/data/large-file.bin

# Migrate existing file to new layout
lfs migrate -c 32 /mount/data/existing-file.bin

# View OST usage
lfs df -h /mount/
```

### Striping Guidelines

- **Large files (>1 GiB)**: Higher stripe count improves throughput. Stripe across many OSTs.
- **Small files (<100 MiB)**: Stripe count of 1. Higher counts add metadata overhead (network round-trip per OST in layout).
- **Stripe count -1**: Stripe across all OSTs. Use for largest files.
- **Stripe size**: Default 1 MiB. Rarely needs changing.
- **Don't set high stripe counts on directories with many small files** — metadata overhead degrades performance.

### What Striping Can't Fix

- **Metadata-heavy workloads** (millions of tiny files, `ls` on huge directories): Limited by MDT IOPS, not striping.
- **Single-threaded sequential reads**: Limited by single OST throughput. Application must use parallel I/O.
- **Random small I/O**: Lustre is optimized for large sequential I/O. Small random reads/writes are limited by latency.

## Metadata IOPS

Metadata operations (file create, open, close, delete, directory operations) are limited by MDT performance.

### PERSISTENT_2: User-Provisioned Metadata IOPS

| Operation | IOPS per Provisioned Unit |
|-----------|--------------------------|
| File create, open, close | 2 |
| File delete | 1 |
| Directory create, rename | 0.1 |
| Directory delete | 0.2 |

Valid provisioned values: 1500, 3000, 6000, 12000, and multiples of 12000 up to 192000.

### SSD: Automatic Mode

| Storage Capacity | Included Metadata IOPS |
|-----------------|----------------------|
| 1.2 TiB | 1500 |
| 2.4 TiB | 3000 |
| 4.8–9.6 TiB | 6000 |
| 12–45.6 TiB | 12000 |
| ≥48 TiB | 12000 per 24 TiB |

### ML Workload Implications

- **Training data loading**: Mostly sequential reads of large files → limited by OST throughput, not metadata. Striping helps.
- **Checkpoint saving**: Large sequential writes → striping helps. But initial file creation hits MDT.
- **Preprocessing with many small files**: Can bottleneck on metadata IOPS. Consider pre-aggregating into fewer large files (TFRecord, WebDataset, etc.).

## Data Repository Associations (DRA)

DRAs are the mechanism for linking FSx for Lustre to S3 buckets. Each DRA maps a filesystem path to an S3 prefix, creating a bidirectional data bridge.

### Key Properties

- **Maximum 8 DRAs per filesystem**
- Each DRA maps one filesystem path to one S3 prefix (1:1)
- Paths cannot overlap (e.g., `/ns1/` and `/ns1/subdir/` cannot coexist)
- `/` as filesystem path is only allowed for the first DRA
- Only one DRA request processed at a time (others queue)
- **Not available** on Scratch 1 or Lustre 2.10 filesystems

### How DRAs Work

1. **Metadata import**: S3 object metadata (name, size, timestamps) is loaded into the MDT. File data is **not** copied — it's lazy-loaded on first access.
2. **First read**: Triggers an HSM (Hierarchical Storage Management) restore from S3 → data fetched and cached on OSTs.
3. **Subsequent reads**: Served from OSTs (no S3 latency).
4. **Auto-import**: S3 changes automatically reflected in Lustre metadata.
5. **Auto-export**: Filesystem changes automatically written back to S3 (asynchronous).

### DRA Configuration

| Setting | Options | Default (Console) | Default (CLI/API) |
|---------|---------|-------------------|-------------------|
| **Import policy** | None, New, Changed, Deleted (any combination) | New + Changed + Deleted | Disabled |
| **Export policy** | None, New, Changed, Deleted (any combination) | New + Changed + Deleted | Disabled |
| **Import metadata on create** | Yes / No | Yes | No |

**Auto-import policies**:

| Policy | Effect |
|--------|--------|
| None | No auto-import. Use import data repository tasks manually. |
| New | Import metadata for new S3 objects |
| Changed | Import metadata for modified S3 objects |
| Deleted | Remove metadata for deleted S3 objects |

**Auto-export**: Exports regular files, symlinks, and directories. Does **not** export special files (FIFO, block, character, socket).

**Conflict handling**: If the same file is modified in both the filesystem and S3 simultaneously, there is **no automatic conflict resolution**. Application-level coordination is required.

### Import and Export Data Repository Tasks

In addition to automatic import/export, you can run on-demand tasks:

- **Import task**: Loads metadata for new/changed files from S3 into the filesystem
- **Export task**: Exports file data and metadata to S3

**Note**: Auto-export and export tasks **cannot be used simultaneously** on the same filesystem. Auto-import and import tasks **can** be used simultaneously.

### Cross-Region and Cross-Account

| Feature | Cross-Region | Cross-Account |
|---------|-------------|---------------|
| Auto-import | ❌ Same region only | ✅ Supported |
| Auto-export | ✅ Supported | ✅ Supported |

### Pre-Warming Data

Lazy loading means first-epoch training has S3 latency on first access. Pre-warm with:

```bash
# Restore specific files from S3 archive to OSTs
lfs hsm_restore /mount/data/file1 /mount/data/file2

# Bulk restore a directory
nohup find /mount/data/ -type f -print0 | xargs -0 -n 1 lfs hsm_restore &
```

### ImportedFileChunkSize

Controls how S3-imported files are striped (default: 1 GiB). Files larger than this value are automatically striped across `ceil(FileSize / ChunkSize) + 1` OSTs.

## Intelligent-Tiering Storage Class

An alternative to SSD/HDD storage classes. Automatically tiers data across three access tiers:

| Tier | Latency | Cost | When Data Moves Here |
|------|---------|------|---------------------|
| Frequent Access | Sub-millisecond (SSD) | Highest | Recently accessed data |
| Infrequent Access | Low (HDD) | Lower | Data not accessed recently |
| Archive | Higher (retrieval delay) | Lowest | Rarely accessed data |

- Optional **SSD read cache** for frequently accessed data
- No minimum storage capacity — pay only for data stored
- Starting at <$0.005/GB-month
- **Not available** with Scratch deployments
- Metadata IOPS: Only 6000 or 12000 (user-provisioned mode only, no automatic)

**ML use case**: Good for long-lived datasets where only a subset is actively used. Training data accessed during current epoch stays in Frequent Access tier; older experiment data tiers down automatically.

## Compression

LZ4 transparent compression (`dataCompressionType: LZ4`) reduces storage costs and can improve throughput for compressible data. Applied to new files written after enabling.

## Backups

- **Automatic backups**: Daily, stored in S3 (11 9's durability). Retention 0-90 days.
- **Manual backups**: On-demand via console/CLI/API.
- **Cross-region/cross-account backup copy**: For disaster recovery and compliance.
- **Restore**: Creates a new filesystem from backup.
- **Incremental**: Only changed data since last backup.
- **Persistent filesystems only** — Scratch filesystems do not support backups.

Backups are managed via AWS Backup for policy-based scheduling.

## Storage Quotas

User-level and group-level quotas to control storage consumption:

```bash
# Set quota for a user (soft limit, hard limit, grace period)
lfs setquota -u username --block-softlimit 100G --block-hardlimit 120G /mount/

# Set quota for a group
lfs setquota -g groupname --block-softlimit 1T --block-hardlimit 1.2T /mount/

# Check quota usage
lfs quota -u username /mount/
lfs quota -g groupname /mount/
```

Useful for multi-team environments sharing a filesystem to prevent any team from exhausting storage.

## Encryption

- **At rest**: All filesystems encrypted. AWS managed key (default) or customer-managed KMS key.
- **In transit**: Encryption of data in transit between clients and file servers. Available in select regions. Enabled automatically for supported client kernel versions.

## GPUDirect Storage (GDS)

For EFA-enabled filesystems with NVIDIA GPUs, GDS enables direct data transfer between GPU memory and FSx for Lustre storage, bypassing CPU and system memory entirely.

- **Throughput**: Up to 1200 Gbps per client (vs 700 Gbps with EFA alone)
- **Requirements**: EFA-enabled filesystem, EFA-enabled GPU instance, NVIDIA GDS driver
- Eliminates the CPU copy bottleneck for checkpoint writes and data loading

## All Features Summary

### Supported

- POSIX filesystem semantics (with caveats below)
- ReadWriteMany — multiple pods across nodes simultaneously
- Data Repository Associations (DRA) — up to 8 S3 links per filesystem
- Auto-import and auto-export with S3
- Import/export data repository tasks (on-demand)
- Progressive file layouts (PFL) with automatic striping
- `lfs` CLI for stripe management, HSM operations, quotas
- Transparent LZ4 compression
- Encryption at rest (AWS managed or customer KMS keys)
- Encryption in transit (select regions)
- EFA (700 Gbps) and GPUDirect Storage (1200 Gbps) per-client throughput
- Intelligent-Tiering storage class (automatic cost optimization)
- Automatic and manual backups (persistent filesystems)
- Cross-region and cross-account backup copy
- Storage quotas (user and group level)
- Lustre client on Amazon Linux 2, AL2023, Ubuntu, RHEL, CentOS, SUSE
- EKS CSI driver (dynamic and static provisioning)
- AWS Backup integration
- CloudWatch metrics (throughput, IOPS, metadata ops, capacity)
- Storage capacity scaling (increase online)

### Not Supported / Limitations

| Limitation | Detail |
|-----------|--------|
| **Single AZ** | FSx for Lustre filesystems exist in one subnet/AZ. No multi-AZ replication. |
| **No NFS/SMB** | Lustre protocol only. Requires Lustre client (kernel module). |
| **No online resize down** | Can increase capacity, cannot shrink. |
| **No snapshots** | No built-in snapshot capability (unlike EBS or FSx ONTAP). |
| **Minimum size** | 1.2 TiB minimum. Can't create small filesystems. |
| **S3 lazy load latency** | First access to uncached S3-backed files has S3 latency. |
| **Metadata IOPS cap** | Directory operations are slow relative to data I/O. Millions of tiny files suffer. |
| **Client compatibility** | Requires specific kernel versions with Lustre client module. |
| **Hard link limit** | Lustre has lower hard link limits than ext4/XFS. |
| **No POSIX ACLs** | Only basic Unix permissions (uid/gid/mode). |

## EKS CSI Driver

The `fsx.csi.aws.com` CSI driver enables dynamic and static provisioning. Install as an EKS add-on (requires Pod Identity agent). Key StorageClass parameters:

| Parameter | Values | Effect |
|-----------|--------|--------|
| `subnetId` | subnet ID | **Required.** Subnet for filesystem ENI |
| `securityGroupIds` | SG IDs | Security groups (must allow TCP 988 inbound) |
| `deploymentType` | `SCRATCH_1`, `SCRATCH_2`, `PERSISTENT_1`, `PERSISTENT_2` | Durability/performance tier |
| `perUnitStorageThroughput` | `125`–`1000` | MB/s per TiB (PERSISTENT only) |
| `dataCompressionType` | `NONE`, `LZ4` | Transparent compression |
| `s3ImportPath` | `s3://bucket/prefix` | S3 data repository source |
| `autoImportPolicy` | `NONE`, `NEW`, `NEW_CHANGED`, `NEW_CHANGED_DELETED` | Auto-import from S3 |

See `references/troubleshooting.md` for common issues.

## Cross-References

- [aws-efa](../aws-efa/) — EFA networking for maximum client throughput (700 Gbps+)
- [ray-train](../ray-train/) — Distributed training jobs consuming FSx-backed PVCs
- [kueue](../kueue/) — Queue training jobs that mount FSx volumes
- [megatron-lm](../megatron-lm/) — Large-scale training with shared checkpoint storage

## Reference

- [FSx for Lustre docs](https://docs.aws.amazon.com/fsx/latest/LustreGuide/)
- [FSx CSI driver](https://github.com/kubernetes-sigs/aws-fsx-csi-driver)
- [FSx pricing](https://aws.amazon.com/fsx/lustre/pricing/)
- `references/troubleshooting.md` — mount issues, performance, data repository tasks
