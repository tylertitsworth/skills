# FSx for Lustre Troubleshooting

## Diagnostic Commands

```bash
# View filesystem layout (MDT + OSTs, usage per target)
lfs df -h /mount/

# Check file stripe layout
lfs getstripe /mount/path/to/file

# Check directory stripe settings
lfs getstripe -d /mount/path/to/dir/

# Check HSM status (S3-backed files)
lfs hsm_state /mount/path/to/file
# States: exists, archived, released (data not on OSTs), dirty

# Check client mount
mount | grep lustre
# Expected: fs-xxx.fsx.region.amazonaws.com@tcp:/mountname on /mount type lustre

# CloudWatch metrics (via AWS CLI)
aws cloudwatch get-metric-statistics \
  --namespace AWS/FSx \
  --metric-name DataReadBytes \
  --dimensions Name=FileSystemId,Value=fs-xxx \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 --statistics Sum
```

## Common Issues

### PVC Stuck in Pending

| Cause | Fix |
|-------|-----|
| Missing `subnetId` in StorageClass | Add subnet ID from same VPC as EKS nodes |
| Missing `securityGroupIds` | Add SG that allows TCP 988 inbound |
| Subnet not in same VPC as EKS | Use subnet reachable from worker nodes |
| IAM permissions missing | Attach FSx + S3 permissions to CSI driver role |
| CSI driver not installed | Install `aws-fsx-csi-driver` EKS add-on |
| Pod Identity agent missing | Install EKS Pod Identity agent add-on |

### Mount Timeout / Failures

| Cause | Fix |
|-------|-----|
| Security group blocking port 988 | Add inbound TCP 988 rule from EKS node CIDR |
| Lustre client not installed on node | Use EKS-optimized AMI (AL2, AL2023) — includes Lustre client |
| Wrong `dnsname` in static PV | Verify with `aws fsx describe-file-systems` |
| Nodes in different VPC | Use VPC peering or same VPC |

### Slow First-Epoch Training (S3-Backed Data)

**Cause**: Lazy loading — first read triggers S3 fetch.

**Fixes**:
1. **Pre-warm**: Run `find /mount/data/ -type f -print0 | xargs -0 -n 1 lfs hsm_restore` before training
2. **Use init container**: Pre-warm in init container before training container starts
3. **Use SCRATCH filesystem**: Copy data to FSx ahead of time instead of lazy-loading
4. **Increase ImportedFileChunkSize**: Larger chunks = fewer OSTs per file = simpler layout

### Slow Metadata Operations (ls, find, file creation)

**Cause**: MDT IOPS bottleneck.

**Fixes**:
1. **Provision more metadata IOPS** (PERSISTENT_2): Increase in console or via API
2. **Reduce file count**: Aggregate small files into larger containers (WebDataset, TFRecord, tar)
3. **Avoid `ls -l` on huge directories**: Use `ls -U` (unsorted) or `lfs find` instead of `find`
4. **Stripe directories**: `lfs setstripe` for new data directories

### OST Full / Unbalanced

**Symptoms**: Write errors on some OSTs while others have free space.

**Cause**: Files not striped, or uneven data distribution.

**Fixes**:
1. **Stripe new files**: `lfs setstripe -c -1 /mount/dir/` (all OSTs)
2. **Migrate existing files**: `lfs migrate -c -1 /mount/path/large-file`
3. **Check usage per OST**: `lfs df -h /mount/`
4. **Enable PFL**: Set progressive file layout on parent directories

### S3 Auto-Import Not Working

| Cause | Fix |
|-------|-----|
| `autoImportPolicy` set to `NONE` | Update to `NEW_CHANGED` or `NEW_CHANGED_DELETED` |
| S3 event notifications not configured | FSx auto-configures these; check S3 bucket event config |
| IAM permissions | CSI driver role needs `s3:GetBucketNotificationConfiguration` |
| Different S3 bucket for import/export | Must use same bucket |

### "Filesystem not found" on Static PV

| Check | Fix |
|-------|-----|
| `volumeHandle` | Must be the filesystem ID (`fs-xxxxxxxxx`) |
| `dnsname` | Must be `fs-xxx.fsx.REGION.amazonaws.com` |
| `mountname` | Check with `aws fsx describe-file-systems` → `LustreConfiguration.MountName` |

### Performance Below Expected

| Symptom | Check | Fix |
|---------|-------|-----|
| Low single-file throughput | Stripe count | Increase stripe count for large files |
| Low aggregate throughput | OST count | Filesystem may need more capacity (more OSTs) |
| High latency reads | In-memory cache miss | Increase filesystem size for more cache, or use SSD |
| Low per-client throughput | Network interface | Use EFA (700 Gbps) or EFA+GDS (1200 Gbps) |
| Burst credits exhausted | CloudWatch `BurstCreditBalance` | Wait for credits, or use PERSISTENT_2 |

## CloudWatch Metrics

Key metrics to monitor:

| Metric | What It Tells You |
|--------|-------------------|
| `DataReadBytes` / `DataWriteBytes` | Data throughput |
| `DataReadOperations` / `DataWriteOperations` | Data IOPS |
| `MetadataOperations` | Metadata IOPS (file creates, deletes, etc.) |
| `FreeDataStorageCapacity` | Available storage |
| `BurstCreditBalance` | Remaining burst throughput credits (SCRATCH/PERSISTENT_1) |

Set alarms on `FreeDataStorageCapacity` and `BurstCreditBalance` to catch issues before training fails.
