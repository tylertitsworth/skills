---
name: flyte-deployment
description: >
  Deploy, configure, and operate Flyte on Kubernetes — the workflow orchestration platform
  for ML and data pipelines. Use when: (1) Installing Flyte via Helm (flyte-binary for
  small clusters, flyte-core for production, multi-cluster for scale),
  (2) Configuring backends (blob storage, PostgreSQL, task execution plugins),
  (3) Setting up multi-tenancy (projects, domains, resource quotas),
  (4) Enabling task plugins (K8s pods, Spark, Ray, MPI, Dask),
  (5) Managing Flyte with flytectl CLI (register workflows, manage projects, launch plans),
  (6) Operating Flyte (upgrades, monitoring, scaling FlytePropeller),
  (7) Setting up authentication (OAuth2, OIDC),
  (8) Debugging platform issues (propeller stuck, pod failures, storage errors).
  NOT for writing Flyte workflows (SDK) — that's a separate skill.
---

# Flyte Deployment

Workflow orchestration platform for ML/data pipelines. This skill covers ops/deployment only.

**Docs:** https://docs-legacy.flyte.org/en/latest/deployment/index.html
**GitHub:** https://github.com/flyteorg/flyte
**Helm charts:** https://github.com/flyteorg/flyte/tree/master/charts

## Architecture

| Component | Purpose |
|---|---|
| **FlyteAdmin** | Control plane API server (gRPC + HTTP) |
| **FlyteConsole** | Web UI |
| **FlytePropeller** | Execution engine (runs on each compute cluster) |
| **DataCatalog** | Caching and artifact tracking |
| **FlyteConnector** | External service integrations |

**Dependencies:** PostgreSQL (metadata) + Object Store (S3/GCS/Azure Blob for data)

## Deployment Paths

| Path | Helm Chart | Use Case |
|---|---|---|
| **Sandbox** | `flyte` / `flyte-sandbox` | Local testing (not production) |
| **Single Cluster** | `flyte-binary` | Production, all-in-one binary, ≤13K nodes |
| **Multi-Cluster** | `flyte-core` | Large scale, control plane separated from execution |

**Recommendation:** Start with `flyte-binary` unless you need multi-cluster scale.

## Installation (Single Cluster)

### Prerequisites

- Kubernetes cluster (EKS, GKE, AKS, or on-prem)
- PostgreSQL database (RDS, CloudSQL, or self-managed)
- Object store bucket (S3, GCS, Azure Blob, or Minio)
- IAM role for Flyte backend

### Install via Helm

```bash
helm repo add flyteorg https://flyteorg.github.io/flyte

# Download starter values (AWS example)
curl -sL https://raw.githubusercontent.com/flyteorg/flyte/master/charts/flyte-binary/eks-starter.yaml -o values.yaml

# Edit values.yaml with your database, storage, and IAM config

# Install
helm install flyte-backend flyteorg/flyte-binary \
  --namespace flyte --create-namespace \
  --values values.yaml

# Verify
kubectl -n flyte get pods
kubectl -n flyte port-forward svc/flyte-binary 8088:8088 8089:8089
# Open http://localhost:8088/console
```

### Minimal values.yaml Structure

```yaml
configuration:
  database:
    postgres:
      host: <rds-endpoint>
      port: 5432
      dbname: flyteadmin
      username: flyteadmin
      passwordPath: /etc/flyte/db-pass  # or use secret
  storage:
    metadataContainer: flyte-metadata
    userDataContainer: flyte-userdata
    provider: s3
    providerConfig:
      s3:
        region: us-west-2
        authType: iam
  inline:
    tasks:
      task-plugins:
        enabled-plugins:
          - container
          - sidecar
          - k8s-array
        default-for-task-types:
          container: container
          container_array: k8s-array
```

## flytectl CLI

```bash
# Install
curl -sL https://ctl.flyte.org/install | bash

# Configure
flytectl config init --host localhost:8088

# Projects
flytectl create project --name my-project --id my-project
flytectl get project

# Register workflows
pyflyte package --image my-image:latest -o flyte-package.tgz
flytectl register files \
  --project my-project \
  --domain development \
  --archive flyte-package.tgz \
  --version "$(git rev-parse HEAD)"

# Run workflows
flytectl create execution \
  --project my-project \
  --domain development \
  --execFile exec-spec.yaml

# Launch plans
flytectl get launchplan --project my-project --domain development
flytectl update launchplan --activate --project my-project --domain development --name my-lp

# Resource attributes (quotas per project-domain)
flytectl update task-resource-attribute \
  --project my-project --domain production \
  --attrFile task-resource-attrs.yaml
```

### Config File (~/.flyte/config.yaml)

```yaml
admin:
  endpoint: dns:///flyte.example.com
  authType: Pkce
  insecure: false
logger:
  show-source: true
  level: 0
```

## Projects and Domains

Flyte organizes work into **projects** (teams/applications) and **domains** (environments):

| Domain | Purpose |
|---|---|
| `development` | Dev/test, lower resource limits |
| `staging` | Pre-production validation |
| `production` | Production workloads, higher quotas |

```bash
# Create project
flytectl create project --name ml-team --id ml-team --description "ML team workflows"

# Set resource quotas per project-domain
# task-resource-attrs.yaml:
# defaults:
#   cpu: "2"
#   memory: "4Gi"
#   gpu: "0"
# limits:
#   cpu: "16"
#   memory: "64Gi"
#   gpu: "8"
flytectl update task-resource-attribute \
  --project ml-team --domain production \
  --attrFile task-resource-attrs.yaml
```

## Task Plugins

For configuring Spark, Ray, MPI, Dask, and other task plugins, see `references/plugins.md`.

## Authentication

For OAuth2/OIDC setup, see `references/auth-and-operations.md`.

## Operations and Troubleshooting

For upgrades, monitoring, scaling, and debugging, see `references/auth-and-operations.md`.
