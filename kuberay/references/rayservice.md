# RayService

Deploy Ray Serve applications on Kubernetes with zero-downtime upgrades.

## Table of Contents

- [What RayService provides](#what-rayservice-provides)
- [Basic RayService example](#basic-rayservice-example)
- [Serve configuration](#serve-configuration)
- [Accessing the service](#accessing-the-service)
- [Zero-downtime upgrades](#zero-downtime-upgrades)
- [High availability](#high-availability)

## What RayService Provides

RayService manages:
1. A **RayCluster** (created and managed automatically)
2. **Ray Serve applications** deployed on that cluster
3. **Kubernetes Services** for routing traffic to Serve endpoints

## Basic RayService Example

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: llm-service
spec:
  serveConfigV2: |
    applications:
    - name: llm-app
      route_prefix: /llm
      import_path: serve_app:deployment
      runtime_env:
        pip:
          - transformers
          - torch
      deployments:
      - name: LLMDeployment
        num_replicas: 2
        ray_actor_options:
          num_gpus: 1
  rayClusterConfig:
    rayVersion: "2.53.0"
    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray-ml:2.53.0
            resources:
              limits:
                cpu: "4"
                memory: 8Gi
              requests:
                cpu: "4"
                memory: 8Gi
    workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 2
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray-ml:2.53.0
            resources:
              limits:
                cpu: "8"
                memory: 32Gi
                nvidia.com/gpu: "1"
              requests:
                cpu: "8"
                memory: 32Gi
                nvidia.com/gpu: "1"
```

## Serve Configuration

The `serveConfigV2` field is a YAML string defining Ray Serve applications:

```yaml
serveConfigV2: |
  applications:
  - name: app-name
    route_prefix: /endpoint
    import_path: module:deployment_graph
    runtime_env:
      pip: [package1, package2]
      env_vars:
        KEY: VALUE
    deployments:
    - name: DeploymentName
      num_replicas: 2
      max_ongoing_requests: 100
      ray_actor_options:
        num_cpus: 1
        num_gpus: 1
      autoscaling_config:
        min_replicas: 1
        max_replicas: 10
        target_ongoing_requests: 5
```

Key deployment options:
- `num_replicas`: Fixed replica count (mutually exclusive with `autoscaling_config`)
- `autoscaling_config`: Auto-scale replicas based on request load
- `ray_actor_options`: CPU/GPU/memory per replica
- `max_ongoing_requests`: Concurrency limit per replica

## Accessing the Service

KubeRay creates two services:

| Service | Purpose | Default Port |
|---|---|---|
| `<name>-head-svc` | Dashboard, client connections | 8265 (dashboard), 10001 (client) |
| `<name>-serve-svc` | HTTP endpoint for Serve apps | 8000 |

```bash
# Port-forward the serve endpoint
kubectl port-forward svc/<name>-serve-svc 8000:8000

# Send a request
curl http://localhost:8000/llm -d '{"prompt": "hello"}'

# From within the cluster
curl http://<name>-serve-svc.<namespace>.svc:8000/llm
```

Expose externally via Ingress or LoadBalancer:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ray-serve-ingress
spec:
  rules:
  - host: llm.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llm-service-serve-svc
            port:
              number: 8000
```

## Zero-Downtime Upgrades

When you update the RayService spec:

1. **Serve config change only** (`serveConfigV2`): In-place update — applications are redeployed on the existing cluster without downtime.

2. **Cluster config change** (`rayClusterConfig`): KubeRay creates a new RayCluster, deploys applications, then switches traffic from the old cluster to the new one. The old cluster is deleted after switchover.

## High Availability

For production, enable GCS fault tolerance so worker pods continue serving during head recovery:

```yaml
spec:
  rayClusterConfig:
    headGroupSpec:
      rayStartParams:
        redis-password: "password"
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

This ensures that if the head pod crashes and restarts, workers keep handling requests without interruption.
