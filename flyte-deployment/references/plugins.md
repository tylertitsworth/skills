# Flyte Task Plugins Configuration

Enable additional task types beyond the default container tasks.

## Table of Contents

- [Enabling plugins](#enabling-plugins)
- [Ray plugin](#ray-plugin)
- [Spark plugin](#spark-plugin)
- [MPI plugin](#mpi-plugin)
- [Dask plugin](#dask-plugin)
- [K8s pod plugin](#k8s-pod-plugin)

## Enabling Plugins

Plugins are enabled in the Helm values under `configuration.inline.tasks`:

```yaml
configuration:
  inline:
    tasks:
      task-plugins:
        enabled-plugins:
          - container
          - sidecar
          - k8s-array
          - ray          # add plugins here
          - spark
        default-for-task-types:
          container: container
          sidecar: sidecar
          container_array: k8s-array
          ray: ray
          spark: spark
```

For `flyte-core`, the config goes in the ConfigMap:

```yaml
configmap:
  enabled_plugins:
    tasks:
      task-plugins:
        enabled-plugins:
          - container
          - sidecar
          - k8s-array
          - ray
        default-for-task-types:
          container: container
          sidecar: sidecar
          container_array: k8s-array
          ray: ray
```

## Ray Plugin

Requires KubeRay operator installed on the cluster.

```yaml
configuration:
  inline:
    tasks:
      task-plugins:
        enabled-plugins:
          - container
          - sidecar
          - k8s-array
          - ray
        default-for-task-types:
          container: container
          container_array: k8s-array
          ray: ray
    plugins:
      ray:
        ttlSecondsAfterFinished: 3600    # cleanup Ray clusters after 1hr
```

Install KubeRay:

```bash
helm install kuberay-operator kuberay/kuberay-operator --version 1.5.1
```

In Flyte workflows, use `@task(task_config=Ray(...))` to submit Ray jobs.

## Spark Plugin

Requires Spark operator installed.

```yaml
configuration:
  inline:
    tasks:
      task-plugins:
        enabled-plugins:
          - container
          - sidecar
          - k8s-array
          - spark
        default-for-task-types:
          container: container
          container_array: k8s-array
          spark: spark
    plugins:
      spark:
        spark-config-default:
          - spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version: "2"
          - spark.kubernetes.allocation.batch.size: "50"
```

Install Spark operator:

```bash
helm install spark-operator kubeflow/spark-operator --version 1.1.15
```

## MPI Plugin

For distributed training with Horovod or other MPI frameworks:

```yaml
configuration:
  inline:
    tasks:
      task-plugins:
        enabled-plugins:
          - container
          - sidecar
          - k8s-array
          - mpi
        default-for-task-types:
          container: container
          container_array: k8s-array
          mpi: mpi
```

Requires MPI operator (Kubeflow Training Operator).

## Dask Plugin

```yaml
configuration:
  inline:
    tasks:
      task-plugins:
        enabled-plugins:
          - container
          - sidecar
          - k8s-array
          - dask
        default-for-task-types:
          container: container
          container_array: k8s-array
          dask: dask
```

Requires Dask operator:

```bash
helm install dask-operator dask/dask-kubernetes-operator
```

## K8s Pod Plugin

The default container plugin runs tasks as single pods. For more control over pod spec (sidecars, init containers, volumes):

```yaml
configuration:
  inline:
    tasks:
      task-plugins:
        enabled-plugins:
          - container
          - sidecar
          - k8s-array
        default-for-task-types:
          container: container
          sidecar: sidecar
          container_array: k8s-array
```

Use `@task(task_config=Pod(...))` in workflows for custom pod specs.

### Pod Templates

Define default pod templates for projects/domains:

```yaml
configuration:
  inline:
    plugins:
      k8s:
        default-pod-template-name: default-template
```

Create the PodTemplate as a K8s resource:

```yaml
apiVersion: v1
kind: PodTemplate
metadata:
  name: default-template
  namespace: flyte
template:
  spec:
    tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
    nodeSelector:
      node-type: gpu
```
