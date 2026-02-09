---
name: operator-sdk
description: >
  Build Kubernetes operators with Operator SDK — custom controllers that encode operational
  knowledge. Use when: (1) Scaffolding new operator projects (Go, Helm, or Ansible), (2) Writing
  Go reconcilers with controller-runtime (watches, predicates, finalizers), (3) Wrapping Helm
  charts or Ansible playbooks as operators, (4) Defining and evolving CRDs (spec, status,
  validation, versioning), (5) Implementing best practices (level-triggered reconciliation,
  status conditions, owner references), (6) Testing operators (envtest, scorecard, e2e),
  (7) Packaging with OLM bundles and catalogs, (8) Debugging RBAC, reconcile loops, or
  watch failures.
---

# Operator SDK

Operator SDK is a framework for building Kubernetes operators — part of the [Operator Framework](https://operatorframework.io). Version: **1.38+** (based on Kubebuilder v4).

## Scaffolding a Project

### Go-Based Operator

```bash
mkdir -p $HOME/projects/my-operator && cd $HOME/projects/my-operator

# Initialize project
operator-sdk init \
  --domain example.com \
  --repo github.com/myorg/my-operator \
  --plugins=go/v4

# Create API + controller
operator-sdk create api \
  --group ml \
  --version v1alpha1 \
  --kind TrainingJob \
  --resource --controller
```

**Generated structure:**
```
my-operator/
├── api/v1alpha1/          # CRD types (spec, status)
│   ├── trainingjob_types.go
│   └── zz_generated.deepcopy.go
├── internal/controller/   # Reconciler logic
│   └── trainingjob_controller.go
├── config/                # Kustomize manifests (CRD, RBAC, manager)
├── cmd/main.go            # Entrypoint
├── Dockerfile
├── Makefile
└── PROJECT
```

### Helm-Based Operator

```bash
operator-sdk init --domain example.com --plugins helm

# Create API from an existing chart
operator-sdk create api \
  --group ml \
  --version v1alpha1 \
  --kind ModelServer \
  --helm-chart ./charts/model-server
# Or from a repo:
  --helm-chart model-server \
  --helm-chart-repo https://charts.example.com
```

The chart is copied to `helm-charts/model-server/`. The operator reconciles by running `helm upgrade --install` on each CR.

### Ansible-Based Operator

```bash
operator-sdk init --domain example.com --plugins ansible

operator-sdk create api \
  --group ml \
  --version v1alpha1 \
  --kind Notebook \
  --generate-role
```

Creates `roles/notebook/` with Ansible tasks. Map CRs to roles in `watches.yaml`.

## Go Operator: CRD Types

Define your API in `api/v1alpha1/trainingjob_types.go`:

```go
package v1alpha1

import (
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TrainingJobSpec defines the desired state
type TrainingJobSpec struct {
    // +kubebuilder:validation:Required
    // +kubebuilder:validation:MinLength=1
    Image string `json:"image"`

    // +kubebuilder:validation:Minimum=1
    // +kubebuilder:default=1
    Replicas int32 `json:"replicas,omitempty"`

    // +kubebuilder:validation:Minimum=0
    GPUs int32 `json:"gpus,omitempty"`

    // +kubebuilder:validation:Enum=pytorch;tensorflow;jax
    Framework string `json:"framework"`

    // +optional
    Config map[string]string `json:"config,omitempty"`
}

// TrainingJobStatus defines the observed state
type TrainingJobStatus struct {
    // +operator-sdk:csv:customresourcedefinitions:type=status
    Conditions []metav1.Condition `json:"conditions,omitempty"`

    // +optional
    Phase string `json:"phase,omitempty"` // Pending, Running, Succeeded, Failed

    // +optional
    StartTime *metav1.Time `json:"startTime,omitempty"`

    // +optional
    CompletionTime *metav1.Time `json:"completionTime,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
// +kubebuilder:resource:shortName=tj
type TrainingJob struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`

    Spec   TrainingJobSpec   `json:"spec,omitempty"`
    Status TrainingJobStatus `json:"status,omitempty"`
}
```

After modifying types:
```bash
make generate   # update deepcopy functions
make manifests  # regenerate CRD YAML and RBAC
```

### Validation Markers

| Marker | Purpose |
|--------|---------|
| `+kubebuilder:validation:Required` | Field must be set |
| `+kubebuilder:validation:Minimum=N` | Minimum numeric value |
| `+kubebuilder:validation:Maximum=N` | Maximum numeric value |
| `+kubebuilder:validation:Enum=a;b;c` | Allowed values |
| `+kubebuilder:validation:Pattern="regex"` | Regex validation |
| `+kubebuilder:validation:MinLength=N` | Min string length |
| `+kubebuilder:default=value` | Default value |
| `+optional` | Field is optional |

## Go Operator: Reconciler

The reconciler is the core logic. It's called whenever a watched resource changes:

```go
package controller

import (
    "context"
    "fmt"
    "time"

    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    "k8s.io/apimachinery/pkg/api/errors"
    "k8s.io/apimachinery/pkg/api/meta"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    "k8s.io/apimachinery/pkg/types"
    "k8s.io/client-go/tools/record"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
    "sigs.k8s.io/controller-runtime/pkg/log"

    mlv1alpha1 "github.com/myorg/my-operator/api/v1alpha1"
)

const finalizerName = "ml.example.com/finalizer"

type TrainingJobReconciler struct {
    client.Client
    Scheme   *runtime.Scheme
    Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=ml.example.com,resources=trainingjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=ml.example.com,resources=trainingjobs/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=ml.example.com,resources=trainingjobs/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch

func (r *TrainingJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := log.FromContext(ctx)

    // 1. Fetch the CR
    var tj mlv1alpha1.TrainingJob
    if err := r.Get(ctx, req.NamespacedName, &tj); err != nil {
        if errors.IsNotFound(err) {
            return ctrl.Result{}, nil // CR deleted, nothing to do
        }
        return ctrl.Result{}, err
    }

    // 2. Handle finalizer for cleanup
    if tj.DeletionTimestamp != nil {
        if controllerutil.ContainsFinalizer(&tj, finalizerName) {
            // Run cleanup logic
            log.Info("Running cleanup for TrainingJob")
            controllerutil.RemoveFinalizer(&tj, finalizerName)
            if err := r.Update(ctx, &tj); err != nil {
                return ctrl.Result{}, err
            }
        }
        return ctrl.Result{}, nil
    }
    if !controllerutil.ContainsFinalizer(&tj, finalizerName) {
        controllerutil.AddFinalizer(&tj, finalizerName)
        if err := r.Update(ctx, &tj); err != nil {
            return ctrl.Result{}, err
        }
    }

    // 3. Create or update child resources
    deploy := r.deploymentForTrainingJob(&tj)
    if err := ctrl.SetControllerReference(&tj, deploy, r.Scheme); err != nil {
        return ctrl.Result{}, err
    }

    found := &appsv1.Deployment{}
    err := r.Get(ctx, types.NamespacedName{Name: deploy.Name, Namespace: deploy.Namespace}, found)
    if err != nil && errors.IsNotFound(err) {
        log.Info("Creating Deployment", "name", deploy.Name)
        if err := r.Create(ctx, deploy); err != nil {
            return ctrl.Result{}, err
        }
        r.Recorder.Event(&tj, corev1.EventTypeNormal, "Created", "Deployment created")
    } else if err != nil {
        return ctrl.Result{}, err
    } else if found.Spec.Replicas != nil && *found.Spec.Replicas != tj.Spec.Replicas {
        found.Spec.Replicas = &tj.Spec.Replicas
        if err := r.Update(ctx, found); err != nil {
            return ctrl.Result{}, err
        }
    }

    // 4. Update status
    meta.SetStatusCondition(&tj.Status.Conditions, metav1.Condition{
        Type:    "Available",
        Status:  metav1.ConditionTrue,
        Reason:  "DeploymentReady",
        Message: "Training deployment is running",
    })
    tj.Status.Phase = "Running"
    if err := r.Status().Update(ctx, &tj); err != nil {
        return ctrl.Result{}, err
    }

    return ctrl.Result{}, nil
}

func (r *TrainingJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&mlv1alpha1.TrainingJob{}).
        Owns(&appsv1.Deployment{}).    // watch child deployments
        Complete(r)
}
```

### Key Reconciler Patterns

**Level-triggered (not edge-triggered):** Don't track what changed. Instead, compare desired state (spec) to actual state every time and take the next step.

**Requeue for polling:**
```go
// Requeue after 30 seconds
return ctrl.Result{RequeueAfter: 30 * time.Second}, nil

// Immediate requeue
return ctrl.Result{Requeue: true}, nil

// Done — don't requeue
return ctrl.Result{}, nil
```

**Owner references:** Set on child resources so they're garbage collected when the parent CR is deleted:
```go
ctrl.SetControllerReference(owner, child, r.Scheme)
```

**Predicates — filter watch events:**
```go
import "sigs.k8s.io/controller-runtime/pkg/predicate"

func (r *TrainingJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&mlv1alpha1.TrainingJob{}).
        Owns(&appsv1.Deployment{}).
        WithEventFilter(predicate.GenerationChangedPredicate{}). // skip status-only updates
        Complete(r)
}
```

## Helm Operator Details

The Helm operator reconciles CRs by running the embedded chart:

**`watches.yaml`:**
```yaml
- group: ml.example.com
  version: v1alpha1
  kind: ModelServer
  chart: helm-charts/model-server
  watchDependentResources: true
  overrideValues:
    replicas: spec.replicas
    gpu.enabled: spec.gpuEnabled
```

CR spec fields map directly to Helm values. The operator runs `helm upgrade --install` on each reconciliation.

**Override values from CR:**
```yaml
apiVersion: ml.example.com/v1alpha1
kind: ModelServer
metadata:
  name: llama-server
spec:
  replicas: 2
  gpuEnabled: true
  # All spec fields become Helm values
```

## Testing

### Unit Tests with envtest

`envtest` spins up a real API server and etcd for integration testing:

```go
var _ = Describe("TrainingJob Controller", func() {
    ctx := context.Background()

    It("should create a Deployment", func() {
        tj := &mlv1alpha1.TrainingJob{
            ObjectMeta: metav1.ObjectMeta{
                Name:      "test-job",
                Namespace: "default",
            },
            Spec: mlv1alpha1.TrainingJobSpec{
                Image:     "training:latest",
                Replicas:  2,
                GPUs:      1,
                Framework: "pytorch",
            },
        }
        Expect(k8sClient.Create(ctx, tj)).To(Succeed())

        // Eventually the controller should create a Deployment
        deploy := &appsv1.Deployment{}
        Eventually(func() error {
            return k8sClient.Get(ctx, types.NamespacedName{
                Name: "test-job", Namespace: "default",
            }, deploy)
        }, time.Second*10, time.Millisecond*250).Should(Succeed())

        Expect(*deploy.Spec.Replicas).To(Equal(int32(2)))
    })
})
```

```bash
make test  # runs envtest-based tests
```

### Scorecard

Validates operator bundle against best practices:

```bash
operator-sdk scorecard bundle/ \
  --kubeconfig ~/.kube/config \
  --namespace test-ns \
  --wait-time 120s
```

### E2E Testing

```bash
# Build and push operator image
make docker-build docker-push IMG=ghcr.io/myorg/my-operator:v0.1.0

# Deploy to a test cluster
make deploy IMG=ghcr.io/myorg/my-operator:v0.1.0

# Apply a test CR
kubectl apply -f config/samples/ml_v1alpha1_trainingjob.yaml

# Verify
kubectl get trainingjobs
kubectl get deployments
```

## OLM (Operator Lifecycle Manager)

Package your operator for distribution:

```bash
# Generate OLM bundle
make bundle IMG=ghcr.io/myorg/my-operator:v0.1.0

# Build and push bundle image
make bundle-build bundle-push \
  BUNDLE_IMG=ghcr.io/myorg/my-operator-bundle:v0.1.0

# Build catalog (index of bundles)
make catalog-build catalog-push \
  CATALOG_IMG=ghcr.io/myorg/my-operator-catalog:latest

# Install via OLM on a cluster
operator-sdk olm install  # install OLM itself
operator-sdk run bundle ghcr.io/myorg/my-operator-bundle:v0.1.0
```

**Bundle structure:**
```
bundle/
├── manifests/
│   ├── my-operator.clusterserviceversion.yaml  # CSV
│   ├── ml.example.com_trainingjobs.yaml        # CRD
│   └── my-operator-controller-manager_rbac.yaml
├── metadata/
│   └── annotations.yaml
└── tests/
    └── scorecard/
```

## CRD Versioning

When evolving your API (e.g., v1alpha1 → v1beta1):

```bash
# Create new version
operator-sdk create api --group ml --version v1beta1 --kind TrainingJob --resource --controller
```

**Conversion webhook** for multi-version support — add to the hub version:
```go
// api/v1beta1/trainingjob_conversion.go

// Hub marks v1beta1 as the storage version
func (*TrainingJob) Hub() {}

// api/v1alpha1/trainingjob_conversion.go

// ConvertTo converts v1alpha1 to the hub version (v1beta1)
func (src *TrainingJob) ConvertTo(dstRaw conversion.Hub) error {
    dst := dstRaw.(*v1beta1.TrainingJob)
    dst.Spec.Image = src.Spec.Image
    dst.Spec.Replicas = src.Spec.Replicas
    // ... map fields
    return nil
}

// ConvertFrom converts from the hub version to v1alpha1
func (dst *TrainingJob) ConvertFrom(srcRaw conversion.Hub) error {
    src := srcRaw.(*v1beta1.TrainingJob)
    dst.Spec.Image = src.Spec.Image
    // ... map fields
    return nil
}
```

## Debugging

See `references/troubleshooting.md` for:
- RBAC permission errors
- Infinite reconcile loops
- Watch and cache issues
- Leader election problems
- Common scaffolding mistakes

## Cross-References

- [kuberay](../kuberay/) — KubeRay operator as a real-world operator pattern
- [kueue](../kueue/) — Kueue as example of CRD-based job scheduling
- [flyte-deployment](../flyte-deployment/) — Flyte's operator-based K8s deployment

## Reference

- [Operator SDK docs](https://sdk.operatorframework.io/)
- [Kubebuilder book](https://book.kubebuilder.io/)
- [controller-runtime godoc](https://pkg.go.dev/sigs.k8s.io/controller-runtime)
- [API conventions](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md)
- `references/troubleshooting.md` — common errors and fixes
