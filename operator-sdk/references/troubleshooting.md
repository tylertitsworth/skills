# Operator SDK Troubleshooting

## RBAC Permission Errors

### "is forbidden: User ... cannot get resource"

The operator's ServiceAccount lacks permissions for the resources it tries to manage.

**Fix**: Add RBAC markers to your reconciler and regenerate:
```go
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch;delete
```

Then:
```bash
make manifests  # regenerates config/rbac/role.yaml
make deploy IMG=...
```

### Missing RBAC for status subresource

If you update `.status` but get forbidden errors, ensure you have the status RBAC marker:
```go
// +kubebuilder:rbac:groups=ml.example.com,resources=trainingjobs/status,verbs=get;update;patch
```

### Missing RBAC for finalizers

```go
// +kubebuilder:rbac:groups=ml.example.com,resources=trainingjobs/finalizers,verbs=update
```

### Checking effective RBAC

```bash
# View the ClusterRole
kubectl get clusterrole my-operator-manager-role -o yaml

# Test permissions
kubectl auth can-i get deployments --as=system:serviceaccount:my-operator-system:my-operator-controller-manager
```

## Infinite Reconcile Loops

### Status update triggers reconcile

Updating `.status` triggers a new event, which triggers reconciliation again.

**Fix**: Use `GenerationChangedPredicate` — the `generation` field only increments on spec changes, not status:
```go
func (r *Reconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&mlv1alpha1.TrainingJob{}).
        WithEventFilter(predicate.GenerationChangedPredicate{}).
        Complete(r)
}
```

### Updating the CR itself in reconcile

If you update the CR object (not just status) inside Reconcile, it increments `.metadata.generation`, which triggers another reconcile.

**Fix**: Only update `.status` via the status subresource. If you must update the CR (e.g., to add a finalizer), do it early and return immediately — the next reconcile handles the rest:
```go
if !controllerutil.ContainsFinalizer(&tj, finalizerName) {
    controllerutil.AddFinalizer(&tj, finalizerName)
    if err := r.Update(ctx, &tj); err != nil {
        return ctrl.Result{}, err
    }
    return ctrl.Result{}, nil  // return immediately, next reconcile continues
}
```

### Child resource updates trigger reconcile

If your operator `Owns()` child resources, any change to those children triggers reconciliation. This is usually correct, but if children are frequently updated (e.g., Pods with status changes), it can cause rapid reconciliation.

**Fix**: Filter child events:
```go
ctrl.NewControllerManagedBy(mgr).
    For(&mlv1alpha1.TrainingJob{}).
    Owns(&appsv1.Deployment{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
    Complete(r)
```

## Watch and Cache Issues

### "no kind is registered for the type"

The type isn't registered in the scheme.

**Fix**: Add to `cmd/main.go`:
```go
import routev1 "github.com/openshift/api/route/v1"

func init() {
    utilruntime.Must(routev1.AddToScheme(scheme))
}
```

### Cache not synced / stale reads

The controller-runtime client uses a **cached** reader by default. After creating an object, an immediate `Get()` may not find it.

**Fix**: Use `client.Reader` for direct API reads when needed:
```go
// In the reconciler struct, add a direct client
type Reconciler struct {
    client.Client
    DirectClient client.Reader  // set to mgr.GetAPIReader()
}

// Direct read (bypasses cache)
r.DirectClient.Get(ctx, key, obj)
```

Or wait for cache sync by requeuing:
```go
if err := r.Get(ctx, key, found); errors.IsNotFound(err) {
    // Might not be in cache yet, requeue
    return ctrl.Result{RequeueAfter: time.Second}, nil
}
```

### Watching resources in specific namespaces

By default, the manager watches all namespaces. To restrict:
```go
mgr, err := ctrl.NewManager(cfg, ctrl.Options{
    Cache: cache.Options{
        DefaultNamespaces: map[string]cache.Config{
            "ml-workloads": {},
            "training":     {},
        },
    },
})
```

## Leader Election

### "failed to acquire lease"

Only one replica should run the reconciler at a time. If the old leader's pod is stuck, the lease may not have expired.

**Check lease:**
```bash
kubectl get lease -n my-operator-system
kubectl describe lease my-operator-controller-manager -n my-operator-system
```

**Fix**: Delete the stale lease:
```bash
kubectl delete lease my-operator-controller-manager -n my-operator-system
```

### Disable leader election for development

```go
mgr, err := ctrl.NewManager(cfg, ctrl.Options{
    LeaderElection: false,
})
```

Or via flag: `--leader-elect=false`

## Scaffolding Issues

### "operator-sdk init" fails with Go module errors

Ensure `GO111MODULE=on` and you're outside `$GOPATH/src`, or use `--repo`:
```bash
# Set GO111MODULE=on in container env
operator-sdk init --domain example.com --repo github.com/myorg/my-op
```

### "make generate" fails

Usually missing tools. Install them:
```bash
make controller-gen  # downloads controller-gen
make kustomize       # downloads kustomize
```

Or manually:
```bash
go install sigs.k8s.io/controller-tools/cmd/controller-gen@latest
```

### "make manifests" doesn't update CRD

Ensure your `+kubebuilder:` markers are on exported types and the comment is on the line directly above the field (no blank lines).

## Build and Deploy Issues

### Operator crashes on startup

Common causes:
1. **Missing CRD**: Install CRDs first: `make install`
2. **Wrong image**: Verify `IMG` matches what's deployed
3. **Missing RBAC**: Check the manager logs: `kubectl logs -n my-operator-system deployment/my-operator-controller-manager`
4. **Certificate issues**: Webhook certs not ready. Use cert-manager or disable webhooks for testing.

### Operator runs but doesn't reconcile

1. Check the CR exists: `kubectl get trainingjobs`
2. Check logs for errors: `kubectl logs -l control-plane=controller-manager -n my-operator-system`
3. Verify the controller is registered in `cmd/main.go`
4. Check the `For()` type matches the CR's GVK exactly

## Helm Operator Issues

### "release not found" on first reconcile

The Helm operator creates releases named after the CR. Ensure the CR name is a valid Helm release name (lowercase, no special chars).

### Values not propagating

Check `watches.yaml` — `overrideValues` maps CR spec fields to chart values:
```yaml
overrideValues:
  image.tag: spec.version  # CR's .spec.version → chart's .Values.image.tag
```

## Ansible Operator Issues

### Playbook not found

Check `watches.yaml` references the correct role or playbook path:
```yaml
- group: ml.example.com
  version: v1alpha1
  kind: Notebook
  role: notebook  # looks for roles/notebook/
```

### Variables not available in playbook

CR spec fields are available as Ansible variables with the same names. Nested fields use underscores:
```yaml
# CR spec: { gpuCount: 2 }
# In Ansible: {{ gpu_count }}  (camelCase → snake_case)
```
