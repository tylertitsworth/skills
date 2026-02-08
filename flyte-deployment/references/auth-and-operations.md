# Flyte Authentication & Operations

## Table of Contents

- [Authentication (OAuth2/OIDC)](#authentication-oauth2oidc)
- [Ingress and DNS](#ingress-and-dns)
- [Upgrades](#upgrades)
- [Monitoring](#monitoring)
- [Scaling FlytePropeller](#scaling-flytepropeller)
- [Troubleshooting](#troubleshooting)

## Authentication (OAuth2/OIDC)

Flyte supports OAuth 2.0 with OIDC providers (Okta, Auth0, Google, Keycloak, etc.).

### Configuration

```yaml
configuration:
  inline:
    server:
      security:
        secure: false
        useAuth: true
        allowCors: true
        allowedOrigins:
          - "https://flyte.example.com"
      auth:
        appAuth:
          thirdPartyConfig:
            flyteClient:
              clientId: flytectl
              redirectUri: http://localhost:53593/callback
              scopes:
                - offline
                - all
          selfAuthServer:
            staticClients:
              flyte-cli:
                id: flyte-cli
                redirect_uris:
                  - http://localhost:53593/callback
                grant_types:
                  - refresh_token
                  - authorization_code
                response_types:
                  - code
                scopes:
                  - all
                  - offline
                  - access_token
                public: true
              flytepropeller:
                id: flytepropeller
                redirect_uris:
                  - http://localhost:3846/callback
                grant_types:
                  - refresh_token
                  - client_credentials
                response_types:
                  - token
                scopes:
                  - all
                  - offline
                  - access_token
        authorizedUris:
          - https://flyte.example.com
          - http://flyte-binary:8088
          - http://flyte-binary:8089
        userAuth:
          openId:
            baseUrl: https://accounts.google.com  # or your OIDC provider
            clientId: <your-client-id>
            scopes:
              - profile
              - openid

  # Store OIDC client secret
  secrets:
    adminOauthClientCredentials:
      clientSecret: <your-client-secret>
```

### Client Configuration

After enabling auth, update flytectl config:

```yaml
# ~/.flyte/config.yaml
admin:
  endpoint: dns:///flyte.example.com
  authType: Pkce
  insecure: false
```

## Ingress and DNS

Enable ingress in Helm values:

```yaml
ingress:
  create: true
  ingressClassName: nginx
  commonAnnotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  httpAnnotations:
    nginx.ingress.kubernetes.io/app-root: /console
  host: flyte.example.com
  tls:
    enabled: true
```

Flyte uses two ports:
- **8088**: HTTP (console + REST API)
- **8089**: gRPC (flytectl, pyflyte)

Both need to be accessible through ingress. Some ingress controllers need separate rules for gRPC.

## Upgrades

```bash
helm repo update flyteorg
helm upgrade flyte-backend flyteorg/flyte-binary \
  --namespace flyte --values values.yaml
```

**Notes:**
- Upgrades follow semantic versioning — expect significant changes in minor versions
- Database migrations run automatically on startup
- For multi-cluster deployments, upgrade all components together
- Always `--dry-run` first

## Monitoring

### Prometheus Metrics

Flyte components expose Prometheus metrics:

```bash
# FlytePropeller metrics
kubectl port-forward -n flyte deploy/flyte-binary 10254:10254
# http://localhost:10254/metrics
```

Key metrics:
| Metric | Purpose |
|---|---|
| `flyte:propeller:all:round:raw_ms` | Propeller round latency |
| `flyte:propeller:all:node:success_duration_ms` | Node execution time |
| `flyte:propeller:all:node:failures_total` | Node failure count |
| `flyte:admin:api:request_duration_ms` | Admin API latency |

### Notifications

Configure Slack/email notifications for workflow events:

```yaml
configuration:
  inline:
    notifications:
      type: aws  # or gcp, sandbox
      aws:
        region: us-west-2
```

## Scaling FlytePropeller

### Sharding (Automatic Scale-Out)

For large clusters, FlytePropeller can shard workflow processing:

```yaml
flyteadmin:
  additionalContainerEnv:
    - name: FLYTE_PROPELLER_MANAGER_SHARD_COUNT
      value: "4"
```

This runs multiple FlytePropeller instances, each responsible for a subset of workflows.

### Performance Tuning

```yaml
configuration:
  inline:
    propeller:
      rawoutput-prefix: s3://flyte-data/
      workers: 40                        # parallel workflow processors
      workflow-reeval-duration: 10s       # how often to re-evaluate workflows
      max-workflow-retries: 50
      kube-client-config:
        qps: 100
        burst: 25
```

## Troubleshooting

### FlytePropeller Stuck

**Symptoms:** Workflows stay in "Running" state, no progress.

```bash
# Check propeller logs
kubectl logs -n flyte deploy/flyte-binary -c flyte-binary-propeller --tail=200

# Check propeller metrics
kubectl port-forward -n flyte deploy/flyte-binary 10254:10254
curl http://localhost:10254/metrics | grep propeller
```

**Common causes:**
1. **Database connection issues** — Check PostgreSQL connectivity
2. **Object store access denied** — Verify IAM permissions for S3/GCS
3. **Kubernetes API throttling** — Increase `kube-client-config.qps` and `burst`
4. **Too many concurrent workflows** — Increase `workers` or enable sharding

### Pod Failures

```bash
# Check task pod events
kubectl describe pod <task-pod-name> -n <project-domain>

# Common namespace pattern: <project>-<domain>
kubectl get pods -n ml-team-production
```

**Common causes:**
- OOM killed → Increase task resource requests
- ImagePullBackOff → Check image name/registry credentials
- Pending → Insufficient cluster resources or node selectors don't match

### Storage Errors

**"Failed to upload"** or **"Failed to download"**:
1. Check IAM role/service account permissions
2. Verify bucket names in configuration
3. Check network connectivity from pods to storage endpoint
4. For Minio: verify endpoint URL and credentials

### Database Migrations

Migrations run automatically on startup. If they fail:

```bash
# Check admin logs for migration errors
kubectl logs -n flyte deploy/flyte-binary -c flyte-binary-admin --tail=200 | grep -i migration

# Manual migration (rarely needed)
kubectl exec -n flyte deploy/flyte-binary -- flyteadmin migrate run
```

### Workflow Registration Failures

```bash
# Check registration output
flytectl register files --archive pkg.tgz --project my-proj --domain dev --version v1

# Common issues:
# - Wrong admin endpoint in ~/.flyte/config.yaml
# - Authentication failure (token expired)
# - Image not accessible from cluster
# - Version already registered (use unique versions)
```

### Useful Diagnostics

```bash
# Check all Flyte pods
kubectl get pods -n flyte

# Admin logs
kubectl logs -n flyte deploy/flyte-binary -c flyte-binary --tail=100

# Propeller logs
kubectl logs -n flyte deploy/flyte-binary -c flyte-binary --tail=100 | grep propeller

# Check CRDs
kubectl get flyteworkflows -A

# Database connectivity
kubectl exec -n flyte deploy/flyte-binary -- pg_isready -h <db-host>
```
