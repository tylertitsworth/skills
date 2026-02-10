#!/usr/bin/env bash
# Scaffold a new Kubernetes operator with Operator SDK.
# Usage: ./scaffold.sh <domain> <group> <kind>
# Example: ./scaffold.sh example.com training TrainingJob
set -euo pipefail

DOMAIN="${1:?Usage: $0 <domain> <group> <kind>}"
GROUP="${2:?Usage: $0 <domain> <group> <kind>}"
KIND="${3:?Usage: $0 <domain> <group> <kind>}"

PROJECT_NAME="${GROUP}-operator"

echo "==> Creating operator project: ${PROJECT_NAME}"
mkdir -p "${PROJECT_NAME}" && cd "${PROJECT_NAME}"

operator-sdk init \
  --domain="${DOMAIN}" \
  --repo="github.com/myorg/${PROJECT_NAME}" \
  --plugins=go/v4

echo "==> Creating API: ${GROUP}/${KIND}"
operator-sdk create api \
  --group="${GROUP}" \
  --version=v1alpha1 \
  --kind="${KIND}" \
  --resource --controller

echo "==> Scaffolding complete."
echo ""
echo "Next steps:"
echo "  1. Edit api/v1alpha1/${KIND,,}_types.go — define your CRD spec/status"
echo "  2. Edit internal/controller/${KIND,,}_controller.go — implement reconcile logic"
echo "  3. Run: make generate manifests  — regenerate deepcopy and CRD YAML"
echo "  4. Run: make docker-build docker-push IMG=<registry>/<name>:<tag>"
echo "  5. Run: make deploy IMG=<registry>/<name>:<tag>"
