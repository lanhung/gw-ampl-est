#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/configs/remote/autodl.env"

mkdir -p "$PROJECT_ROOT/results/remote"

rsync -az --itemize-changes \
  --include='*/' \
  --include='*.csv' \
  --include='*.json' \
  --include='*.yaml' \
  --include='*.yml' \
  --include='*.md' \
  --include='*.txt' \
  --include='*.log' \
  --include='*.png' \
  --include='*.pdf' \
  --exclude='*' \
  "$AUTODL_HOST:$AUTODL_RUNS_ROOT/" \
  "$PROJECT_ROOT/results/remote/"
