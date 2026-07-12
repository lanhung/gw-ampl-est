#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/configs/remote/autodl.env"

RSYNC_ARGS=(
  -az
  --itemize-changes
  --prune-empty-dirs
  --exclude=.git/
  --exclude=.venv/
  --exclude=.mypy_cache/
  --exclude=.pytest_cache/
  --exclude=.ruff_cache/
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude=/data/
  --exclude=/data_v2/
  --exclude=/gwosc_cache/
  --exclude=/runs/
  --exclude=/cache/
  --exclude=/checkpoints/
  --exclude='*.npy'
  --exclude='*.npz'
  --exclude='*.h5'
  --exclude='*.hdf5'
  --exclude='*.gwf'
  --exclude=configs/remote/autodl.env
)

if [[ "${1:-}" == "--dry-run" ]]; then
  RSYNC_ARGS+=(-n)
fi

rsync "${RSYNC_ARGS[@]}" \
  "$PROJECT_ROOT/" \
  "$AUTODL_HOST:$AUTODL_REPO_ROOT/"
