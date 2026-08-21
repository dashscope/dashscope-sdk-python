#!/usr/bin/env bash
# Sync the standalone agenticCLI-examples repo into dashscope/acli/examples.
# Examples are maintained in agenticCLI-examples; after modifying them there,
# run this script to vendor the updates here.
# Usage: scripts/sync_examples.sh [path-to-agenticCLI-examples]
set -euo pipefail

SRC_REPO="${1:-/Users/zhansheng.lzs/ali/pro/ptm/agenticCLI-examples}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$REPO_ROOT/dashscope/acli/examples"

if [ ! -d "$SRC_REPO" ]; then
    echo "error: source not found: $SRC_REPO" >&2
    exit 1
fi

echo "==> Sync $SRC_REPO -> $DST"
rsync -a --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    "$SRC_REPO/" "$DST/"

echo "==> Check for internal references in synced examples"
if grep -rnE 'alibaba-inc|gitlab\.alibaba|code\.alibaba' "$DST"; then
    echo "error: internal references found in examples (see above);" >&2
    echo "       fix them in agenticCLI-examples before syncing" >&2
    exit 1
fi

echo "==> Done"
