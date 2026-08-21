#!/usr/bin/env bash
# Sync agenticCLI src/acli into dashscope/acli with import path rewrite.
# Usage: scripts/sync_acli.sh [path-to-agenticCLI]
set -euo pipefail

SRC_REPO="${1:-/Users/zhansheng.lzs/ali/pro/ptm/agenticCLI}"
SRC="$SRC_REPO/src/acli"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$REPO_ROOT/dashscope/acli"

if [ ! -d "$SRC" ]; then
    echo "error: source not found: $SRC" >&2
    exit 1
fi

echo "==> Sync $SRC -> $DST"
# Preserve local examples when the source repo no longer ships them.
KEPT_EXAMPLES=""
if [ ! -d "$SRC_REPO/examples" ] && [ -d "$DST/examples" ]; then
    KEPT_EXAMPLES="$(mktemp -d)/examples"
    mv "$DST/examples" "$KEPT_EXAMPLES"
fi
rm -rf "$DST"
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' "$SRC/" "$DST/"

if [ -d "$SRC_REPO/examples" ]; then
    echo "==> Sync $SRC_REPO/examples -> $DST/examples"
    rm -rf "$DST/examples"
    rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' "$SRC_REPO/examples/" "$DST/examples/"
elif [ -n "$KEPT_EXAMPLES" ]; then
    echo "==> Source has no examples/; keeping existing $DST/examples"
    mv "$KEPT_EXAMPLES" "$DST/examples"
fi

echo "==> Rewrite imports acli.* -> dashscope.acli.*"
find "$DST" -name '*.py' -print0 | xargs -0 sed -i '' -E '
    s/^([[:space:]]*)from acli\./\1from dashscope.acli./g;
    s/^([[:space:]]*)from acli (import)/\1from dashscope.acli \2/g;
    s/^([[:space:]]*)import acli\./\1import dashscope.acli./g;
    s/^([[:space:]]*)import acli$/\1import dashscope.acli as acli/g;
    s/"acli\.cli\."/"dashscope.acli.cli."/g;
    s/'"'"'acli\.cli\.'"'"'/'"'"'dashscope.acli.cli.'"'"'/g;
'

echo "==> Check for leftover bare acli imports"
if grep -rnE '^[[:space:]]*(from|import) acli(\.|[[:space:]]|$)' "$DST" --include='*.py'; then
    echo "error: unrewritten imports remain (see above)" >&2
    exit 1
fi

# The acli. -> dashscope.acli. rewrite makes import lines 10 chars longer,
# so the synced tree must be re-formatted with the repo's pinned black
# (23.3.0 --line-length=79 via pre-commit) to match CI's --all-files run.
echo "==> Format synced tree (black via pre-commit, matching CI)"
cd "$REPO_ROOT"
if command -v pre-commit >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    pre-commit run black --files $(find dashscope/acli -name '*.py') || true
else
    echo "warn: pre-commit not on PATH; synced tree not auto-formatted" >&2
fi

echo "==> Verify all modules import"
cd "$REPO_ROOT"
python - <<'EOF'
import pkgutil, importlib, sys
import dashscope.acli
failed = []
for m in pkgutil.walk_packages(dashscope.acli.__path__, prefix="dashscope.acli."):
    try:
        importlib.import_module(m.name)
    except Exception as e:
        failed.append((m.name, e))
if failed:
    for name, e in failed:
        print(f"FAIL {name}: {e}", file=sys.stderr)
    sys.exit(1)
print(f"OK: all modules under dashscope.acli import cleanly")
EOF

echo "==> Done"
