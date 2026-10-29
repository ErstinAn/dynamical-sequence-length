#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "${DYNAMIC_SEQ_LEN_PYTHON:-python}" "$HERE/scripts/train.py" "$@"
