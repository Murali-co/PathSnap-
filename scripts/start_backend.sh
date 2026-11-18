#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD"
.venv/bin/python scripts/check_environment.py --role backend
env_args=()
if [[ -f .env ]]; then env_args=(--env-file .env); fi
exec .venv/bin/python -m uvicorn backend.app.main:app --host "${PATHSNAP_HOST:-0.0.0.0}" --port "${PATHSNAP_PORT:-8000}" "${env_args[@]}"
