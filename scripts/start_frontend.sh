#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BACKEND_URL:?Set BACKEND_URL=http://<Laptop 1 LAN IP>:8000}"
export BACKEND_URL
.venv-frontend/bin/python scripts/check_connection.py
exec .venv-frontend/bin/python -m streamlit run frontend/app/app.py --server.address 127.0.0.1 --server.port "${UI_PORT:-8501}"
