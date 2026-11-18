#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-frontend
.venv-frontend/bin/python -m pip install --upgrade pip
.venv-frontend/bin/python -m pip install -r frontend/requirements.txt
