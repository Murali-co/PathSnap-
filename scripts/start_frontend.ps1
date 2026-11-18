$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (!$env:BACKEND_URL) { throw 'Set BACKEND_URL=http://<Laptop 1 LAN IP>:8000' }
& .\.venv-frontend\Scripts\python.exe scripts/check_connection.py
if ($LASTEXITCODE -ne 0) { throw 'Backend connection/readiness check failed' }
$uiPort = '8501'
if ($env:UI_PORT) { $uiPort = $env:UI_PORT }
& .\.venv-frontend\Scripts\python.exe -m streamlit run frontend/app/app.py --server.address 127.0.0.1 --server.port $uiPort
