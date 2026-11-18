$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONPATH="$PWD\src;$PWD"
$bindHost = '0.0.0.0'
$bindPort = '8000'
if ($env:PATHSNAP_HOST) { $bindHost = $env:PATHSNAP_HOST }
if ($env:PATHSNAP_PORT) { $bindPort = $env:PATHSNAP_PORT }
& .\.venv\Scripts\python.exe scripts/check_environment.py --role backend
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed' }
$envArgs = @()
if (Test-Path .env) { $envArgs = @('--env-file', '.env') }
& .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host $bindHost --port $bindPort @envArgs
