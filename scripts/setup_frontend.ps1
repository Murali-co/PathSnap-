$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (!(Test-Path '.venv-frontend')) { py -3.12 -m venv .venv-frontend }
& .\.venv-frontend\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-frontend\Scripts\python.exe -m pip install -r frontend/requirements.txt
