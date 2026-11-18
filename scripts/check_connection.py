#!/usr/bin/env python3
import argparse,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontend'))
from client import APIClient
p=argparse.ArgumentParser();p.add_argument('--backend-url',default=os.getenv('BACKEND_URL'));p.add_argument('--timeout',type=float,default=10);a=p.parse_args()
if not a.backend_url:raise SystemExit('Set BACKEND_URL=http://<Laptop 1 LAN IP>:8000 or pass --backend-url.')
try:
    result=APIClient(a.backend_url,a.timeout).request('/health');print(result)
    if not result.get('artifacts_ready') or not result.get('data_ready'):raise SystemExit('Connected, but backend needs model artifacts or data.')
except (ValueError,RuntimeError) as exc:raise SystemExit(str(exc))
