# Laptop 2

Extract `dist/PathSnap_Frontend.zip`. This package contains frontend source, Leaflet assets/license, dependencies and scripts. It contains no data, references, models, CUDA or backend source.

```bash
bash scripts/setup_frontend.sh
export BACKEND_URL="http://<LAPTOP_1_LAN_IP>:8000"
bash scripts/start_frontend.sh
```

```powershell
.\scripts\setup_frontend.ps1
$env:BACKEND_URL="http://<LAPTOP_1_LAN_IP>:8000"
.\scripts\start_frontend.ps1
```

Open http://localhost:8501. Check backend readiness, choose any configured trip, click Match and reconstruct, then inspect the actual route/layers/metrics. Repeat for another trip; completed results remain available when you return to that selection.

`REQUEST_TIMEOUT_SECONDS` defaults to 30 and `JOB_TIMEOUT_SECONDS` to 900. Errors release controls for retry. The vector-road map embeds local JS/CSS and has no tile requests. Network HTTP requests originate from the frontend Python process, so Streamlit requires no browser CORS exception.

Windows commands were prepared for PowerShell 5.1+; physical Windows and friend-laptop execution remain external checks. Use Python 3.12.
