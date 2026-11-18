# Laptop 1

Use Python 3.12. Extract the backend ZIP, then place authorised data separately. Set `PATHSNAP_DATA_ROOT` to the directory directly containing `train/` and `student_test/`.

```bash
bash scripts/setup_backend.sh
export PATHSNAP_DATA_ROOT="/absolute/path/to/Student folder"
bash scripts/start_backend.sh
```

```powershell
.\scripts\setup_backend.ps1
$env:PATHSNAP_DATA_ROOT="C:\path\to\Student folder"
.\scripts\start_backend.ps1
```

The server binds 0.0.0.0:8000. Verify `curl http://127.0.0.1:8000/health`. Readiness requires a loaded compatible model and complete input files. The package includes selected weights and registry; no training occurs at startup.

Find Laptop 1's reachable LAN IPv4 address using `hostname -I` on Linux or `ipconfig` on Windows. On Laptop 2 use `http://<that_IP>:8000`; 0.0.0.0 is not a client URL. Use one uvicorn worker. Logs are in `outputs/logs/backend.log`.

For retraining, use the full repository, install `requirements-train.txt`, and run `scripts/run_full_pipeline.py --data-root <path> --device cuda --trials 15 --resume`. CUDA/PyTorch installation must match the laptop's driver; `scripts/check_environment.py --role training --require-gpu` performs genuine small GPU probes.

Selected MLP deployments additionally need PyTorch. The package generates `requirements-selected.txt` for the selected family. XGBoost and sklearn models can serve on CPU. No database or paid service is required.
