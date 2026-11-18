# PathSnap

GPS-to-directed-road matching: validated raw inputs → metric projection → spatial candidates → 22 relative features → saved classifier → directed Viterbi → route reconstruction → private scoring → FastAPI → separate Streamlit/Folium frontend.

Python 3.12 is the tested version. The frontend laptop needs no data, model, GPU or backend installation. Supplied raw and reserved data stay outside public Git commits.

## Laptop 1: setup and startup

```bash
bash scripts/setup_backend.sh
export PATHSNAP_DATA_ROOT="$(cd .. && pwd)"
bash scripts/start_backend.sh
```

The data root contains `train/` and `student_test/`. Alternatively copy `.env.example` to `.env` and edit it. The server binds `0.0.0.0:8000`; that is a bind address. Connect using Laptop 1's actual LAN IPv4 address.

PowerShell: run `scripts/setup_backend.ps1`, set `$env:PATHSNAP_DATA_ROOT`, then run `scripts/start_backend.ps1`. See `docs/BACKEND_LAPTOP_SETUP.md`.

## Retrain everything, tune, export, verify and package

Install training dependencies once:

```bash
.venv/bin/python -m pip install -r requirements-train.txt
```

Run one complete cycle:

```bash
.venv/bin/python scripts/run_full_pipeline.py --data-root .. --device cuda --trials 15 --resume
```

For explicit CPU training use `--device cpu`. `--device cuda` fails clearly if GPU execution is unavailable rather than silently claiming GPU training. Environment probes use tiny synthetic workloads only.

The runner loops over all five families and three trip-disjoint CV folds, refits each on all 80 training trips, compares saved OOF/student-test predictions, selects by training mean-trip observation Top-1, tunes the winning family with persistent TPE, compares tuned/untuned results, exports a checksummed CPU-portable inference artifact, runs tests, exercises all ten development trips over real HTTP, generates the consolidated PDF, and packages both laptops. `--cycles N` repeats a bounded cycle (1–10). `--resume` reuses completed compatible training runs. Backend requests never train models.

Existing-model verification without training/tuning:

```bash
.venv/bin/python scripts/run_full_pipeline.py --data-root .. --skip-training --skip-tuning --skip-audit --skip-prepare
```

A failed stage stops the run; `outputs/pipeline_state.json` records the stage and status. Fix the cause and rerun. Do not launch concurrent training pipelines writing the same output directories.

## Individual commands

```bash
.venv/bin/python scripts/check_environment.py --role training --require-gpu
.venv/bin/python scripts/check_data.py --data-root ..
.venv/bin/python -m pathsnap.cli prepare --data-root ..
.venv/bin/python scripts/train_model.py --model xgboost --device cuda --resume
.venv/bin/python scripts/evaluate_models.py
.venv/bin/python scripts/tune_and_export.py --trials 15 --startup 5 --device cuda
.venv/bin/python scripts/export_model.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_runtime.py --data-root ..
.venv/bin/python scripts/generate_report.py
.venv/bin/python scripts/package_dist.py
```

Training family names: `logistic`, `decision_tree`, `random_forest`, `xgboost`, `pytorch_mlp`. Training uses only resolved candidate rows from training trips; student-test data is development evaluation. Tuning supports all five families and retains the untuned configuration when it wins under the recorded rule. TPE's search history is retained in SQLite, CSV and JSON; `--trials` is the total study target, not extra trials on every restart. New feature/model/search identities create independent studies.

`export_model.py` explicitly exports the untuned baseline. After successful tuning, the tuning script automatically exports its chosen tuned/untuned result; do not overwrite that choice with the baseline export command unless intentional.

## Laptop 2: frontend only

Copy and extract `dist/PathSnap_Frontend.zip`. From the extracted folder:

```bash
bash scripts/setup_frontend.sh
export BACKEND_URL="http://<LAPTOP_1_LAN_IP>:8000"
bash scripts/start_frontend.sh
```

Open `http://localhost:8501`. Select a trip, run matching, inspect the actual decoded route and scores, then select/run another trip. Results survive map interactions and ordinary reruns. Polling is bounded; failures release the controls for retry. The default map has no tiles and embeds vendored Leaflet JS/CSS, so it requires no internet map assets.

Windows PowerShell:

```powershell
.\scripts\setup_frontend.ps1
$env:BACKEND_URL="http://<LAPTOP_1_LAN_IP>:8000"
.\scripts\start_frontend.ps1
```

## Endpoint contract

- `GET /health`: actual loaded-model and data readiness; model version and mode, no local paths.
- `GET /trips`: configured allowlisted IDs.
- `POST /predict/{trip_id}`: HTTP 202 and job ID; 404 unknown trip; 503 missing model/data; 429 another job active.
- `GET /jobs/{job_id}`: queued/running/completed/failed, stage and safe errors.
- `GET /results/{job_id}`: actual public input/predicted geometry, one match/status per observation, timing, continuity and aggregate scores.

One backend process/worker is supported. It allows one active prediction, retains at most 32 recent jobs, logs to stdout and rotating `outputs/logs/backend.log`, and expires in-memory jobs on restart. A restarted or expired job returns 404; resubmit it. UI timeout defaults to 900 seconds. Cooperative backend time limits are checked between stages; bounded routing prevents unbounded all-pairs searches.

## Reserved evaluation

Freeze/export the selected artifact before supplying reserved data. Set `PATHSNAP_MODE=evaluation` and `PATHSNAP_EVALUATION_DATA_ROOT` to the directory directly containing the ten complete reserved trip directories. Alternatively set `PATHSNAP_EVALUATION_MANIFEST` to a JSON object with exactly ten `trips` entries, each containing `trip_id` and a relative `directory`. `PATHSNAP_REFERENCE_ROOT` may hold references separately as `<root>/<trip_id>/<trip_id>.route`.

Prediction opens only explicit track/nodes/arcs paths. Private scoring reads references after prediction and returns aggregate metrics. No reference arcs, coordinates, aligned labels or pointwise correctness enter API responses or the frontend. Missing references produce unavailable scores, never fake zeros. `scripts/evaluate_evaluator_test.py` evaluates only the frozen selected model, never selects among five models. Do not use reserved scores to tune.

## Evidence and limitations

- `outputs/audit/`: inventory and hashes for 360 development files / 90 trips.
- `outputs/prepared/`: data/config/schema-keyed caches and final retained candidate recall.
- `outputs/experiments/models/<family>/`: OOF predictions, fold trip assignments, per-trip metrics, fit/device metadata, genuine MLP histories and final weights.
- `outputs/experiments/student_test/`: comparison, per-trip metrics and explicit selection rule.
- `outputs/experiments/tuning_v2/`: persistent study, search distributions, trials, best parameters, tuned-vs-untuned comparison.
- `artifacts/`: selected portable model, checksum, feature schema and prediction configuration.
- `outputs/verification/`: actual API loop results, logs, browser screenshots and checks.
- `reports/PathSnap_Technical_Report.pdf`: consolidated evidence-based development report.
- `dist/`: complete backend/frontend ZIPs with checksums and per-file manifests.

Observation scoring is provisional ordered-reference alignment, not an organiser-confirmed pointwise convention. Directed arc IDs are trip-local. Missed candidates count as wrong on known labels; undefined labels are excluded from accuracy with visible coverage. Route-set F1 and ordered edit distance are separate metrics. Full-trip features are offline, not causal. No legal turn restriction data is available.

Physical Laptop 2 networking, Windows execution, and official organiser scoring require those external environments; localhost checks do not establish them. New full-data retraining/tuning remains the user's action. Existing saved-model inference results and synthetic training regression tests are identified separately. AI assistance was used for implementation, debugging and report generation; the team must verify and understand the submission.

Deployment and supplied ZIP archives are intentionally excluded from GitHub. Generate current deployment packages locally with `python scripts/package_dist.py`.

## React frontend from the supplied archive

The extracted React/Vite/TypeScript application is in `frontend-web/`. It displays the backend's actual decoded/classifier/network geometry and derived-reference scores. The Streamlit frontend remains available separately.

```bash
cd frontend-web
npm ci
export VITE_API_BASE_URL="http://<LAPTOP_1_LAN_IP>:8000"
npm run dev
# TypeScript and production build:
npm run build
```

Open the Vite URL (normally http://localhost:5173). Browser requests use its same-origin `/api` development proxy. Configure an equivalent reverse proxy when serving production build files. Dependencies, `.env.local`, build output and ZIP archives are excluded from GitHub.

`outputs/verification/react_browser_checks.json` records the local browser integration check: ten available trip choices, an actual backend prediction, decoded geometry, an aggregate score, and a repeated request passed. This is not physical two-laptop verification.

The consolidated mentor report is `reports/PathSnap_Technical_Report.pdf`. Regenerate it from current saved evidence using `python scripts/generate_report.py`.
