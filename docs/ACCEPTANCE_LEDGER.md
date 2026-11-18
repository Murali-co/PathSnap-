# Acceptance ledger

| Requirement | State | Evidence |
|---|---|---|
| Official documents reviewed | Completed | Supplied handoff + five PDF + DOCX; report references |
| Development data integrity | Completed | 360 matching file hashes, outputs/audit/ |
| Features/geometry/candidates | Audited and tested | docs/FEATURE_AUDIT.md, tests/test_data_geometry.py |
| Reference alignment | Implemented, provisional | reference.py; scoring assumptions; official convention unresolved |
| Five-family training/CV | User-supplied fits present; repaired code tested synthetically | outputs/experiments/models/, tests/test_training.py |
| New full-data retraining | User action pending | run_full_pipeline.py |
| Generic Bayesian tuning | Implemented; new full-data run pending | tune_and_export.py, SQLite/trial outputs on execution |
| HMM/reconstruction | Fixed and tested on actual development predictions | outputs/verification/api_trip_metrics.csv |
| API/Frontend prediction loop | Locally executed | verify_runtime.py, browser checks where present |
| Reference isolation | Tested | tests/test_api.py, no reference fields in API results |
| Offline map | Implemented and asset-tested | frontend/assets/, tests/test_frontend.py |
| Packaging | Script validates required files/artifact/checksums | dist/dist_manifest.json |
| Physical LAN/Windows | Pending physical hardware | docs/TWO_LAPTOP_DEMO.md |
| Reserved evaluation | Provisioning implemented; final checks pending | evaluation mode; no development reads of reserved references |
| Consolidated report | Development PDF generated from actual evidence | reports/PathSnap_Technical_Report.pdf |
