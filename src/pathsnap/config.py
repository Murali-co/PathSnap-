from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "seed": 42,
    "radii_m": [50, 100, 200, 300],
    "max_candidates": 16,
    "gap_seconds": 60,
    "max_speed_mps": 55,
    "movement_slack_m": 50,
    "backward_tolerance_m": 12,
    "transition_beta_m": 35,
    "uturn_penalty": 1.0,
    "folds": 3,
    "trials": 15,
    "startup_trials": 5,
    "label_method": "ordered_occurrence_dp_v1",
    "label_max_residual_m": 300,
    "label_skip_penalty": 0.15,
    "selection": ["cv_mean_trip_top1", "cv_macro_f1", "negative_cv_conditional_observation_loss", "negative_cv_top1_std", "negative_fit_seconds"],
}

def config(path=None):
    values = dict(DEFAULTS)
    if path:
        values.update(json.loads(Path(path).read_text()))
    if not values['radii_m'] or any(r <= 0 for r in values['radii_m']) or sorted(values['radii_m']) != values['radii_m']:
        raise ValueError('Candidate radii must be positive and sorted.')
    if values['max_candidates'] < 1 or values['transition_beta_m'] <= 0 or values['gap_seconds'] <= 0:
        raise ValueError('Candidate count, transition scale and gap threshold must be positive.')
    return values

def data_root(value=None):
    if value or os.getenv("PATHSNAP_DATA_ROOT"):
        return Path(value or os.environ["PATHSNAP_DATA_ROOT"]).expanduser().resolve()
    for p in (ROOT / "data", ROOT.parent, Path.cwd()):
        if (p / "train").is_dir() and (p / "student_test").is_dir():
            return p.resolve()
    raise ValueError("Set PATHSNAP_DATA_ROOT to the directory containing train/ and student_test/.")

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
