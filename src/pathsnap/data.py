"""Input loading deliberately never discovers or opens reference files."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np

@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    valid: np.ndarray

def read_table(path, columns, integer=False, allow_invalid=False):
    path = Path(path)
    try:
        a = np.loadtxt(path, ndmin=2)
    except Exception as exc:
        raise ValueError(f"{path.parent.name}/{path.name}: numeric/schema parsing failed: {exc}") from exc
    if a.shape[1] != columns or len(a) == 0:
        raise ValueError(f"{path.parent.name}/{path.name}: expected {columns} columns and nonempty rows")
    good = np.isfinite(a).all(axis=1)
    if integer:
        good &= (a == np.floor(a)).all(axis=1)
    if not allow_invalid and not good.all():
        row = int(np.flatnonzero(~good)[0])
        raise ValueError(f"{path.parent.name}/{path.name}: invalid value at zero-based row {row}")
    return a.astype(np.int64) if integer else a

def load_trip(track_path, nodes_path, arcs_path):
    track = read_table(track_path, 3, allow_invalid=True)
    nodes = read_table(nodes_path, 2)
    arcs = read_table(arcs_path, 2, integer=True)
    coord_ok = lambda a: (np.abs(a[:, 0]) <= 180) & (np.abs(a[:, 1]) <= 90)
    if not coord_ok(nodes).all():
        raise ValueError(f"{Path(nodes_path).name}: coordinate range at row {int(np.flatnonzero(~coord_ok(nodes))[0])}")
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        bad = np.flatnonzero(((arcs < 0) | (arcs >= len(nodes))).any(axis=1))[0]
        raise ValueError(f"{Path(arcs_path).name}: node index out of range at row {bad}")
    finite_time = np.isfinite(track[:, 2])
    if np.any(np.diff(track[finite_time, 2]) <= 0):
        raise ValueError(f"{Path(track_path).name}: timestamps must be strictly increasing")
    valid = np.isfinite(track).all(axis=1) & coord_ok(track)
    return Trip(Path(track_path).stem, track, nodes, arcs, valid)

def input_paths(directory, trip_id=None):
    directory = Path(directory)
    name = trip_id or directory.name
    return tuple(directory / f"{name}.{ext}" for ext in ("track", "nodes", "arcs"))

def list_trips(root, partition):
    return sorted(p for p in (Path(root) / partition).iterdir() if p.is_dir())

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
