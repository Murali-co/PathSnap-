"""Replaceable provisional scorer/aligner. Never imported by feature extraction."""
import numpy as np
from .data import read_table
from .spatial import project_to_segments

def load_reference(path, arc_count):
    route = read_table(path, 1, integer=True).ravel()
    if (route < 0).any() or (route >= arc_count).any():
        row = int(np.flatnonzero((route < 0) | (route >= arc_count))[0])
        raise ValueError(f"{path}: invalid reference arc at row {row}")
    return route

def align(trip, net, route, cfg):
    """Monotone route-occurrence DP, with same-state progress and skip costs.

    Breaks at GPS gaps/invalid rows reset alignment. Reference discontinuities
    remain explicit diagnostics; monotonic traversal may cross them at a penalty.
    Repeated IDs retain distinct occurrence states. No pointwise organiser scorer
    was supplied: these labels are derived references, not official truth.
    """
    n, k = len(trip.track), len(route)
    labels = np.full(n, -1, dtype=np.int64)
    occurrences = np.full(n, -1, dtype=np.int64)
    residual = np.full(n, np.inf)
    ambiguous = np.zeros(n, dtype=bool)
    if not k:
        return dict(labels=labels, occurrences=occurrences, residual=residual, ambiguous=ambiguous)
    breaks = np.r_[False, net.arcs[route[:-1], 1] != net.arcs[route[1:], 0]]
    boundary_cost = np.cumsum(breaks)*4.0
    indices = np.arange(k)
    skip = cfg["label_skip_penalty"]
    blocks, current = [], []
    for i in range(n):
        if not trip.valid[i]:
            if current: blocks.append(current)
            current = []
            continue
        if current and trip.track[i, 2]-trip.track[current[-1], 2] > cfg["gap_seconds"]:
            blocks.append(current); current = []
        current.append(i)
    if current: blocks.append(current)
    for block in blocks:
        back = np.zeros((len(block), k), dtype=np.int32)
        costs = np.zeros(k)
        last_fraction = None
        distances = []
        for row, i in enumerate(block):
            d, fraction = project_to_segments(net.obs[i], net.starts[route], net.vectors[route])
            distances.append(d)
            emission = np.minimum((d/25)**2, 1e6)
            if row:
                # prefix minima implement all monotone skips in O(route length).
                adjusted = costs - indices*skip - boundary_cost
                prefix = np.minimum.accumulate(adjusted)
                arg = np.maximum.accumulate(np.where(adjusted <= prefix, indices, 0))
                shifted = np.r_[np.inf, prefix[:-1]] + (indices-1)*skip + boundary_cost
                previous = np.r_[0, arg[:-1]]
                backwards = np.maximum((last_fraction-fraction)*net.lengths[route]-cfg["backward_tolerance_m"], 0)/10
                stay = costs + backwards
                jump = shifted < stay
                back[row] = np.where(jump, previous, indices)
                costs = emission + np.minimum(stay, shifted)
            else:
                costs = emission
                back[row] = indices
            costs -= costs.min()
            last_fraction = fraction
        state = int(np.argmin(costs))
        for row in range(len(block)-1, -1, -1):
            i = block[row]
            occurrences[i] = state
            residual[i] = distances[row][state]
            other = distances[row][route != route[state]]
            ambiguous[i] = bool(len(other) and abs(float(other.min())-residual[i]) < 2)
            if residual[i] <= cfg["label_max_residual_m"]:
                labels[i] = route[state]
            state = int(back[row, state])
    return dict(labels=labels, occurrences=occurrences, residual=residual, ambiguous=ambiguous,
                reference_breaks=int(breaks.sum()))
