"""Log-space Viterbi with explicit, independently backtracked route blocks."""
import numpy as np


def emissions(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(np.nan_to_num(p, nan=0.5, posinf=1, neginf=0), 1e-7, 1-1e-7)
    return np.log(p / p.sum()) if len(p) else np.empty(0)


def decode_observation_groups(net, groups, probabilities, offsets, trip_track, cfg):
    n = len(trip_track)
    states = np.full(n, -1, dtype=np.int64)
    status = np.full(n, 'unmatched', dtype='U24')
    blocks = np.full(n, -1, dtype=np.int64)
    path_rows, backs, score = [], [], None
    block_id, restarts = 0, []

    def finish():
        nonlocal path_rows, backs, score, block_id
        if path_rows:
            state = int(np.argmax(score))
            for j in range(len(path_rows)-1, -1, -1):
                obs = path_rows[j]
                states[obs] = groups[obs].ids[state]
                status[obs] = 'decoded'
                blocks[obs] = block_id
                if j:
                    state = int(backs[j][state])
            block_id += 1
        path_rows, backs, score = [], [], None

    for i, group in enumerate(groups):
        if not np.isfinite(net.obs[i]).all() or not np.isfinite(trip_track[i, 2]):
            finish()
            status[i] = 'invalid_observation'
            continue
        if not len(group.ids):
            finish()
            restarts.append(dict(observation=i, reason='no_candidates'))
            continue
        emission = emissions(probabilities[offsets[i]:offsets[i+1]])
        if path_rows:
            previous = path_rows[-1]
            dt = float(trip_track[i, 2]-trip_track[previous, 2])
            if dt <= 0 or dt > cfg['gap_seconds']:
                finish()
                restarts.append(dict(observation=i, reason='time_gap'))
        if not path_rows:
            path_rows, backs, score = [i], [None], emission
            continue
        previous = path_rows[-1]
        dt = float(trip_track[i, 2]-trip_track[previous, 2])
        displacement = float(np.linalg.norm(net.obs[i]-net.obs[previous]))
        cur = np.full(len(group.ids), -np.inf)
        back = np.full(len(group.ids), -1, dtype=np.int32)
        for b, bid in enumerate(group.ids):
            for a, aid in enumerate(groups[previous].ids):
                if not np.isfinite(score[a]):
                    continue
                travelled, _ = net.transition(int(aid), groups[previous].fraction[a], int(bid), group.fraction[b], dt, displacement, cfg)
                if not np.isfinite(travelled):
                    continue
                penalty = abs(travelled-displacement)/cfg['transition_beta_m']
                if aid != bid and np.array_equal(net.arcs[aid], net.arcs[bid][::-1]):
                    penalty += cfg.get('uturn_penalty', 1)
                value = score[a] - penalty + emission[b]
                if value > cur[b]:
                    cur[b], back[b] = value, a
        if not np.isfinite(cur).any():
            finish()
            restarts.append(dict(observation=i, reason='no_feasible_transition'))
            path_rows, backs, score = [i], [None], emission
        else:
            cur -= np.max(cur)
            path_rows.append(i)
            backs.append(back)
            score = cur
    finish()
    return dict(states=states, status=status, block=blocks, restarts=restarts, blocks=block_id)


def reconstruct(net, groups, decoded, track, cfg):
    """Return components of actual original arcs; never bridge declared breaks."""
    components, current, interpolation = [], [], 0
    previous = None
    for i, arc in enumerate(decoded['states']):
        if arc < 0:
            if current: components.append(current)
            current, previous = [], None
            continue
        if previous is None or decoded['block'][i] != decoded['block'][previous]:
            if current: components.append(current)
            current = [int(arc)]
        else:
            a = int(decoded['states'][previous])
            fa = float(groups[previous].fraction[np.flatnonzero(groups[previous].ids == a)[0]])
            fb = float(groups[i].fraction[np.flatnonzero(groups[i].ids == arc)[0]])
            distance, middle = net.transition(a, fa, int(arc), fb, track[i, 2]-track[previous, 2],
                                              np.linalg.norm(net.obs[i]-net.obs[previous]), cfg)
            if not np.isfinite(distance):
                if current: components.append(current)
                current = [int(arc)]
            elif a != arc or middle:
                current.extend(middle)
                current.append(int(arc))
                interpolation += len(middle)
        previous = i
    if current: components.append(current)
    return components, interpolation
