import numpy as np
from .spatial import Network

SCHEMA = ["distance_m", "distance_rank", "distance_delta_m", "fraction", "endpoint_distance_m", "length_m",
          "heading_cosine", "signed_motion_m", "heading_reliability", "previous_displacement_m", "next_displacement_m",
          "speed_mps", "previous_dt_s", "next_dt_s", "gap_flag", "stationary", "straightness_30s",
          "start_in_degree", "end_out_degree", "reverse_available", "candidate_count", "radius_m"]

def motion(trip, network, cfg):
    xy, t = network.obs, trip.track[:, 2]
    n = len(t)
    dt = np.r_[0, np.diff(t)]
    displacement = np.vstack((np.zeros(2), np.diff(xy, axis=0)))
    good = (dt > 0) & (dt <= cfg["gap_seconds"]) & np.isfinite(displacement).all(axis=1)
    displacement[~good] = 0
    lengths = np.linalg.norm(displacement, axis=1)
    next_d = np.r_[lengths[1:], 0]
    next_dt = np.r_[dt[1:], 0]
    heading = displacement.copy()
    # Full-trip offline features may use the next valid GPS observation, never future labels.
    for i in range(n-1):
        if good[i+1]:
            heading[i] += displacement[i+1]
    norm = np.linalg.norm(heading, axis=1)
    direction = heading / np.maximum(norm[:, None], 1e-12)
    straightness = np.zeros(n)
    prefix = np.r_[0, np.cumsum(lengths)]
    block_start = 0
    for i in range(n):
        if i and not good[i]:
            block_start = i
        left = max(block_start, int(np.searchsorted(t, t[i]-30)))
        path = prefix[i+1]-prefix[left+1]
        if path > 0:
            straightness[i] = np.linalg.norm(xy[i]-xy[left])/path
    reliability = np.minimum(norm/10, 1) * np.clip(straightness, 0, 1)
    return dict(dt=dt, next_dt=next_dt, displacement=lengths, next_d=next_d, direction=direction,
                norm=norm, reliability=reliability, straightness=np.clip(straightness, 0, 1), good=good)

def extract(trip, cfg, network=None):
    net = network or Network(trip)
    m = motion(trip, net, cfg)
    groups = [net.candidates(point, cfg) for point in net.obs]
    offsets = np.r_[0, np.cumsum([len(c.ids) for c in groups])].astype(np.int64)
    x = np.zeros((offsets[-1], len(SCHEMA)), dtype=np.float32)
    for i, c in enumerate(groups):
        ids = c.ids
        if not len(ids):
            continue
        unit = net.vectors[ids] / np.maximum(net.lengths[ids, None], 1e-12)
        dot = unit @ m["direction"][i]
        previous_dt = m["dt"][i]
        values = [c.distance, np.arange(len(ids)), c.distance-c.distance[0], c.fraction,
                  np.minimum(c.fraction, 1-c.fraction)*net.lengths[ids], net.lengths[ids],
                  dot*m["reliability"][i], dot*m["norm"][i], m["reliability"][i], m["displacement"][i], m["next_d"][i],
                  m["displacement"][i]/max(previous_dt, 1), previous_dt, m["next_dt"][i],
                  float(not m["good"][i] or m["next_dt"][i]>cfg["gap_seconds"]), float(m["displacement"][i]<2),
                  m["straightness"][i], net.in_degree[net.arcs[ids, 0]], net.out_degree[net.arcs[ids, 1]],
                  net.reverse[ids], len(ids), c.radius]
        x[offsets[i]:offsets[i+1]] = np.column_stack([np.broadcast_to(v, len(ids)) for v in values])
    return net, groups, offsets, np.nan_to_num(x, nan=0, posinf=1e6, neginf=-1e6)
