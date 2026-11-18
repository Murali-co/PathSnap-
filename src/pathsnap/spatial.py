from collections import OrderedDict
from dataclasses import dataclass
import heapq
import numpy as np
from pyproj import CRS, Transformer
from shapely import STRtree, box, linestrings

@dataclass
class Candidates:
    ids: np.ndarray
    distance: np.ndarray
    fraction: np.ndarray
    radius: float

def project_to_segments(point, starts, vectors):
    length2 = np.einsum("ij,ij->i", vectors, vectors)
    fraction = np.clip(np.sum((point - starts) * vectors, axis=1) / np.maximum(length2, 1e-15), 0, 1)
    foot = starts + fraction[:, None] * vectors
    return np.linalg.norm(point - foot, axis=1), fraction

class Network:
    def __init__(self, trip):
        center = np.mean(trip.nodes, axis=0)
        local = CRS.from_proj4(f"+proj=aeqd +lat_0={center[1]} +lon_0={center[0]} +datum=WGS84 +units=m")
        self.forward = Transformer.from_crs(4326, local, always_xy=True)
        self.inverse = Transformer.from_crs(local, 4326, always_xy=True)
        self.xy = np.column_stack(self.forward.transform(trip.nodes[:, 0], trip.nodes[:, 1]))
        self.obs = np.full((len(trip.track), 2), np.nan)
        self.obs[trip.valid] = np.column_stack(self.forward.transform(trip.track[trip.valid, 0], trip.track[trip.valid, 1]))
        self.arcs, self.nodes = trip.arcs, trip.nodes
        self.starts = self.xy[self.arcs[:, 0]]
        self.vectors = self.xy[self.arcs[:, 1]] - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.tree = STRtree(linestrings(np.stack((self.starts, self.starts + self.vectors), axis=1)))
        self.out_degree = np.bincount(self.arcs[:, 0], minlength=len(self.xy))
        self.in_degree = np.bincount(self.arcs[:, 1], minlength=len(self.xy))
        pairs = set(map(tuple, self.arcs.tolist()))
        self.reverse = np.array([(int(v), int(u)) in pairs for u, v in self.arcs], dtype=np.float32)
        # Edge-aware adjacency: duplicate arcs are never summed or renumbered.
        self.adj = [[] for _ in range(len(self.xy))]
        for arc_id, (u, v) in enumerate(self.arcs):
            self.adj[u].append((int(v), float(self.lengths[arc_id]), arc_id))
        self.cache = OrderedDict()

    def candidates(self, point, cfg):
        if not np.isfinite(point).all():
            return Candidates(np.array([], dtype=np.int64), np.array([]), np.array([]), 0)
        for radius in cfg["radii_m"]:
            ids = self.tree.query(box(point[0]-radius, point[1]-radius, point[0]+radius, point[1]+radius))
            distances, fraction = project_to_segments(point, self.starts[ids], self.vectors[ids])
            use = distances <= radius
            ids, distances, fraction = ids[use], distances[use], fraction[use]
            if len(ids):
                order = np.lexsort((ids, distances))[:cfg["max_candidates"]]
                return Candidates(ids[order], distances[order], fraction[order], radius)
        return Candidates(np.array([], dtype=np.int64), np.array([]), np.array([]), float(cfg["radii_m"][-1]))

    def shortest(self, source, target, cutoff):
        source, target = int(source), int(target)
        if cutoff < 0 or not np.isfinite(cutoff):
            return float('inf'), []
        if source == target:
            return 0.0, []
        # Source searches reuse a rounded-up bound. Always enforce the exact caller bound.
        # The cache belongs to this network instance; failure at one bound never poisons another.
        bucket = float(np.ceil(cutoff / 50) * 50)
        key = (source, bucket)
        if key in self.cache:
            self.cache.move_to_end(key)
            best, prev = self.cache[key]
        else:
            best, prev = {source: 0.0}, {}
            queue = [(0.0, source)]
            while queue:
                distance, node = heapq.heappop(queue)
                if distance != best[node]:
                    continue
                for other, weight, arc in self.adj[node]:
                    nd = distance + weight
                    if nd <= bucket and nd < best.get(other, float('inf')):
                        best[other], prev[other] = nd, (node, arc)
                        heapq.heappush(queue, (nd, other))
            self.cache[key] = (best, prev)
            if len(self.cache) > 2000:
                self.cache.popitem(last=False)
        distance = best.get(target, float('inf'))
        if distance > cutoff:
            return float('inf'), []
        path, node = [], target
        while node != source:
            parent, arc = prev[node]
            path.append(arc)
            node = parent
        return distance, path[::-1]

    def transition(self, a, fa, b, fb, dt, displacement, cfg):
        if dt <= 0 or dt > cfg["gap_seconds"]:
            return float("inf"), []
        limit = cfg["max_speed_mps"] * dt + cfg["movement_slack_m"]
        if a == b:
            delta = (fb - fa) * self.lengths[a]
            if delta >= -cfg["backward_tolerance_m"]:
                return (max(0.0, delta), []) if max(0.0, delta) <= limit else (float("inf"), [])
        remainder = (1-fa)*self.lengths[a] + fb*self.lengths[b]
        if remainder > limit:
            return float("inf"), []
        d, middle = self.shortest(self.arcs[a, 1], self.arcs[b, 0], limit-remainder)
        return remainder + d, middle

    def arc_coords(self, arc_id):
        return self.nodes[self.arcs[arc_id]].tolist()
