import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .config import digest, write_json
from .data import file_hash, input_paths, list_trips, load_trip
from .features import SCHEMA, extract
from .reference import align, load_reference
from .spatial import Network

def audit(root, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    inventory, manifest = [], []
    for partition in ("train", "student_test"):
        for directory in list_trips(root, partition):
            paths = input_paths(directory)
            trip = load_trip(*paths)
            net = Network(trip)
            route_path = directory / f"{directory.name}.route"
            route = load_reference(route_path, len(trip.arcs))
            dt = np.diff(trip.track[:, 2])
            unique = np.unique(trip.arcs, axis=0)
            # Connectivity uses boolean adjacency, so duplicate weights cannot inflate distances.
            adj = csr_matrix((np.ones(len(unique)), (unique[:, 0], unique[:, 1])), shape=(len(trip.nodes), len(trip.nodes)))
            weak = connected_components(adj, directed=True, connection="weak", return_labels=False)
            strong = connected_components(adj, directed=True, connection="strong", return_labels=False)
            sample = np.linspace(0, len(trip.arcs)-1, min(100, len(trip.arcs)), dtype=int)
            ends = trip.nodes[trip.arcs[sample]]
            _, _, gd = Geod(ellps="WGS84").inv(ends[:, 0, 0], ends[:, 0, 1], ends[:, 1, 0], ends[:, 1, 1])
            rel = np.abs(net.lengths[sample]-gd)/np.maximum(gd, 1)
            row = dict(partition=partition, trip_id=trip.trip_id, observations=len(trip.track), nodes=len(trip.nodes), arcs=len(trip.arcs),
                       reference_entries=len(route), invalid_observations=int((~trip.valid).sum()), one_second_intervals=int((dt==1).sum()),
                       intervals=len(dt), max_gap_s=float(dt.max(initial=0)), gaps_over_60s=int((dt>60).sum()),
                       zero_length_arcs=int((net.lengths<1e-8).sum()), duplicate_endpoint_pairs=len(trip.arcs)-len(unique),
                       reference_breaks=int((trip.arcs[route[:-1], 1] != trip.arcs[route[1:], 0]).sum()),
                       weak_components=int(weak), strong_components=int(strong), isolated_nodes=int(((net.in_degree+net.out_degree)==0).sum()),
                       lon_min=float(trip.track[trip.valid, 0].min()), lon_max=float(trip.track[trip.valid, 0].max()),
                       lat_min=float(trip.track[trip.valid, 1].min()), lat_max=float(trip.track[trip.valid, 1].max()),
                       sampled_projection_max_relative_error=float(rel.max(initial=0)))
            inventory.append(row)
            for p in (*paths, route_path):
                manifest.append(dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=file_hash(p)))
            print(f"audited {partition}/{trip.trip_id}", flush=True)
    pd.DataFrame(inventory).to_csv(output/"inventory.csv", index=False)
    write_json(output/"data_manifest.json", manifest)
    summary = {}
    for partition in ("train", "student_test"):
        rows = [r for r in inventory if r["partition"]==partition]
        summary[partition] = {"trips":len(rows), **{key:sum(r[key] for r in rows) for key in ["observations","nodes","arcs","reference_entries","invalid_observations","zero_length_arcs","duplicate_endpoint_pairs","reference_breaks"]}}
    write_json(output/"audit_summary.json", summary)
    return summary

def prepare(root, output, cfg, resume=True, limit=None):
    output = Path(output); cache = output/"cache"; cache.mkdir(parents=True, exist_ok=True)
    entries, diagnostics = [], []
    for partition in ("train", "student_test"):
        directories = list_trips(root, partition)
        if limit: directories = directories[:limit]
        for directory in directories:
            trip_id = directory.name
            paths = input_paths(directory)
            ref_path = directory/f"{trip_id}.route"
            identity = digest(dict(files=[file_hash(p) for p in (*paths,ref_path)], cfg=cfg, schema=SCHEMA, version=1))
            target = cache/f"{partition}_{trip_id}_{identity[:16]}.npz"
            start = perf_counter()
            if not (resume and target.exists()):
                trip = load_trip(*paths)
                net, groups, offsets, x = extract(trip, cfg)
                route = load_reference(ref_path, len(trip.arcs))
                labels = align(trip, net, route, cfg)
                ids = np.concatenate([c.ids for c in groups])
                obs = np.repeat(np.arange(len(trip.track)), np.diff(offsets))
                y = (ids == labels["labels"][obs]).astype(np.int8)
                # Unknown labels excluded at fitting time, never silently removed from coverage.
                np.savez_compressed(target, x=x, y=y, ids=ids, obs=obs, offsets=offsets, labels=labels["labels"],
                                    occurrences=labels["occurrences"], residual=labels["residual"], ambiguous=labels["ambiguous"],
                                    fractions=np.concatenate([c.fraction for c in groups]), schema=np.array(SCHEMA), identity=identity)
            with np.load(target) as a:
                known = a["labels"]>=0
                hits = np.bincount(a["obs"], weights=a["y"], minlength=len(known))>0
                diagnostics.append(dict(partition=partition, trip_id=trip_id, observations=len(known), label_coverage=float(known.mean()),
                                        candidate_recall=float(hits[known].mean()) if known.any() else None,
                                        ambiguous_fraction=float(a["ambiguous"].mean()), candidate_rows=len(a["y"]), prepare_seconds=perf_counter()-start))
            entries.append(dict(partition=partition,trip_id=trip_id,file=target.relative_to(output).as_posix(),identity=identity))
            print(f"prepared {partition}/{trip_id}", flush=True)
    write_json(output/"feature_manifest.json", dict(config=cfg, schema=SCHEMA, entries=entries, smoke_subset=bool(limit)))
    pd.DataFrame(diagnostics).to_csv(output/"candidate_diagnostics.csv",index=False)

def candidate_sweep(root, output, cfg, trip_limit=8):
    """Training-only recall/runtime sweep; never inserts reference candidates."""
    rows=[]
    for directory in list_trips(root,"train")[:trip_limit]:
        trip=load_trip(*input_paths(directory)); net=Network(trip)
        reference=align(trip,net,load_reference(directory/f"{directory.name}.route",len(trip.arcs)),cfg)["labels"]
        for radius,k in [(50,12),(100,16),(200,20),(300,20)]:
            current={**cfg,"radii_m":[radius],"max_candidates":k}
            start=perf_counter(); groups=[net.candidates(p,current) for p in net.obs]
            known=reference>=0
            hits=np.array([reference[i] in c.ids for i,c in enumerate(groups)])
            rows.append(dict(trip_id=trip.trip_id,radius_m=radius,k=k,known=int(known.sum()),hits=int(hits[known].sum()),seconds=perf_counter()-start))
    pd.DataFrame(rows).to_csv(Path(output)/"candidate_sweep.csv",index=False)
