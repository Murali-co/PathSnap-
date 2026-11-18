"""Prediction opens only explicitly supplied track/nodes/arcs. No reference imports."""
from pathlib import Path
from time import perf_counter
import json
import joblib
import numpy as np
from .config import config, digest
from .data import load_trip, input_paths, file_hash
from .features import SCHEMA, extract
from .decoding.viterbi import decode_observation_groups, reconstruct
from .evaluation.metrics import top1_states, continuity


def load_artifact(directory):
    directory=Path(directory).resolve()
    registry=json.loads((directory/'model_registry.json').read_text())
    filename=registry.get('selected_model_file','selected_model.joblib')
    model_path=(directory/filename).resolve()
    if model_path.parent != directory:
        raise ValueError('Model artifact must be inside the artifact directory.')
    if registry.get('feature_schema', SCHEMA) != SCHEMA:
        raise ValueError('Artifact feature order differs from this implementation; retrain/export it.')
    if registry.get('sha256') and file_hash(model_path) != registry['sha256']:
        raise ValueError('Model artifact checksum failed.')
    try:
        model=joblib.load(model_path)
    except RuntimeError as exc:
        if 'CUDA' not in str(exc) and 'cuda' not in str(exc):raise
        from .training import load_saved_model
        model=load_saved_model(model_path)
    if hasattr(model,'get_booster'):
        model.set_params(device='cpu')
    if hasattr(model,'to_cpu'):
        model.to_cpu()
    if getattr(model,'n_features_in_',len(SCHEMA)) != len(SCHEMA):
        raise ValueError('Model feature count differs from the feature schema.')
    return model, registry, config_from_registry(registry)


def config_from_registry(registry):
    cfg=config()
    cfg.update(registry.get('prediction_config',{}))
    return cfg


def route_geometry(net, components):
    lines=[]
    for arcs in components:
        line=[]
        for arc in arcs:
            pair=net.arc_coords(arc)
            if line and line[-1] != pair[0]:
                if len(line)>1:lines.append(line)
                line=[]
            if not line:line.append(pair[0])
            line.append(pair[1])
        if len(line)>1:lines.append(line)
    return lines


def predict_trip(trip, model, cfg, progress=None):
    mark=progress or (lambda stage:None)
    start=perf_counter();mark('candidates_and_features')
    net,groups,offsets,x=extract(trip,cfg)
    feature_seconds=perf_counter()-start
    mark('classifier');t=perf_counter()
    probabilities=model.predict_proba(x)[:,1] if len(x) else np.empty(0)
    if not np.isfinite(probabilities).all():raise ValueError('Model returned nonfinite probabilities.')
    ids=np.concatenate([c.ids for c in groups])
    raw=top1_states(ids,probabilities,offsets)
    classifier_seconds=perf_counter()-t
    mark('viterbi');t=perf_counter()
    decoded=decode_observation_groups(net,groups,probabilities,offsets,trip.track,cfg)
    decode_seconds=perf_counter()-t
    mark('route_reconstruction');t=perf_counter()
    components,interpolated=reconstruct(net,groups,decoded,trip.track,cfg)
    raw_blocks=np.zeros(len(raw),dtype=int)
    for i in range(1,len(raw)):
        raw_blocks[i]=raw_blocks[i-1]+int(raw[i-1]<0 or raw[i]<0 or not trip.valid[i-1] or not trip.valid[i] or trip.track[i,2]-trip.track[i-1,2]>cfg['gap_seconds'])
    raw_components,_=reconstruct(net,groups,dict(states=raw,block=raw_blocks),trip.track,cfg)
    records=[]
    for i,arc in enumerate(decoded['states']):
        point=None
        # Match coordinates are projected onto the selected segment, preserving IDs.
        if arc>=0:
            c=groups[i];fraction=c.fraction[np.flatnonzero(c.ids==arc)[0]]
            xy=net.starts[arc]+fraction*net.vectors[arc]
            point=list(net.inverse.transform(*xy))
        records.append(dict(observation_id=i,arc_id=int(arc) if arc>=0 else None,status=str(decoded['status'][i]),
                            block_id=int(decoded['block'][i]) if arc>=0 else None,matched_coordinate=point))
    relevant=np.unique(ids)
    # Public road context is bounded independently of references.
    road_limit=5000
    geometry=dict(gps=[r[:2].tolist() if v else None for r,v in zip(trip.track,trip.valid)],
                  gps_segments=gps_segments(trip,cfg),predicted=route_geometry(net,components),
                  classifier=route_geometry(net,raw_components),
                  network=[net.arc_coords(int(a)) for a in relevant[:road_limit]],
                  network_omitted_arcs=max(0,len(relevant)-road_limit),predicted_arc_ids=[r['arc_id'] for r in records])
    diagnostics=dict(unmatched_observations=int((decoded['states']<0).sum()),blocks=decoded['blocks'],
                     restarts=len(decoded['restarts']),interpolated_arcs=interpolated,
                     classifier=continuity(net,groups,raw,trip.track,cfg),decoded=continuity(net,groups,decoded['states'],trip.track,cfg))
    timings=dict(features_seconds=feature_seconds,classifier_seconds=classifier_seconds,decoder_seconds=decode_seconds,
                 reconstruction_seconds=perf_counter()-t,prediction_seconds=perf_counter()-start)
    result=dict(geometry=geometry,matches=records,route_components=components,observations=len(trip.track),
                diagnostics=diagnostics,timing=timings,predictions_available=True)
    # Internal arrays are returned only to the server-side caller, never serialized by the API.
    context=dict(trip=trip,net=net,groups=groups,offsets=offsets,ids=ids,probabilities=probabilities,
                 raw=raw,decoded=decoded,components=components,raw_components=raw_components)
    return result,context


def gps_segments(trip,cfg):
    result,current=[],[]
    for i,row in enumerate(trip.track):
        gap=i and (not trip.valid[i-1] or row[2]-trip.track[i-1,2]>cfg['gap_seconds'])
        if not trip.valid[i] or gap:
            if current:result.append(current)
            current=[]
        if trip.valid[i]:current.append(row[:2].tolist())
    if current:result.append(current)
    return result
