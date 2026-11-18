#!/usr/bin/env python3
"""Final-only scoring of the frozen selected artifact. Never selects among models."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pathsnap.inference import load_artifact,predict_trip
from pathsnap.data import input_paths,load_trip
from pathsnap.evaluation.scorer import score_prediction
from pathsnap.config import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--artifacts',default='artifacts');p.add_argument('--output',default='private/evaluation_results');a=p.parse_args()
    model,registry,cfg=load_artifact(ROOT/a.artifacts)
    if not registry.get('sha256'):raise SystemExit('Export/checksum the selected model before final evaluation.')
    dirs=sorted(p for p in Path(a.data_root).iterdir() if p.is_dir())
    if len(dirs)!=10:raise SystemExit('Exactly ten organiser-supplied evaluation trip directories are required.')
    results=[]
    for directory in dirs:
        tid=directory.name;trip=load_trip(*input_paths(directory));prediction,context=predict_trip(trip,model,cfg)
        reference=directory/f'{tid}.route'
        score=score_prediction(context,reference,cfg) if reference.is_file() else dict(evaluation_status='reference_unavailable',metrics=None)
        score.get('metrics',{}).pop('label_ambiguity_fraction',None) if score.get('metrics') else None
        results.append(dict(trip_id=tid,model_version=registry['model_version'],**score,timing=prediction['timing']))
    write_json(ROOT/a.output/'aggregate_results.json',results)
    print('Final selected-model evaluation written server-side. Do not use it to tune or select a different model.')
if __name__=='__main__':main()
