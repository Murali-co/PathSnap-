"""Compare saved fits, select by training OOF and export a checksummed artifact."""
from pathlib import Path
from time import perf_counter
import json
import shutil
import numpy as np
import pandas as pd
from .config import digest,write_json
from .data import file_hash
from .features import SCHEMA
from .training import MODELS,load_prepared,load_saved_model,training_arrays,oof_trip_metrics,selection_key
from .evaluation.metrics import candidate_metrics,observation_top1


def evaluate_cached(model,rows):
    records=[];ys=[];ps=[];total_time=0
    for partition,tid,a in rows:
        if partition!='student_test':continue
        t=perf_counter();p=model.predict_proba(a['x'])[:,1] if len(a['x']) else np.empty(0);elapsed=perf_counter()-t
        if not np.isfinite(p).all():raise ValueError('Nonfinite model probabilities.')
        known=np.repeat(a['labels']>=0,np.diff(a['offsets']))
        ys.append(a['y'][known]);ps.append(p[known]);total_time+=elapsed
        row=dict(trip_id=tid,prediction_seconds=elapsed,**observation_top1(a['ids'],p,a['offsets'],a['labels']))
        row.update(candidate_metrics(a['y'][known],p[known]>=.5,p[known]));records.append(row)
    if not records:raise ValueError('No student-test trips available.')
    y,p=np.concatenate(ys),np.concatenate(ps)
    metrics=candidate_metrics(y,p>=.5,p)
    known=sum(r['known_observations'] for r in records)
    return dict(student_trips=len(records),candidate_rows=len(y),**metrics,
                mean_trip_observation_top1=float(np.mean([r['observation_top1'] for r in records if r['observation_top1'] is not None])),
                pooled_observation_top1=sum(r['correct_observations'] for r in records)/known if known else None,
                candidate_miss_rate=sum(r['candidate_miss_count'] for r in records)/known if known else None,
                label_coverage=known/sum(r['observations'] for r in records),prediction_seconds=total_time),records


def cv_summary(model_dir,train,y):
    model_dir=Path(model_dir)
    if not (model_dir/'oof_probability.npy').exists():raise ValueError(f'{model_dir.name}: missing OOF predictions; run grouped training first.')
    p=np.load(model_dir/'oof_probability.npy')
    if len(p)!=len(y) or not np.isfinite(p).all():raise ValueError('OOF row count/probability mismatch; retrain with current features.')
    rows=oof_trip_metrics(train,p)
    losses=[r['conditional_observation_loss'] for r in rows if r['conditional_observation_loss'] is not None]
    result=dict(cv_mean_trip_top1=float(np.mean([r['observation_top1'] for r in rows])),
                cv_macro_f1=candidate_metrics(y,p>=.5,p)['candidate_macro_f1'],
                cv_conditional_observation_loss=float(np.mean(losses)) if losses else 1e9,
                cv_top1_std=float(np.std([r['observation_top1'] for r in rows])))
    full=pd.read_csv(model_dir/'full_fit.csv');result['fit_seconds']=float(full['fit_seconds'].iloc[0])
    return result,rows


def compare(prepared,models,output,require_five=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest,rows=load_prepared(prepared);X,y,g,train=training_arrays(rows);del X,g
    results=[]
    for name in MODELS:
        directory=Path(models)/name
        if not (directory/'selected_model.joblib').exists():
            if require_five:raise ValueError('Missing trained family '+name)
            continue
        model=load_saved_model(directory/'selected_model.joblib')
        if hasattr(model,'get_booster'):model.set_params(device='cpu')
        if hasattr(model,'to_cpu'):model.to_cpu()
        summary,per_trip=evaluate_cached(model,rows)
        cv,cv_rows=cv_summary(directory,train,y)
        pd.DataFrame(per_trip).to_csv(output/f'{name}_per_trip.csv',index=False)
        pd.DataFrame(cv_rows).to_csv(output/f'{name}_cv_per_trip.csv',index=False)
        result=dict(model=name,**cv,**summary);results.append(result)
        print(f'compared {name}: CV mean-trip Top-1={cv["cv_mean_trip_top1"]:.6f}',flush=True)
    if not results:raise ValueError('No models to compare.')
    results.sort(key=selection_key,reverse=True)
    pd.DataFrame(results).to_csv(output/'model_comparison.csv',index=False)
    write_json(output/'selection.json',dict(model=results[0]['model'],rule=manifest['config'].get('selection',[]),
               basis='training_out_of_fold_predictions',ranked_models=[r['model'] for r in results],
               prepared_identity=digest(manifest),student_test_is_development=True))
    return results


def export_artifact(source,artifact_dir,prepared,model_name,provenance=None):
    artifact_dir=Path(artifact_dir);artifact_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((Path(prepared)/'feature_manifest.json').read_text())
    # Re-serialize CPU portable weights so inference does not require CUDA.
    model=load_saved_model(source)
    if hasattr(model,'get_booster'):model.set_params(device='cpu')
    if hasattr(model,'to_cpu'):model.to_cpu()
    import joblib
    target=artifact_dir/'selected_model.joblib';temporary=artifact_dir/'selected_model.tmp'
    joblib.dump(model,temporary);temporary.replace(target)
    checksum=file_hash(target)
    registry=dict(model_family=model_name,model_version=f'{model_name}-{checksum[:12]}',selected_model_file=target.name,
                  sha256=checksum,feature_schema=SCHEMA,prediction_config=manifest['config'],prepared_identity=digest(manifest),
                  training_partition='train_80',development_partition='student_test_10',provenance=provenance or {})
    write_json(artifact_dir/'model_registry.json',registry)
    (artifact_dir.parent/'requirements-selected.txt').write_text('torch>=2.5,<3\n' if model_name=='pytorch_mlp' else '# Selected model does not require PyTorch for inference.\n')
    return registry
