#!/usr/bin/env python3
"""Resumable TPE for whichever family wins training OOF. Never fits student-test."""
import argparse,json,sys,time
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pathsnap.config import digest,write_json
from pathsnap.training import MODELS,load_prepared,training_arrays,build_model,oof_trip_metrics,selection_key
from pathsnap.experiments import cv_summary,evaluate_cached,export_artifact
from pathsnap.evaluation.metrics import candidate_metrics


def suggest(trial,name):
    if name=='logistic':return dict(C=trial.suggest_float('C',1e-3,100,log=True),solver=trial.suggest_categorical('solver',['lbfgs','liblinear']))
    if name=='decision_tree':return dict(max_depth=trial.suggest_int('max_depth',4,20),min_samples_leaf=trial.suggest_int('min_samples_leaf',5,100),criterion=trial.suggest_categorical('criterion',['gini','entropy']))
    if name=='random_forest':return dict(n_estimators=trial.suggest_int('n_estimators',60,240,step=30),max_depth=trial.suggest_int('max_depth',6,20),min_samples_leaf=trial.suggest_int('min_samples_leaf',3,40),max_features=trial.suggest_float('max_features',.4,1.0))
    if name=='xgboost':return dict(n_estimators=trial.suggest_int('n_estimators',100,500),max_depth=trial.suggest_int('max_depth',3,10),learning_rate=trial.suggest_float('learning_rate',.01,.2,log=True),min_child_weight=trial.suggest_float('min_child_weight',.5,10,log=True),subsample=trial.suggest_float('subsample',.6,1),colsample_bytree=trial.suggest_float('colsample_bytree',.6,1),reg_alpha=trial.suggest_float('reg_alpha',1e-8,10,log=True),reg_lambda=trial.suggest_float('reg_lambda',1e-5,100,log=True))
    return dict(hidden1=trial.suggest_categorical('hidden1',[32,64,128]),hidden2=trial.suggest_categorical('hidden2',[32,64]),lr=trial.suggest_float('lr',1e-4,1e-2,log=True),weight_decay=trial.suggest_float('weight_decay',1e-6,1e-2,log=True),dropout=trial.suggest_float('dropout',0,.3),batch_size=trial.suggest_categorical('batch_size',[1024,2048,4096]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',default='outputs/prepared');p.add_argument('--models',default='outputs/experiments/models')
    p.add_argument('--selection',default='outputs/experiments/student_test/selection.json');p.add_argument('--output',default='outputs/experiments/tuning_v2')
    p.add_argument('--trials',type=int,default=15);p.add_argument('--startup',type=int,default=5);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--artifacts',default='artifacts');a=p.parse_args()
    if a.trials<=a.startup:raise SystemExit('Total trials must exceed startup trials to demonstrate TPE.')
    selection=json.loads((ROOT/a.selection).read_text());name=selection['model'];out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    manifest,rows=load_prepared(ROOT/a.prepared);X,y,groups,train=training_arrays(rows)
    if len(train)!=80 or manifest.get('smoke_subset'):raise SystemExit('Tuning requires all 80 training trips.')
    seed=manifest['config'].get('seed',42);folds=manifest['config'].get('folds',3)
    identity=digest(dict(manifest=manifest,model=name,folds=folds,startup=a.startup,objective='mean_trip_observation_top1_v2'))
    study=optuna.create_study(storage='sqlite:///'+str((out/'tuning.sqlite3').resolve()),study_name='pathsnap_'+name+'_'+identity[:12],direction='maximize',load_if_exists=True,sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=a.startup))
    write_json(out/'search_config.json',dict(model=name,identity=identity,objective='grouped_cv_mean_trip_observation_top1',startup_trials=a.startup,total_target=a.trials,seed=seed,folds=folds))
    def save(study,trial):
        study.trials_dataframe().to_csv(out/'trials.csv',index=False)
        write_json(out/'search_space.json',{k:optuna.distributions.distribution_to_json(v) for k,v in trial.distributions.items()})
    def objective(trial):
        params=suggest(trial,name);prob=np.full(len(y),np.nan);t=time.perf_counter()
        for fold,(tr,va) in enumerate(GroupKFold(folds).split(X,y,groups)):
            model=build_model(name,seed,a.device,params);model.fit(X[tr],y[tr]);prob[va]=model.predict_proba(X[va])[:,1]
        per_trip=oof_trip_metrics(train,prob);scores=[r['observation_top1'] for r in per_trip]
        trial.set_user_attr('cv_macro_f1',candidate_metrics(y,prob>=.5,prob)['candidate_macro_f1'])
        trial.set_user_attr('fit_seconds',time.perf_counter()-t)
        return float(np.mean(scores))
    started=time.perf_counter();finished=[t for t in study.trials if t.state.is_finished()]
    remaining=max(0,a.trials-len(finished))
    study.optimize(objective,n_trials=remaining,callbacks=[save],catch=(RuntimeError,ValueError))
    completed=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    if len(completed)<=a.startup:raise SystemExit('Study has insufficient successful trials beyond startup. Inspect trials.csv and retry.')
    write_json(out/'best_params.json',study.best_params)
    # Recompute best OOF for all tie-breakers and save independently inspectable predictions.
    best_oof=np.empty(len(y))
    for tr,va in GroupKFold(folds).split(X,y,groups):
        m=build_model(name,seed,a.device,study.best_params);m.fit(X[tr],y[tr]);best_oof[va]=m.predict_proba(X[va])[:,1]
    np.save(out/'oof_probability.npy',best_oof)
    best=build_model(name,seed,a.device,study.best_params);t=time.perf_counter();best.fit(X,y);fit=time.perf_counter()-t
    joblib.dump(best,out/'selected_model.joblib');pd.DataFrame([dict(fit_seconds=fit)]).to_csv(out/'full_fit.csv',index=False)
    tuned,_=cv_summary(out,train,y);baseline,_=cv_summary(ROOT/a.models/name,train,y)
    tuned_test,per_trip=evaluate_cached(best,rows);pd.DataFrame(per_trip).to_csv(out/'tuned_student_test_per_trip.csv',index=False)
    from pathsnap.training import load_saved_model
    raw=load_saved_model(ROOT/a.models/name/'selected_model.joblib')
    if hasattr(raw,'get_booster'):raw.set_params(device='cpu')
    if hasattr(raw,'to_cpu'):raw.to_cpu()
    baseline_test,_=evaluate_cached(raw,rows)
    pd.DataFrame([dict(configuration='untuned',**baseline,**baseline_test),dict(configuration='tuned',**tuned,**tuned_test)]).to_csv(out/'tuned_vs_untuned.csv',index=False)
    winner='tuned' if selection_key(tuned)>selection_key(baseline) else 'untuned'
    summary=dict(model=name,selected_configuration=winner,completed_trials=len(completed),failed_trials=len([t for t in study.trials if t.state==optuna.trial.TrialState.FAIL]),startup_trials=a.startup,tpe_guided_completed_trials=max(0,len(completed)-a.startup),elapsed_seconds=time.perf_counter()-started,best_value=study.best_value,best_params=study.best_params,study_name=study.study_name)
    write_json(out/'tuning_summary.json',summary)
    source=out/'selected_model.joblib' if winner=='tuned' else ROOT/a.models/name/'selected_model.joblib'
    export_artifact(source,ROOT/a.artifacts,ROOT/a.prepared,name,summary)
    print(summary)
if __name__=='__main__':main()
