"""Reproducible model factories and grouped full-trip cross-validation."""
from copy import deepcopy
from pathlib import Path
import hashlib
import io
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from .config import write_json,digest
from .features import SCHEMA
from .evaluation.metrics import candidate_metrics,observation_top1

MODELS=('logistic','decision_tree','random_forest','xgboost','pytorch_mlp')


def load_prepared(output):
    base=Path(output);manifest=json.loads((base/'feature_manifest.json').read_text())
    if manifest['schema']!=SCHEMA:raise ValueError('Prepared feature order is incompatible; run prepare again.')
    rows=[]
    for e in manifest['entries']:
        with np.load(base/e['file']) as archive:
            a={k:archive[k] for k in archive.files}
        if a['schema'].tolist()!=SCHEMA or str(a['identity'])!=e['identity']:
            raise ValueError('Feature cache identity/schema mismatch; reprepare the data.')
        if not np.isfinite(a['x']).all():raise ValueError('Nonfinite prepared feature values.')
        rows.append((e['partition'],e['trip_id'],a))
    return manifest,rows


def resolve_device(requested='auto'):
    if requested not in ('auto','cpu','cuda'):raise ValueError('device must be auto, cpu or cuda')
    if requested=='cpu':return 'cpu'
    import torch
    available=torch.cuda.is_available()
    if requested=='cuda' and not available:raise RuntimeError('CUDA requested but unavailable. Check the NVIDIA driver and PyTorch installation.')
    if available:
        tensor=torch.ones(8,device='cuda');assert float((tensor*2).sum())==16
        return 'cuda'
    return 'cpu'


def build_model(name,seed=42,device='auto',params=None):
    params=params or {}
    if name=='logistic':
        return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=700,class_weight='balanced',random_state=seed,**params))
    if name=='decision_tree':
        values=dict(max_depth=12,min_samples_leaf=20,class_weight='balanced',random_state=seed);values.update(params)
        return DecisionTreeClassifier(**values)
    if name=='random_forest':
        values=dict(n_estimators=200,max_depth=16,min_samples_leaf=5,class_weight='balanced',n_jobs=2,random_state=seed);values.update(params)
        return RandomForestClassifier(**values)
    if name=='xgboost':
        import xgboost as xgb
        values=dict(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,
                    objective='binary:logistic',eval_metric='logloss',tree_method='hist',device=resolve_device(device),random_state=seed,n_jobs=2)
        values.update(params)
        return xgb.XGBClassifier(**values)
    if name=='pytorch_mlp':return TorchMLP(seed=seed,device=resolve_device(device),**params)
    raise ValueError('Unknown model '+name)


def fit_models(X=None,y=None,seed=42,device='auto',model_filter=None):
    names=[model_filter] if model_filter else MODELS
    return {name:build_model(name,seed,device) for name in names}


class TorchMLP:
    """Batched GPU/CPU classifier with fold-local scaling and portable CPU persistence."""
    def __init__(self,seed=42,hidden1=64,hidden2=64,epochs=20,batch_size=4096,lr=1e-3,weight_decay=1e-4,dropout=.1,device='auto'):
        self.seed=seed;self.hidden1=hidden1;self.hidden2=hidden2;self.epochs=epochs;self.batch_size=batch_size
        self.lr=lr;self.weight_decay=weight_decay;self.dropout=dropout;self.history=[]
        self.device=resolve_device(device);self.mean=None;self.std=None

    def fit(self,X,y):
        import torch
        from torch import nn
        torch.set_num_threads(2)
        X=np.asarray(X,dtype=np.float32);y=np.asarray(y,dtype=np.float32)
        if not len(y) or len(np.unique(y))<2:raise ValueError('Training requires both binary classes.')
        torch.manual_seed(self.seed)
        if str(self.device).startswith('cuda'):torch.cuda.manual_seed_all(self.seed)
        self.mean=X.mean(axis=0);self.std=X.std(axis=0);self.std[self.std<1e-6]=1
        self.n_features_in_=X.shape[1]
        xs=torch.from_numpy((X-self.mean)/self.std);ys=torch.from_numpy(y).view(-1,1)
        self.net=nn.Sequential(nn.Linear(X.shape[1],self.hidden1),nn.ReLU(),nn.Linear(self.hidden1,self.hidden2),
                               nn.ReLU(),nn.Dropout(getattr(self,'dropout',.1)),nn.Linear(self.hidden2,1)).to(self.device)
        optimizer=torch.optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        loss_fn=nn.BCEWithLogitsLoss();self.history=[]
        generator=torch.Generator().manual_seed(self.seed)
        self.net.train()
        for epoch in range(self.epochs):
            order=torch.randperm(len(xs),generator=generator);total=0.
            for lo in range(0,len(xs),self.batch_size):
                ix=order[lo:lo+self.batch_size];xb=xs[ix].to(self.device);yb=ys[ix].to(self.device)
                loss=loss_fn(self.net(xb),yb);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                total+=float(loss.detach().cpu())*len(ix)
            self.history.append(total/len(xs))
        return self

    def predict_proba(self,X):
        import torch
        X=(np.asarray(X,dtype=np.float32)-self.mean)/self.std
        self.net.eval();parts=[]
        with torch.no_grad():
            for lo in range(0,len(X),getattr(self,'batch_size',4096)):
                parts.append(torch.sigmoid(self.net(torch.from_numpy(X[lo:lo+self.batch_size]).to(self.device))).view(-1).cpu().numpy())
        probability=np.concatenate(parts) if parts else np.empty(0)
        return np.column_stack((1-probability,probability))

    def to_cpu(self):
        self.net.cpu();self.device='cpu';return self

    def __getstate__(self):
        state=self.__dict__.copy()
        if 'net' in state:state['net']=deepcopy(state['net']).cpu()
        state['device']='cpu'
        return state


def load_saved_model(path):
    try:return joblib.load(path)
    except RuntimeError as exc:
        if 'CUDA' not in str(exc) and 'cuda' not in str(exc):raise
        # Migration for old locally supplied Torch joblib artifacts saved on CUDA.
        import torch
        original=torch.storage._load_from_bytes
        try:
            torch.storage._load_from_bytes=lambda data:torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
            model=joblib.load(path)
            return model.to_cpu() if hasattr(model,'to_cpu') else model
        finally:torch.storage._load_from_bytes=original


def training_arrays(rows):
    train=[r for r in rows if r[0]=='train']
    xs,ys,groups=[],[],[]
    for index,(_,tid,a) in enumerate(train):
        keep=np.repeat(a['labels']>=0,np.diff(a['offsets']))
        xs.append(a['x'][keep]);ys.append(a['y'][keep]);groups.append(np.full(int(keep.sum()),index,dtype=np.int32))
    if not train:raise ValueError('No training trips in feature manifest.')
    X,y,g=np.concatenate(xs),np.concatenate(ys),np.concatenate(groups)
    if len(np.unique(y))<2:raise ValueError('Training requires both binary classes.')
    return X,y,g,train


def oof_trip_metrics(train,probabilities):
    rows=[];lo=0
    for _,tid,a in train:
        counts=np.diff(a['offsets']);known=a['labels']>=0
        keep=np.repeat(known,counts)
        ids=a['ids'][keep];offsets=np.r_[0,np.cumsum(counts[known])]
        hi=lo+len(ids)
        row=observation_top1(ids,probabilities[lo:hi],offsets,a['labels'][known])
        row.update(trip_id=tid,full_observations=len(a['labels']),label_coverage=float(known.mean()))
        rows.append(row);lo=hi
    if lo!=len(probabilities):raise ValueError('OOF predictions do not correspond to prepared rows.')
    return rows


def selection_key(row):
    return (row['cv_mean_trip_top1'],row['cv_macro_f1'],-row.get('cv_conditional_observation_loss',1e9),
            -row.get('cv_top1_std',1e9),-row.get('fit_seconds',1e9))


def train_one(name,prepared,output,folds=3,seed=42,device='auto',params=None,skip_cv=False,resume=False,allow_smoke=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest,rows=load_prepared(prepared)
    X,y,groups,train=training_arrays(rows)
    if not allow_smoke and (len(train)!=80 or manifest.get('smoke_subset')):raise ValueError('Full training requires all 80 training trips; use --allow-smoke only for tests.')
    identity=digest(dict(entries=manifest['entries'],schema=SCHEMA,config=manifest['config'],model=name,params=params or {},folds=folds,seed=seed,device=device,training_version=2))
    meta_path=out/'run.json'
    if resume and meta_path.exists():
        old=json.loads(meta_path.read_text())
        if old.get('identity')==identity and old.get('status')=='completed' and (out/'selected_model.joblib').exists():return old
    write_json(meta_path,dict(identity=identity,status='running',model=name,training_trips=len(train),config=manifest['config'],seed=seed))
    records=[];oof=np.full(len(y),np.nan);fold_assignments={};histories={}
    if not skip_cv:
        for fold,(tr,va) in enumerate(GroupKFold(folds).split(X,y,groups)):
            train_groups=set(groups[tr]);valid_groups=set(groups[va]);assert train_groups.isdisjoint(valid_groups)
            fold_assignments[str(fold)]=dict(train=[train[i][1] for i in sorted(train_groups)],validation=[train[i][1] for i in sorted(valid_groups)])
            model=build_model(name,seed,device,params);t=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(X[tr],y[tr])
            fit_seconds=time.perf_counter()-t;t=time.perf_counter()
            p=model.predict_proba(X[va])[:,1];prediction_seconds=time.perf_counter()-t;oof[va]=p
            records.append(dict(model=name,fold=fold,fit_seconds=fit_seconds,prediction_seconds=prediction_seconds,
                                **candidate_metrics(y[va],p>=.5,p),warnings=[str(w.message) for w in caught]))
            if hasattr(model,'history'):histories[str(fold)]=model.history
            write_json(out/'fold_assignments.json',fold_assignments)
            pd.DataFrame(records).to_csv(out/'cv_metrics.csv',index=False)
            print(f'{name}: fold {fold+1}/{folds} complete',flush=True)
        np.save(out/'oof_probability.npy',oof)
        trip_metrics=oof_trip_metrics(train,oof);pd.DataFrame(trip_metrics).to_csv(out/'cv_per_trip.csv',index=False)
        pooled=candidate_metrics(y,oof>=.5,oof)
        summary=dict(cv_mean_trip_top1=float(np.mean([r['observation_top1'] for r in trip_metrics])),
                     cv_macro_f1=pooled['candidate_macro_f1'],cv_top1_std=float(np.std([r['observation_top1'] for r in trip_metrics])),
                     cv_conditional_observation_loss=float(np.mean([r['conditional_observation_loss'] for r in trip_metrics if r['conditional_observation_loss'] is not None])))
    else:summary={}
    final=build_model(name,seed,device,params);t=time.perf_counter();final.fit(X,y);fit_seconds=time.perf_counter()-t
    actual_device='cpu'
    if hasattr(final,'get_booster'):
        actual_device=json.loads(final.get_booster().save_config())['learner']['generic_param']['device']
        if device=='cuda' and not actual_device.startswith('cuda'):raise RuntimeError('XGBoost silently fell back to CPU; requested GPU training did not occur.')
    if hasattr(final,'device'):actual_device=str(final.device)
    if hasattr(final,'history'):histories['full_fit']=final.history
    write_json(out/'loss_history.json',histories)
    joblib.dump(final,out/'selected_model.joblib')
    pd.DataFrame([dict(model=name,training_trips=len(train),resolved_candidate_rows=len(y),positive_candidate_rows=int(y.sum()),
                       negative_candidate_rows=int((1-y).sum()),fit_seconds=fit_seconds,partition='train_full_resolved',device=actual_device)]).to_csv(out/'full_fit.csv',index=False)
    meta=dict(identity=identity,status='completed',model=name,training_trips=len(train),candidate_rows=len(y),params=params or {},
              config=manifest['config'],feature_schema=SCHEMA,seed=seed,actual_device=actual_device,fit_seconds=fit_seconds,**summary)
    write_json(meta_path,meta)
    return meta


def benchmark(root_output,rows,folds=3,seed=42,model_filter=None):
    # Kept for CLI compatibility; the complete pipeline uses train_one and export.
    X,y,g,train=training_arrays(rows);results=[];oof={}
    for name in ([model_filter] if model_filter else MODELS):
        p=np.empty(len(y))
        for fold,(tr,va) in enumerate(GroupKFold(folds).split(X,y,g)):
            m=build_model(name,seed);t=time.perf_counter();m.fit(X[tr],y[tr]);p[va]=m.predict_proba(X[va])[:,1]
            results.append(dict(model=name,fold=fold,fit_seconds=time.perf_counter()-t,**candidate_metrics(y[va],p[va]>=.5,p[va])))
        oof[name]=p
    Path(root_output).mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(Path(root_output)/'cv_metrics.csv',index=False)
    np.savez(Path(root_output)/'oof_predictions.npz',**oof)
    return results,oof
