import numpy as np
import pytest
from pathsnap.training import build_model,resolve_device
from pathsnap.features import SCHEMA

@pytest.mark.parametrize('name,params',[
 ('logistic',{}),('decision_tree',{'max_depth':3,'min_samples_leaf':2}),
 ('random_forest',{'n_estimators':3,'max_depth':3}),('xgboost',{'n_estimators':3,'max_depth':2}),
 ('pytorch_mlp',{'epochs':2,'hidden1':8,'hidden2':8,'batch_size':16})])
def test_five_real_families_fit_synthetic_and_serialize_cpu(name,params,tmp_path):
    rng=np.random.default_rng(42);x=rng.normal(size=(80,len(SCHEMA))).astype('float32');y=(x[:,0]>0).astype(int)
    m=build_model(name,device='cpu',params=params);m.fit(x,y);p=m.predict_proba(x)
    assert p.shape==(80,2) and np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)
    import joblib
    joblib.dump(m,tmp_path/'model.joblib');loaded=joblib.load(tmp_path/'model.joblib')
    assert np.allclose(p,loaded.predict_proba(x),atol=1e-6)
    if name=='pytorch_mlp':assert str(loaded.device)=='cpu' and len(loaded.history)==2
