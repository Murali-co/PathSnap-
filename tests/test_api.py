from pathlib import Path
from unittest.mock import patch
import json,time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import Service,create_app

class DistanceModel:
    def predict_proba(self,x):
        p=np.exp(-np.minimum(x[:,0],1000)/25)
        return np.column_stack((1-p,p))

@pytest.fixture
def service(tmp_path,monkeypatch):
    monkeypatch.setenv('PATHSNAP_DATA_ROOT',str(tmp_path/'data'))
    monkeypatch.setenv('PATHSNAP_LOG_DIR',str(tmp_path/'logs'))
    monkeypatch.setenv('PATHSNAP_MODE','development')
    monkeypatch.setenv('PATHSNAP_ARTIFACT_DIR',str(tmp_path/'artifacts'))
    base=tmp_path/'data/student_test/00000001';base.mkdir(parents=True)
    (base/'00000001.track').write_text('.0002 0 1\n.0004 0 2\n')
    (base/'00000001.nodes').write_text('0 0\n.001 0\n')
    (base/'00000001.arcs').write_text('0 1\n1 0\n')
    (base/'00000001.route').write_text('0\n')
    s=Service()
    from pathsnap.config import config
    s.model=DistanceModel();s.cfg=config();s.registry={'model_version':'synthetic-test-only'};s.model_error=None
    return s


def wait(client,jid):
    for _ in range(300):
        j=client.get('/jobs/'+jid).json()
        if j['status'] in ('completed','failed'):return j
        time.sleep(.01)
    pytest.fail('Job did not finish')


def test_real_job_repeat_and_private_reference_boundary(service):
    with TestClient(create_app(service)) as c:
        assert c.get('/health').json()['artifacts_ready']
        ids=[]
        for _ in range(2):
            submitted=c.post('/predict/00000001');assert submitted.status_code==202
            jid=submitted.json()['job_id'];ids.append(jid);assert wait(c,jid)['status']=='completed'
            result=c.get('/results/'+jid).json()
            assert result['geometry']['predicted'] and len(result['matches'])==2
            assert result['evaluation_status']=='derived_reference'
            assert result['metrics']['decoded']['known_observations']==2
            assert result['timing']['total_seconds']>=result['timing']['decoder_seconds']
            text=json.dumps(result)
            assert not any(key in text for key in ['"labels"','"reference_geometry"','"reference_arc_ids"','"occurrences"'])
        assert ids[0]!=ids[1]


def test_unknown_trip_and_missing_model(service):
    with TestClient(create_app(service)) as c:
        assert c.post('/predict/unknown').status_code==404
        assert c.get('/jobs/unknown').status_code==404
        service.model=None
        assert not c.get('/health').json()['artifacts_ready']
        assert c.post('/predict/00000001').status_code==503


def test_failure_terminal_status_and_sanitized_error(service):
    with TestClient(create_app(service)) as c:
        with patch('backend.app.main.predict_trip',side_effect=ValueError('/secret/path should not leak')):
            jid=c.post('/predict/00000001').json()['job_id']
            job=wait(c,jid)
            assert job['status']=='failed' and '/secret' not in json.dumps(job)
            assert c.get('/results/'+jid).status_code==409


def test_prediction_works_without_reference(service):
    path=service.root/'student_test/00000001/00000001.route';path.unlink()
    with TestClient(create_app(service)) as c:
        jid=c.post('/predict/00000001').json()['job_id'];assert wait(c,jid)['status']=='completed'
        result=c.get('/results/'+jid).json()
        assert result['evaluation_status']=='reference_unavailable' and result['metrics'] is None
        assert result['geometry']['predicted']


def test_evaluation_root_and_redacted_diagnostics(service):
    base=service.root/'student_test/00000001'
    service.mode='evaluation';service.eval_root=base.parent
    manifest=base.parent/'manifest.json'
    manifest.write_text(json.dumps({'trips':[dict(trip_id='00000001',directory='00000001')]}))
    service.eval_manifest=manifest
    with TestClient(create_app(service)) as c:
        assert c.get('/trips').status_code==503  # incomplete reserved ten cannot masquerade as valid provisioning


def test_concurrency_bound(service):
    with TestClient(create_app(service)) as c:
        with patch('backend.app.main.predict_trip',side_effect=lambda *args,**kwargs:time.sleep(.2)):
            assert c.post('/predict/00000001').status_code==202
            assert c.post('/predict/00000001').status_code==429
