"""Single-process, bounded jobs. Prediction and private scoring are separate stages."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import sys
import threading
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from fastapi import FastAPI,HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from pathsnap.config import ROOT as PROJECT_ROOT,data_root,write_json
from pathsnap.data import input_paths,load_trip
from pathsnap.inference import load_artifact,predict_trip


class Service:
    def __init__(self):
        self.mode=os.getenv('PATHSNAP_MODE','development')
        if self.mode not in ('development','evaluation'):raise ValueError('PATHSNAP_MODE must be development or evaluation')
        try:self.root=data_root()
        except ValueError:self.root=None
        self.artifact=Path(os.getenv('PATHSNAP_ARTIFACT_DIR',PROJECT_ROOT/'artifacts')).resolve()
        self.eval_root=Path(os.environ['PATHSNAP_EVALUATION_DATA_ROOT']).resolve() if os.getenv('PATHSNAP_EVALUATION_DATA_ROOT') else None
        self.eval_manifest=Path(os.environ['PATHSNAP_EVALUATION_MANIFEST']).resolve() if os.getenv('PATHSNAP_EVALUATION_MANIFEST') else None
        self.reference_root=Path(os.environ['PATHSNAP_REFERENCE_ROOT']).resolve() if os.getenv('PATHSNAP_REFERENCE_ROOT') else None
        self.log_dir=Path(os.getenv('PATHSNAP_LOG_DIR',PROJECT_ROOT/'outputs/logs'))
        self.log_dir.mkdir(parents=True,exist_ok=True)
        self.logger=logging.getLogger('pathsnap.'+uuid.uuid4().hex)
        self.logger.setLevel(logging.INFO)
        formatter=logging.Formatter('%(asctime)s %(message)s')
        for handler in (logging.StreamHandler(),RotatingFileHandler(self.log_dir/'backend.log',maxBytes=5_000_000,backupCount=2)):
            handler.setFormatter(formatter);self.logger.addHandler(handler)
        self.jobs={};self.lock=threading.RLock()
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pathsnap')
        self.max_jobs=int(os.getenv('PATHSNAP_MAX_JOBS','32'))
        self.timeout=float(os.getenv('PATHSNAP_JOB_TIMEOUT','900'))
        self.model,self.registry,self.cfg=None,{},None
        self.model_error='Model artifacts are unavailable; export a trained model first.'
        self.load_model()

    def load_model(self):
        try:
            self.model,self.registry,self.cfg=load_artifact(self.artifact)
            self.model_error=None
        except Exception:
            self.logger.exception('stage=model_load status=failed')

    def manifest(self):
        if self.mode=='development':
            base=self.root/'student_test' if self.root else None
            if base is None or not base.is_dir():return {}
            return {p.name:p for p in sorted(base.iterdir()) if p.is_dir() and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',p.name) and all(x.is_file() for x in input_paths(p))}
        if self.eval_manifest:
            obj=json.loads(self.eval_manifest.read_text())
            items=obj.get('trips',[])
            if len(items)!=10:raise ValueError('Evaluation manifest must contain exactly ten trips.')
            entries={}
            for item in items:
                tid=item['trip_id'];relative=Path(item.get('directory',tid))
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',tid) or relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid evaluation manifest entry.')
                base=self.eval_root or self.eval_manifest.parent
                directory=(base/relative).resolve()
                if not directory.is_relative_to(base.resolve()):raise ValueError('Evaluation directory escapes data root.')
                if not all(p.is_file() for p in input_paths(directory,tid)):raise ValueError('Evaluation input file unavailable.')
                if tid in entries:raise ValueError('Duplicate evaluation ID.')
                entries[tid]=directory
            return entries
        if self.eval_root and self.eval_root.is_dir():
            entries={p.name:p for p in sorted(self.eval_root.iterdir()) if p.is_dir() and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',p.name) and all(x.is_file() for x in input_paths(p))}
            if len(entries)!=10:raise ValueError('Evaluation data root must contain exactly ten complete trips.')
            return entries
        return {}

    def update(self,jid,**values):
        with self.lock:
            self.jobs[jid].update(values,updated_at=time.time())
        self.logger.info('job_id=%s trip_id=%s stage=%s status=%s model=%s',jid,self.jobs[jid]['trip_id'],self.jobs[jid]['stage'],self.jobs[jid]['status'],self.registry.get('model_version','unavailable'))

    def run(self,jid,base,trip_id):
        start=time.perf_counter()
        try:
            self.update(jid,status='running',stage='loading')
            trip=load_trip(*input_paths(base,trip_id))
            def progress(stage):
                if time.perf_counter()-start>self.timeout:raise TimeoutError('Job exceeded time limit')
                self.update(jid,stage=stage)
            result,context=predict_trip(trip,self.model,self.cfg,progress)
            progress('private_evaluation')
            reference=(self.reference_root/trip_id/f'{trip_id}.route') if self.reference_root else base/f'{trip_id}.route'
            score=dict(evaluation_status='reference_unavailable',label_method=None,metrics=None)
            if reference.is_file():
                from pathsnap.evaluation.scorer import score_prediction
                try:
                    score=score_prediction(context,reference,self.cfg)
                    if self.mode=='evaluation':
                        score['metrics'].pop('label_ambiguity_fraction',None)
                        score['metrics'].pop('reference_breaks',None)
                except Exception:
                    self.logger.exception('job_id=%s stage=private_evaluation status=failed',jid)
                    score=dict(evaluation_status='scoring_failed',label_method=self.cfg['label_method'],metrics=None)
            result.update(score,job_id=jid,trip_id=trip_id,status='completed',mode=self.mode,
                          model_version=self.registry.get('model_version','unknown'))
            result['timing']['total_seconds']=time.perf_counter()-start
            self.update(jid,status='completed',stage='completed',result=result)
            self.logger.info('job_id=%s duration_seconds=%.6f success=true',jid,result['timing']['total_seconds'])
        except Exception:
            self.logger.exception('job_id=%s status=failed',jid)
            self.update(jid,status='failed',stage='failed',safe_error='Prediction failed. Check the backend log, then retry.')

    def cleanup(self):
        finished=[k for k,v in self.jobs.items() if v['status'] in ('completed','failed')]
        while len(self.jobs)>=self.max_jobs and finished:self.jobs.pop(finished.pop(0))

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)
        for h in self.logger.handlers[:]:h.close();self.logger.removeHandler(h)


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        if not hasattr(app.state,'service'):app.state.service=Service()
        yield
        app.state.service.close()
    app=FastAPI(title='PathSnap API',version='2.0',lifespan=lifespan)
    if service is not None:app.state.service=service
    origins=[s.strip() for s in os.getenv('PATHSNAP_ALLOWED_ORIGINS','').split(',') if s.strip()]
    if origins:app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type'])
    def current():
        if not hasattr(app.state,'service'):app.state.service=Service()
        return app.state.service

    @app.middleware('http')
    async def log_request(request:Request,call_next):
        request_id=uuid.uuid4().hex
        response=await call_next(request)
        if hasattr(app.state,'service'):
            app.state.service.logger.info('request_id=%s method=%s endpoint=%s status=%s',request_id,request.method,request.url.path,response.status_code)
        response.headers['X-Request-ID']=request_id
        return response

    @app.get('/health')
    def health():
        s=current()
        try:count=len(s.manifest());data_error=None
        except Exception:count=0;data_error='Trip configuration is invalid. Check the backend log/configuration.'
        return dict(status='ok' if s.model is not None and count else 'not_ready',mode=s.mode,
                    model_version=s.registry.get('model_version','unavailable'),artifacts_ready=s.model is not None,
                    data_ready=count>0,trip_count=count,error=s.model_error or data_error)

    @app.get('/trips')
    def trips():
        s=current()
        try:entries=s.manifest()
        except Exception:raise HTTPException(503,'Trip configuration is invalid.')
        return dict(mode=s.mode,trips=list(entries),warning=None if entries else 'Provision the configured data partition first.')

    @app.post('/predict/{trip_id}',status_code=202)
    def predict(trip_id:str):
        s=current()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',trip_id):raise HTTPException(404,'Unknown trip ID')
        try:entries=s.manifest()
        except Exception:raise HTTPException(503,'Trip configuration is invalid.')
        if trip_id not in entries:raise HTTPException(404,'Unknown trip ID')
        if s.model is None:raise HTTPException(503,s.model_error)
        with s.lock:
            s.cleanup()
            if any(v['status'] in ('queued','running') for v in s.jobs.values()):raise HTTPException(429,'A prediction is already running. Retry after it completes.')
            if len(s.jobs)>=s.max_jobs:raise HTTPException(429,'Job capacity reached. Retry later.')
            jid=uuid.uuid4().hex
            s.jobs[jid]=dict(job_id=jid,trip_id=trip_id,status='queued',stage='pending',created_at=time.time(),updated_at=time.time())
            s.executor.submit(s.run,jid,entries[trip_id],trip_id)
        return dict(job_id=jid,status='queued')

    @app.get('/jobs/{job_id}')
    def job(job_id:str):
        s=current()
        with s.lock:
            if job_id not in s.jobs:raise HTTPException(404,'Job unavailable or expired; submit the trip again.')
            return {k:v for k,v in s.jobs[job_id].items() if k!='result'}

    @app.get('/results/{job_id}')
    def result(job_id:str):
        s=current()
        with s.lock:
            if job_id not in s.jobs:raise HTTPException(404,'Job unavailable or expired; submit the trip again.')
            j=s.jobs[job_id]
            if j['status']!='completed':raise HTTPException(409,'Job is not completed')
            return j['result']
    return app


app=create_app()
