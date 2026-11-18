#!/usr/bin/env python3
"""Bounded, resumable pipeline cycles. Failing stages stop before stale export."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pathsnap.config import data_root,write_json
from pathsnap.training import MODELS


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root');p.add_argument('--cycles',type=int,default=1)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--trials',type=int,default=15)
    p.add_argument('--resume',action='store_true');p.add_argument('--skip-training',action='store_true');p.add_argument('--skip-tuning',action='store_true')
    p.add_argument('--skip-audit',action='store_true');p.add_argument('--skip-prepare',action='store_true');p.add_argument('--skip-verification',action='store_true')
    a=p.parse_args()
    if not 1<=a.cycles<=10:raise SystemExit('--cycles must be between 1 and 10.')
    root=data_root(a.data_root);env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT),MPLCONFIGDIR=str(ROOT/'outputs/.mplcache'),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    state=dict(started_at=time.time(),status='running',cycles=a.cycles,stages=[]);ledger=ROOT/'outputs/pipeline_state.json'
    def run(stage,*cmd):
        item=dict(stage=stage,status='running',started_at=time.time());state['stages'].append(item);write_json(ledger,state)
        print('\nPathSnap stage: '+stage,flush=True)
        try:subprocess.run([sys.executable,*map(str,cmd)],cwd=ROOT,env=env,check=True)
        except Exception:
            item.update(status='failed',finished_at=time.time());state['status']='failed';write_json(ledger,state);raise
        item.update(status='completed',finished_at=time.time());write_json(ledger,state)
    for cycle in range(a.cycles):
        state['cycle']=cycle+1
        run('environment','scripts/check_environment.py','--role','backend' if a.skip_training else 'training',*(['--require-gpu'] if a.device=='cuda' and not a.skip_training else []))
        if not a.skip_audit:run('audit','scripts/check_data.py','--data-root',root)
        if not a.skip_prepare:run('prepare','-m','pathsnap.cli','prepare','--data-root',root)
        if not a.skip_training:
            for name in MODELS:run('train_'+name,'scripts/train_model.py','--model',name,'--device',a.device,*(['--resume'] if a.resume else []))
        run('compare','scripts/evaluate_models.py')
        if not a.skip_tuning:run('bayesian_tuning','scripts/tune_and_export.py','--trials',a.trials,'--device',a.device)
        else:run('export_baseline','scripts/export_model.py')
        run('regression_tests','-m','pytest','-q')
        if not a.skip_verification:run('api_loop','scripts/verify_runtime.py','--data-root',root)
        run('technical_report','scripts/generate_report.py')
        run('package','scripts/package_dist.py')
    state.update(status='completed',finished_at=time.time());write_json(ledger,state)
    print('Pipeline complete. See outputs/pipeline_state.json and dist/.')
if __name__=='__main__':main()
