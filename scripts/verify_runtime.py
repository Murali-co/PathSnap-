#!/usr/bin/env python3
"""Launch the real HTTP backend, exercise every development trip and repeated requests."""
import argparse,json,os,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'frontend'));sys.path.insert(0,str(ROOT/'src'))
from client import APIClient
from pathsnap.config import write_json


def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',default='outputs/verification');p.add_argument('--limit',type=int);p.add_argument('--artifacts',default='artifacts');a=p.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True);port=free_port()
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT),PATHSNAP_DATA_ROOT=str(Path(a.data_root).resolve()),PATHSNAP_ARTIFACT_DIR=str((ROOT/a.artifacts).resolve()),PATHSNAP_MODE='development',PATHSNAP_LOG_DIR=str(out/'logs'))
    log=(out/'http_server.log').open('w');server=subprocess.Popen([sys.executable,'-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    client=APIClient(f'http://127.0.0.1:{port}',30);results=[]
    try:
        for _ in range(120):
            if server.poll() is not None:raise RuntimeError('Backend exited; inspect http_server.log')
            try:
                health=client.request('/health',timeout=2)
                if health.get('artifacts_ready') and health.get('data_ready'):break
            except RuntimeError:pass
            time.sleep(.5)
        else:raise RuntimeError('Backend readiness timeout.')
        trips=client.request('/trips')['trips']
        if a.limit:trips=trips[:a.limit]
        for tid in trips:
            start=time.perf_counter();jid=client.submit(tid);result=client.wait(jid,timeout=900)
            assert len(result['matches'])==result['observations']
            assert result['geometry']['predicted']
            assert result['timing']['total_seconds']>=result['timing']['decoder_seconds']
            assert not any(k in json.dumps(result) for k in ['"reference_geometry"','"reference_arc_ids"','"labels"','"occurrences"'])
            write_json(out/f'{tid}.json',result)
            row=dict(trip_id=tid,status='completed',wall_seconds=time.perf_counter()-start,reported_seconds=result['timing']['total_seconds'],blocks=result['diagnostics']['blocks'])
            for name,m in (result.get('metrics') or {}).items():
                if name in ('classifier','decoded'):
                    row[name+'_top1']=m['observation_top1'];row[name+'_continuity']=m['route_continuity'];row[name+'_edit_distance']=m['normalized_edit_distance']
            results.append(row);print(row,flush=True)
        jid=client.submit(trips[0]);repeated=client.wait(jid,900)
        assert repeated['trip_id']==trips[0]
        try:client.submit('not-an-allowed-trip')
        except RuntimeError as exc:assert '404' in str(exc)
        else:raise AssertionError('Unknown trip accepted')
        import pandas as pd
        pd.DataFrame(results).to_csv(out/'api_trip_metrics.csv',index=False)
        write_json(out/'summary.json',dict(mode='development',trip_count=len(results),repeated_prediction_passed=True,unknown_trip_rejected=True,physical_lan_test=False,official_evaluation=False))
        print('Runtime verification passed',flush=True)
    finally:
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait()
        log.close()
if __name__=='__main__':main()
