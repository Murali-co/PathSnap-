#!/usr/bin/env python3
"""Real browser integration of the supplied React frontend with the saved-model API."""
from pathlib import Path
import json,os,shutil,socket,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'frontend'))
from client import APIClient
from playwright.sync_api import sync_playwright

def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def main():
    out=ROOT/'outputs/verification';out.mkdir(exist_ok=True,parents=True);backend_port,web_port=port(),port()
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT),PATHSNAP_DATA_ROOT=str(ROOT.parent),PATHSNAP_MODE='development',PATHSNAP_LOG_DIR=str(out/'web_logs'),VITE_API_BASE_URL=f'http://127.0.0.1:{backend_port}')
    logs=[(out/'react_backend.log').open('w'),(out/'react_frontend.log').open('w')]
    processes=[subprocess.Popen([sys.executable,'-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port',str(backend_port)],cwd=ROOT,env=env,stdout=logs[0],stderr=subprocess.STDOUT),subprocess.Popen(['npm','run','dev','--','--host','127.0.0.1','--port',str(web_port),'--strictPort'],cwd=ROOT/'frontend-web',env=env,stdout=logs[1],stderr=subprocess.STDOUT)]
    try:
        client=APIClient(f'http://127.0.0.1:{backend_port}')
        for _ in range(100):
            try:
                if client.request('/health',timeout=1)['artifacts_ready']:break
            except RuntimeError:pass
            time.sleep(.2)
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=shutil.which('google-chrome'),headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
            page=browser.new_page(viewport={'width':1450,'height':1100});errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{web_port}',wait_until='domcontentloaded',timeout=30000)
            page.locator('#trip-select option').nth(1).wait_for(state='attached',timeout=30000)
            assert page.locator('#trip-select option').count()==10
            page.select_option('#trip-select','00000052');run=page.get_by_role('button',name='Run map match')
            run.click();page.get_by_text('Match overview',exact=True).wait_for(timeout=60000)
            page.get_by_text('Decoded route from the API',exact=False).wait_for(timeout=10000)
            assert page.get_by_text('96.56%',exact=True).is_visible()
            page.screenshot(path=str(out/'react_ui.png'),full_page=True)
            run.click();page.get_by_text('Match overview',exact=True).wait_for(timeout=60000)
            assert run.is_enabled()
            assert not errors,errors
            browser.close()
        result=dict(react_build_passed=True,ten_trip_options=True,actual_backend_prediction=True,decoded_geometry_displayed=True,aggregate_score_displayed=True,repeated_run=True,browser_errors=errors,physical_lan_test=False)
        (out/'react_browser_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
    finally:
        for p in processes:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in logs:f.close()
if __name__=='__main__':main()
