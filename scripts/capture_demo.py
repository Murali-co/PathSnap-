#!/usr/bin/env python3
"""Capture only the local project UI and vector map; no desktop/private content."""
import os,socket,subprocess,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'frontend'));sys.path.insert(0,str(ROOT/'src'))
from client import APIClient
from map_view import make_map
from playwright.sync_api import sync_playwright

def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def main():
    out=ROOT/'outputs/verification';out.mkdir(parents=True,exist_ok=True);bp,fp=port(),port()
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT),PATHSNAP_DATA_ROOT=str(ROOT.parent),PATHSNAP_MODE='development',PATHSNAP_LOG_DIR=str(out/'logs'),BACKEND_URL=f'http://127.0.0.1:{bp}')
    logs=[(out/'demo_backend.log').open('w'),(out/'demo_frontend.log').open('w')]
    processes=[subprocess.Popen([sys.executable,'-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port',str(bp)],cwd=ROOT,env=env,stdout=logs[0],stderr=subprocess.STDOUT),
               subprocess.Popen([sys.executable,'-m','streamlit','run','frontend/app/app.py','--server.address','127.0.0.1','--server.port',str(fp),'--server.headless','true','--browser.gatherUsageStats','false'],cwd=ROOT,env=env,stdout=logs[1],stderr=subprocess.STDOUT)]
    try:
        client=APIClient(env['BACKEND_URL'])
        for _ in range(90):
            try:
                if client.request('/health',timeout=1)['artifacts_ready']:break
            except RuntimeError:pass
            time.sleep(.3)
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
            page=browser.new_page(viewport={'width':1440,'height':1100})
            page.goto(f'http://127.0.0.1:{fp}',wait_until='domcontentloaded',timeout=30000)
            button=page.get_by_role('button',name='Match and reconstruct',exact=True);button.wait_for(timeout=45000);button.click()
            page.get_by_text('Trip 00000000 reconstructed.',exact=False).wait_for(timeout=60000)
            page.screenshot(path=str(out/'ui.png'),full_page=True)
            # Repeated action verifies the selector/button loop remains usable after completion.
            button.click();page.get_by_text('Trip 00000000 reconstructed.',exact=False).wait_for(timeout=60000)
            sample=json.loads((out/'00000052.json').read_text());html=make_map(sample).get_root().render()
            external=[]
            map_page=browser.new_page(viewport={'width':1200,'height':800})
            map_page.route('http://**/*',lambda route:(external.append(route.request.url),route.abort()))
            map_page.route('https://**/*',lambda route:(external.append(route.request.url),route.abort()))
            map_page.set_content(html,wait_until='load');map_page.locator('.leaflet-container').wait_for();map_page.screenshot(path=str(out/'map.png'))
            assert not external,external
            (out/'browser_checks.json').write_text(json.dumps(dict(ui_prediction_completed=True,repeat_button_worked=True,offline_map_external_requests=external),indent=2))
            browser.close()
        print('Browser UI and fully offline map checks passed')
    finally:
        for process in processes:process.terminate()
        for process in processes:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in logs:log.close()
if __name__=='__main__':main()
