import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from streamlit_folium import st_folium
from client import APIClient
from map_view import make_map

st.set_page_config(page_title='PathSnap',page_icon='🧭',layout='wide')
st.title('PathSnap')
st.caption('Match GPS observations to directed roads and reconstruct a connected journey.')
backend=st.sidebar.text_input('Laptop 1 backend URL',os.getenv('BACKEND_URL','http://127.0.0.1:8000'))
request_timeout=float(os.getenv('REQUEST_TIMEOUT_SECONDS','30'))
job_timeout=float(os.getenv('JOB_TIMEOUT_SECONDS','900'))
if st.session_state.get('backend')!=backend:
    st.session_state.update(backend=backend,active_job=None,results={},last_error=None)
try:
    client=APIClient(backend,request_timeout)
    health=client.request('/health',timeout=3)
    if health.get('artifacts_ready') and health.get('data_ready'):
        st.sidebar.success('Backend ready')
    else:st.sidebar.warning(health.get('error') or 'Backend needs data or model artifacts.')
    st.sidebar.caption(f"Mode: {health.get('mode')} · Model: {health.get('model_version')}")
    manifest=client.request('/trips')
except (RuntimeError,ValueError) as exc:
    st.error(str(exc));st.caption('Start the backend, verify its IP and port, then refresh this page.');st.stop()
trips=manifest.get('trips',[])
if not trips:st.warning(manifest.get('warning') or 'No trips have been provisioned.');st.stop()
if manifest['mode']=='development':st.info('Development mode · These are student-test trips, scored against provisional derived references.')
else:st.info('Evaluation mode · Reference routes remain on Laptop 1. Scores use the declared scoring method.')
trip=st.selectbox('Choose a trip',trips,disabled=st.session_state.get('active_job') is not None)
if st.session_state.get('last_error'):st.error(st.session_state.last_error)
ready=health.get('artifacts_ready') and health.get('data_ready')
if st.button('Match and reconstruct',type='primary',disabled=not ready or st.session_state.get('active_job') is not None):
    try:
        jid=client.submit(trip)
        st.session_state.update(active_job=dict(job_id=jid,trip_id=trip,started=time.monotonic()),last_error=None)
        st.rerun()
    except RuntimeError as exc:st.error(str(exc))

@st.fragment(run_every='1s' if st.session_state.get('active_job') else None)
def poll():
    active=st.session_state.get('active_job')
    if not active:return
    try:
        if time.monotonic()-active['started']>job_timeout:raise RuntimeError('Prediction exceeded the UI time limit. Check the backend log and retry.')
        job=client.request('/jobs/'+active['job_id'])
        if job['status']=='failed':raise RuntimeError(job.get('safe_error','Prediction failed.'))
        if job['status']=='completed':
            st.session_state.results[active['trip_id']]=client.request('/results/'+active['job_id'])
            st.session_state.active_job=None
            st.rerun()
        st.info(f"Working on trip {active['trip_id']} · {job['stage'].replace('_',' ')}")
    except RuntimeError as exc:
        st.session_state.last_error=str(exc)
        st.session_state.active_job=None
        st.rerun()
poll()
result=st.session_state.results.get(trip)
if result:
    st.success(f"Trip {trip} reconstructed. Select another trip to continue.")
    metrics=result.get('metrics') or {}
    decoded=metrics.get('decoded',{})
    raw=metrics.get('classifier',{})
    cols=st.columns(5)
    cols[0].metric('GPS observations',result['observations'])
    cols[1].metric('Decoded Top-1',f"{decoded['observation_top1']:.2%}" if decoded.get('observation_top1') is not None else 'Unavailable')
    cols[2].metric('Directed continuity',f"{result['diagnostics']['decoded']['route_continuity']:.2%}" if result['diagnostics']['decoded']['route_continuity'] is not None else 'Unavailable')
    cols[3].metric('Route blocks',result['diagnostics']['blocks'])
    cols[4].metric('Total latency',f"{result['timing']['total_seconds']:.2f} s")
    st.caption(f"Scoring: {result['evaluation_status']} · {result.get('label_method') or 'No reference supplied'} · Disconnected blocks remain separate.")
    st_folium(make_map(result),height=600,use_container_width=True,key='map_'+result['job_id'],returned_objects=[])
    with st.expander('Classifier and decoder comparison'):
        if metrics:
            st.dataframe([dict(pipeline='Classifier',**raw),dict(pipeline='Classifier + Viterbi',**decoded)],use_container_width=True)
        else:st.write('Reference scores are unavailable. Predictions and continuity remain available.')
        st.json(result['timing'])
