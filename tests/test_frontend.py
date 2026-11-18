import importlib.util
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontend'))
from client import APIClient
from map_view import make_map


def test_client_rejects_bad_url():
    with pytest.raises(ValueError):APIClient('not a url')


def test_offline_map_contains_prediction_without_external_assets():
    result=dict(job_id='test',geometry=dict(gps=[[0,0],[.001,0]],gps_segments=[[[0,0],[.001,0]]],network=[[[0,0],[.001,0]]],predicted=[[[0,0],[.001,0]]],classifier=[]),matches=[])
    html=make_map(result).get_root().render()
    import re
    assert not re.search(r'<script[^>]+src=',html)
    assert not re.search(r'<link[^>]+href=',html)
    assert 'Decoded route' in html and 'Leaflet' in html
