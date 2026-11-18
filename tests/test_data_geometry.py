from pathlib import Path
import numpy as np
import pytest
from pathsnap.spatial import project_to_segments
from pathsnap.data import load_trip
from pathsnap.features import extract,motion,SCHEMA
from pathsnap.reference import align


def test_projection_zero_length_and_known_distance(tiny_net):
    d,f=project_to_segments(np.array([1.,1.]),np.array([[0.,0.],[0.,0.]]),np.array([[10.,0.],[0.,0.]]))
    assert np.allclose(d,[1,np.sqrt(2)]) and np.allclose(f,[.1,0])
    assert 110<tiny_net.lengths[0]<112


def test_long_segment_spatial_lookup_and_ids(tiny_trip,cfg):
    tiny_trip.nodes[1]=[.1,0]
    from pathsnap.spatial import Network
    net=Network(tiny_trip);point=np.array(net.forward.transform(.00001,0.))
    candidates=net.candidates(point,cfg)
    assert 0 in candidates.ids and 3 in candidates.ids and 8 in candidates.ids


def test_routing_duplicate_arc_not_summed_and_cutoff_safe(tiny_net):
    distance,path=tiny_net.shortest(0,2,300)
    assert 220<distance<225 and len(path)==2
    assert np.isinf(tiny_net.shortest(0,2,100)[0])
    assert np.isfinite(tiny_net.shortest(0,2,300)[0])


def test_same_arc_forward_backwards_and_zero_length(tiny_net,cfg):
    assert np.isfinite(tiny_net.transition(0,.1,0,.3,1,20,cfg)[0])
    strict={**cfg,'max_speed_mps':1,'movement_slack_m':0,'backward_tolerance_m':1}
    assert np.isinf(tiny_net.transition(0,.8,0,.1,1,0,strict)[0])
    assert tiny_net.transition(9,.2,9,.9,1,0,cfg)[0]==0


def test_features_finite_and_reference_free(tiny_trip,cfg,tmp_path):
    _,groups,offsets,x=extract(tiny_trip,cfg)
    assert x.shape[1]==len(SCHEMA) and np.isfinite(x).all()
    assert offsets[-1]==sum(len(g.ids) for g in groups)
    forbidden={'trip_id','arc_id','route_membership','reference_distance'}
    assert not forbidden.intersection(SCHEMA)


def test_time_window_respects_gap(tiny_trip,cfg):
    from pathsnap.spatial import Network
    tiny_trip.track[:,2]=[0,1,200,201]
    m=motion(tiny_trip,Network(tiny_trip),cfg)
    assert m['displacement'][2]==0 and m['straightness'][2]==0
    assert 0<=m['straightness'][3]<=1


def test_label_occurrences_monotone_with_repeat(tiny_trip,tiny_net,cfg):
    route=np.array([0,1,2,5,4,3,0,1])
    a=align(tiny_trip,tiny_net,route,cfg)
    assert np.all(np.diff(a['occurrences'])>=0)
    assert len(a['ambiguous'])==len(tiny_trip.track)


def test_loader_rejects_invalid_index_preserves_invalid_observation(tmp_path):
    (tmp_path/'t.track').write_text('0 0 1\nnan 0 2\n')
    (tmp_path/'t.nodes').write_text('0 0\n.001 0\n')
    (tmp_path/'t.arcs').write_text('0 1\n')
    trip=load_trip(tmp_path/'t.track',tmp_path/'t.nodes',tmp_path/'t.arcs')
    assert trip.valid.tolist()==[True,False]
    (tmp_path/'t.arcs').write_text('0 2\n')
    with pytest.raises(ValueError,match='row 0'):load_trip(tmp_path/'t.track',tmp_path/'t.nodes',tmp_path/'t.arcs')
