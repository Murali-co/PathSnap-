import numpy as np
from pathsnap.decoding.viterbi import emissions,decode_observation_groups,reconstruct
from conftest import candidates


def test_emissions():
    for p in ([.2,.8],[np.nan,np.inf],[]):
        e=emissions(p);assert np.isfinite(e).all()
        if len(e):assert np.isclose(np.exp(e).sum(),1)


def run(net,groups,cfg,times=None):
    offsets=np.r_[0,np.cumsum([len(g.ids) for g in groups])]
    track=np.column_stack((np.zeros((len(groups),2)),np.arange(len(groups)) if times is None else times))
    return decode_observation_groups(net,groups,np.ones(offsets[-1]),offsets,track,cfg),track


def test_empty_middle_retains_both_blocks(tiny_net,cfg):
    groups=[candidates([0]),candidates([]),candidates([1])]
    result,_=run(tiny_net,groups,cfg)
    assert result['states'].tolist()==[0,-1,1]
    assert result['block'].tolist()==[0,-1,1]


def test_empty_first_and_last(tiny_net,cfg):
    result,_=run(tiny_net,[candidates([]),candidates([0]),candidates([])],cfg)
    assert result['states'].tolist()==[-1,0,-1]


def test_impossible_transition_restarts_without_losing_prefix(tiny_net,cfg):
    groups=[candidates([2],[.9]),candidates([0],[.1])]
    result,_=run(tiny_net,groups,{**cfg,'max_speed_mps':1,'movement_slack_m':0})
    assert result['states'].tolist()==[2,0]
    assert result['blocks']==2
    assert result['restarts'][0]['reason']=='no_feasible_transition'


def test_gap_and_single_observation(tiny_net,cfg):
    groups=[candidates([0]),candidates([1])]
    result,track=run(tiny_net,groups,cfg,[0,1000])
    components,_=reconstruct(tiny_net,groups,result,track,cfg)
    assert components==[[0],[1]]
    assert result['blocks']==2


def test_invalid_observation_retains_output(tiny_net,cfg):
    tiny_net.obs[1]=np.nan
    result,_=run(tiny_net,[candidates([0]),candidates([0]),candidates([1])],cfg)
    assert len(result['states'])==3 and result['status'][1]=='invalid_observation'
