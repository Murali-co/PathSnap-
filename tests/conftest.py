import numpy as np
import pytest
from pathsnap.data import Trip
from pathsnap.spatial import Network,Candidates
from pathsnap.config import config

@pytest.fixture
def cfg():return config()

@pytest.fixture
def tiny_trip():
    nodes=np.array([[0.,0.],[.001,0.],[.002,0.],[.003,0.],[.001,.001]])
    arcs=np.array([[0,1],[1,2],[2,3],[1,0],[2,1],[3,2],[1,4],[4,1],[0,1],[4,4]])
    track=np.array([[.0002,0.,0.],[.0008,0.,1.],[.0014,0.,2.],[.0024,0.,3.]])
    return Trip('synthetic',track,nodes,arcs,np.ones(len(track),dtype=bool))

@pytest.fixture
def tiny_net(tiny_trip):return Network(tiny_trip)

def candidates(ids,fractions=None):
    return Candidates(np.array(ids,dtype=np.int64),np.zeros(len(ids)),np.array(fractions if fractions is not None else [.5]*len(ids)),50)
