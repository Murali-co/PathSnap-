import numpy as np
from pathsnap.evaluation.metrics import observation_top1,sequence_metrics

def test_candidate_miss_is_not_wrong_winner():
    r=observation_top1(np.array([1,2]),np.array([.9,.1]),np.array([0,2]),np.array([2]))
    assert r['observation_top1']==0 and r['candidate_miss_count']==0

def test_missing_candidates_count_and_unknown_coverage():
    r=observation_top1(np.array([1]),np.array([.9]),np.array([0,1,1,1]),np.array([2,3,-1]))
    assert r['candidate_miss_count']==2 and r['label_coverage']==2/3
    assert r['known_observations']==2 and r['conditional_observation_loss'] is None

def test_ordered_quality_preserves_repeated_visits():
    assert sequence_metrics([1,2,1],[1,2,1])['normalized_edit_distance']==0
    assert sequence_metrics([1,2],[1,2,1])['normalized_edit_distance']==1/3
