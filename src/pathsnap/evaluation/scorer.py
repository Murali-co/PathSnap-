"""Private post-prediction evaluation. Only aggregate values leave this module."""
import numpy as np
from ..reference import align,load_reference
from .metrics import observation_top1,candidate_metrics,sequence_metrics,collapse,continuity


def score_prediction(context, reference_path, cfg):
    trip,net=context['trip'],context['net']
    route=load_reference(reference_path,len(trip.arcs))
    aligned=align(trip,net,route,cfg)
    labels=aligned['labels'];known=labels>=0
    row_known=known[np.repeat(np.arange(len(labels)),np.diff(context['offsets']))]
    row_labels=np.repeat(labels,np.diff(context['offsets']))
    targets=(context['ids']==row_labels).astype(int)
    probabilities=context['probabilities']
    metrics=observation_top1(context['ids'],probabilities,context['offsets'],labels)
    metrics.update(candidate_metrics(targets[row_known],probabilities[row_known]>=.5,probabilities[row_known]))
    raw=dict(metrics)
    raw.update(sequence_metrics([a for c in context['raw_components'] for a in c],collapse(route)))
    raw.update(continuity(net,context['groups'],context['raw'],trip.track,cfg))
    decoded=dict(observation_top1(context['ids'],probabilities,context['offsets'],labels))
    states=context['decoded']['states']
    decoded.update(observation_top1=float((states[known]==labels[known]).mean()) if known.any() else None,
                   correct_observations=int(((states==labels)&known).sum()),unmatched_observations=int((states<0).sum()))
    decoded.update(sequence_metrics([a for c in context['components'] for a in c],collapse(route)))
    decoded.update(continuity(net,context['groups'],states,trip.track,cfg))
    return dict(evaluation_status='derived_reference',label_method=cfg['label_method'],
                metrics=dict(classifier=raw,decoded=decoded,label_ambiguity_fraction=float(aligned['ambiguous'].mean()),
                             reference_breaks=int(aligned.get('reference_breaks',0))))
