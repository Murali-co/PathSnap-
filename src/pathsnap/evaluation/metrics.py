"""Keep candidate errors, missing references and candidate misses separate."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix


def candidate_metrics(y_true, y_pred, prob):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if not len(y_true):
        return dict(candidate_accuracy=None, candidate_precision=None, candidate_recall=None,
                    candidate_macro_f1=None, candidate_binary_log_loss=None, confusion_matrix=[[0,0],[0,0]])
    return dict(candidate_accuracy=float(accuracy_score(y_true,y_pred)),
                candidate_precision=float(precision_score(y_true,y_pred,zero_division=0)),
                candidate_recall=float(recall_score(y_true,y_pred,zero_division=0)),
                candidate_macro_f1=float(f1_score(y_true,y_pred,labels=[0,1],average='macro',zero_division=0)),
                candidate_binary_log_loss=float(log_loss(y_true,np.clip(prob,1e-7,1-1e-7),labels=[0,1])),
                confusion_matrix=confusion_matrix(y_true,y_pred,labels=[0,1]).tolist())


def top1_states(ids, probabilities, offsets):
    states = np.full(len(offsets)-1, -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        if hi > lo:
            states[i] = ids[lo+int(np.argmax(probabilities[lo:hi]))]
    return states


def observation_top1(group_ids, group_probs, group_offsets, aligned_labels):
    labels = np.asarray(aligned_labels)
    known = labels >= 0
    states = top1_states(group_ids, group_probs, group_offsets)
    hits, losses = np.zeros(len(labels),dtype=bool), []
    for i, label in enumerate(labels):
        lo, hi = group_offsets[i:i+2]
        if label >= 0 and hi > lo:
            matches = group_ids[lo:hi] == label
            hits[i] = matches.any()
            if hits[i]:
                p = np.clip(group_probs[lo:hi], 1e-7, 1-1e-7)
                losses.append(-np.log(p[matches].sum()/p.sum()))
    misses = int((known & ~hits).sum())
    return dict(observation_top1=float((states[known]==labels[known]).mean()) if known.any() else None,
                correct_observations=int(((states==labels)&known).sum()), known_observations=int(known.sum()),
                observations=len(labels), candidate_miss_count=misses, candidate_misses=misses,
                candidate_miss_rate=float(misses/known.sum()) if known.any() else None,
                label_coverage=float(known.mean()) if len(labels) else 0,
                unmatched_observations=int((states<0).sum()),
                conditional_observation_loss=float(np.mean(losses)) if losses else None)


def collapse(sequence):
    return [int(a) for i,a in enumerate(sequence) if a >= 0 and (i==0 or a != sequence[i-1])]


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row = [i]
        for j,y in enumerate(b,1):
            row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous = row
    return previous[-1]


def sequence_metrics(predicted, reference):
    p,r=set(predicted),set(reference)
    common=len(p&r)
    precision=common/len(p) if p else 0.
    recall=common/len(r) if r else 0.
    return dict(route_precision=precision,route_recall=recall,
                route_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                normalized_edit_distance=edit_distance(predicted,reference)/max(len(predicted),len(reference),1))


def continuity(net, groups, states, track, cfg):
    legal, eligible, gap_pairs, unmatched = 0,0,0,0
    for i in range(1,len(states)):
        dt=track[i,2]-track[i-1,2]
        if not np.isfinite(dt) or dt<=0 or dt>cfg['gap_seconds']:
            gap_pairs+=1; continue
        if states[i-1]<0 or states[i]<0:
            unmatched+=1; continue
        eligible+=1
        a,b=int(states[i-1]),int(states[i])
        ia=np.flatnonzero(groups[i-1].ids==a)[0]; ib=np.flatnonzero(groups[i].ids==b)[0]
        distance,_=net.transition(a,groups[i-1].fraction[ia],b,groups[i].fraction[ib],dt,
                                  np.linalg.norm(net.obs[i]-net.obs[i-1]),cfg)
        legal+=int(np.isfinite(distance))
    return dict(route_continuity=legal/eligible if eligible else None,legal_pairs=legal,
                eligible_pairs=eligible,gap_pairs=gap_pairs,unmatched_pairs=unmatched)
