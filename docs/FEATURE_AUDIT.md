# Feature and pipeline audit

All 22 features are available from track/network alone, in the exact order in `features.SCHEMA`. No IDs, route membership, aligned progress, absolute location or reference distances enter model inputs. Full-trip offline mode allows the next GPS observation; this is not a causal live tracker.

| Feature | Units / treatment |
|---|---|
| distance_m | Metric point-to-segment distance, zero lengths handled |
| distance_rank | Zero-based rank in retained directed candidates; deterministic distance/ID ties |
| distance_delta_m | Difference from nearest retained distance |
| fraction | Clipped projection fraction in [0,1] |
| endpoint_distance_m | Along-segment distance to nearer endpoint |
| length_m | Directed arc length under local projection |
| heading_cosine | Direction cosine multiplied by heading reliability |
| signed_motion_m | Signed displacement along candidate direction; zero if direction undefined |
| heading_reliability | Movement magnitude factor times 30-second straightness |
| previous_displacement_m | Zero across invalid/large-gap pairs |
| next_displacement_m | Offline next valid displacement, zero across gaps |
| speed_mps | Previous displacement / max(previous interval,1); retained as noisy GPS speed |
| previous_dt_s | Original previous timestamp interval; zero at first point |
| next_dt_s | Next interval, zero at last point |
| gap_flag | Missing previous reliable interval or next large gap |
| stationary | Previous displacement below 2 m; heuristic, not proof of parking |
| straightness_30s | Net displacement / path length inside a timestamp-based 30-second window and valid block |
| start_in_degree | Original start-node incoming arc count, duplicates preserved |
| end_out_degree | Original end-node outgoing arc count |
| reverse_available | Reverse endpoint pair present |
| candidate_count | Number of final retained directed arcs |
| radius_m | Radius at which candidates first became available |

Nonfinite feature values are deterministically replaced; scalers/imputers learn only from fold-training trips. Invalid observations remain output rows. Reference labels with residual >300 m are unknown, excluded from fitting and retained in coverage reporting. Ambiguous derived labels remain disclosed rather than silently presented as certified truth.

Regression coverage: known metric distance and coordinate order; long segments; opposite directions and duplicate IDs; zero length; exact routing cutoffs/cache isolation; irregular timestamps and gaps; repeat route occurrences; candidate misses versus wrong winner; empty/all-impossible decoder steps; finite probabilities; artifact feature order/checksum; CPU round-trip for all five model families; reference-free inference; private scoring and API redaction; repeated requests and failure recovery.

Current candidate policy: first nonempty radius in 50/100/200/300 m; top 16 directed arcs by distance then ID. It preserves original directions/IDs but cannot guarantee retention of every reverse twin at the top-K boundary. Final candidate recall is measured on allowed data. Do not promise an improvement from a larger radius without measuring retained recall.
