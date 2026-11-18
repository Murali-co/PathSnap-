Technical Project Report · Mentor Submission

PES University · UE24CS352A: Machine Learning · 18 November 2026

| Team 1 member | SRN |

| --- | --- |

| Murali R | PES2UG25AM805 |

| Ranjith G | PES2UG25AM806 |

| Kushal J | PES2UG24AM205 |

| Yashas V | PES2UG24AM188 |

Team details are taken from the supplied official hackathon instructions.

| Training trips | Model families | Development trips | Decoded mean trip Top-1 |

| --- | --- | --- | --- |

| 80 | 5 | 10 | 91.01% |

Evaluation status: recorded development experiments and locally verified applications. Physical two-laptop demonstration, final organiser scoring and a completed new tuning/export comparison remain pending. Reported accuracy uses provisional derived reference labels.

Mentor / evaluator: ______________________________

Source repository: <link href="https://github.com/Murali-co/PathSnap-" color="#007d87">Murali-co/PathSnap-</link><br/>Frontend integration branch: <link href="https://github.com/Murali-co/PathSnap-/tree/feature/frontend-web" color="#007d87">feature/frontend-web</link>


# Contents



# 1  Abstract and scope


PathSnap reconstructs a plausible directed road journey from noisy GPS observations using supervised candidate scoring and sequential decoding. Each trip supplies a trajectory, a local road graph and an ordered reference route. A trip-local metric projection and Shapely segment index generate candidate arcs; 22 geometry, motion and topology features feed five distinct classifiers. The selected classifier supplies normalized emission potentials to an HMM-style log-space Viterbi decoder with directed routing and explicit breaks.

The permitted development data contains 80 training trips and 10 student-test trips, totaling 227,577 GPS observations. Full-fit records show 1,609,263 resolved training candidate rows. XGBoost leads the recorded grouped OOF comparison: mean trip observation Top-1 is 83.20%, with pooled candidate macro-F1 0.8996. Student-test classifier mean trip Top-1 is 87.49%. Directed decoding raises it to 91.01%, a 3.52 percentage-point change. Mean per-trip continuity rises from 95.49% to 99.95%.

FastAPI and separate Python/Streamlit and React/Vite clients implement the demonstration workflow. All ten development trips and repeated requests passed actual HTTP verification; 30 focused regression tests passed. A saved older tuning database records ten completed trials and one unresolved RUNNING entry. A finished 15-trial study, a final tuned-versus-untuned student-test gain, physical LAN execution and official reserved-data scores are not claimed.

How to read the results: this is a complete account of the available project and evidence, not a claim that every external acceptance condition is satisfied. Candidate classification metrics, observation matching, route overlap and ordered sequence quality are kept distinct. The reference alignment is provisional; its uncertainty limits all pointwise accuracy claims.


# 2  Problem statement and objectives


GPS noise, missing samples and nearby parallel roads can place observations away from the road actually traversed. Selecting the nearest segment independently can confuse opposite directions or create disconnected routes. PathSnap separates the problem into local road plausibility and sequence feasibility: a classifier scores nearby directed arcs, then the decoder chooses a feasible chronological sequence.

| Objective | Implemented response |

| --- | --- |

| Understand inputs | Audit all 90 permitted trips, preserve row identities and validate file schemas. |

| Learn road plausibility | Fit five classifier families to candidate-level binary labels with grouped validation. |

| Reconstruct journeys | Use directed lengths, projected positions, shortest paths and Viterbi backtracking. |

| Demonstrate a distributed application | Laptop 1 hosts model/data/API/private scoring; Laptop 2 hosts a frontend. |

| Support defensible evaluation | Export real experiment records, coverage/misses, route metrics, timings and a reproducible report. |

No external road download, paid routing service, LLM prediction endpoint, cloud GPU or hosted database is required. “Distributed” refers to the two-laptop client/server demonstration. The implementation processes complete trips offline; neighbouring future GPS observations are permitted input features and do not make this a causal live-tracking system.


# 3  System architecture and pipeline


The diagrams show implementation stages, not a simulated experiment. The reference path enters only training labels and post-prediction private scoring; it is absent from candidate generation, features, classifier inference and decoding.

| Component | Responsibility / entry point |

| --- | --- |

| Data and geometry | data.py, spatial.py: validated original IDs, local projection, segment lookup and directed adjacency. |

| Features and supervision | features.py and reference.py: fixed 22-feature schema and replaceable ordered-occurrence labels. |

| Experiments | training.py, experiments.py: grouped CV, full fits, model comparison, selection and portable export. |

| Sequence inference | decoding/viterbi.py and inference.py: emissions, directed transitions, independent blocks and intermediate road reconstruction. |

| Applications | backend/app/main.py; frontend/app/app.py; frontend-web/: jobs, map display and repeated trip interaction. |

| Repeatable workflow | scripts/run_full_pipeline.py: bounded cycles, resumable training and a persisted stage ledger. |

The cycle is audit → prepare → train five families → compare OOF/student-test results → tune the selected family → export → regression tests → all-ten HTTP verification → PDF → local packages. A failed stage stops execution before stale results are exported. `--resume` reuses compatible completed fits; `--cycles` bounds repeated execution. The verified saved-model cycle deliberately skipped real-data fitting and tuning; backend requests never trigger training.


# 4  Dataset understanding and EDA


| Partition | Trips | GPS rows | Node rows | Directed arc rows | Route entries |

| --- | --- | --- | --- | --- | --- |

| Training | 80 | 208,417 | 1,933,485 | 3,876,904 | 31,336 |

| Student test | 10 | 19,160 | 90,689 | 185,041 | 3,477 |

| Permitted total | 90 | 227,577 | 2,024,174 | 4,061,945 | 34,813 |

| Reserved evaluation | 10 expected | Not analysed | Not analysed | Not analysed | Not analysed |

Counts above sum trip-local graphs; they are not counts of globally unique roads. The supplied development package has 360 raw files. Each original row retains its identity: `.track` is longitude/latitude/timestamp, `.nodes` is longitude/latitude, `.arcs` is an ordered directed endpoint pair, and `.route` is a chronological arc sequence. Arc IDs are local to a trip, so a global multiclass arc-ID target would be invalid.


![Figure 1: Development trip coordinates and per-trip graph/trajectory scale. Coordinates are shown for EDA and excluded from learning features.](reports/assets/eda.png)



## 4.1  Quality findings and their treatment


| Measured finding | Engineering consequence |

| --- | --- |

| 0 invalid GPS rows in the audited inputs | Validation remains active; future invalid observations retain their output positions. |

| 96.68% of intervals are exactly 1 s | Time-based windows are used; row counts are not treated as elapsed seconds. |

| Maximum timestamp gap: 153,064 s | Split decoding blocks; do not infer continuous travel or a proven pause. |

| 106 zero-length arcs; 1,903 duplicate directed endpoint pairs | Handle projection safely; preserve original arc IDs and edge-aware adjacency. |

| 14 adjacent reference-route discontinuities | Retain diagnostics; do not assume reference routes are error-free. |

| Weak components vary from 1 to 421 per trip | Unreachable transitions are valid cases, not a reason to fabricate connecting geometry. |

| Maximum sampled relative projection discrepancy: 0.00000653 | Geodesic comparison samples 100 or fewer arcs per trip; this is not a global distortion bound. |


![Figure 2: Per-trip gap extremes and final retained-candidate recall. All inputs are permitted development data.](reports/assets/data_quality.png)



# 5  Preprocessing, candidates and reference labels



## 5.1  Metric geometry and directed candidates


Coordinates are interpreted as longitude/latitude under an explicit WGS84 assumption; the file schema does not independently establish the datum. Each trip receives a local azimuthal-equidistant projection. Segment lengths and projection distances are computed in metres. An STRtree indexes whole line-segment bounding boxes, including long arcs whose midpoints may be far from an observation. Opposite directions and duplicate geometries retain their original IDs.

The current prediction-only policy searches radii 50, 100, 200 and 300 m and stops at the first nonempty radius. It retains at most 16 directed arcs, sorted by distance then original ID. Reference labels never insert candidates. Expansion does not guarantee that the correct offset road is retrieved when a closer false road already exists; this is a measured limitation rather than an assumed solved problem.

| Partition | GPS rows | Known labels | Unknown | Candidate rows | Final recall | Ambiguity flag |

| --- | --- | --- | --- | --- | --- | --- |

| train | 208,417 | 208,002 | 415 | 1,612,959 | 89.91% | 22.09% |

| student_test | 19,160 | 19,160 | 0 | 169,191 | 99.83% | 25.18% |

Training caches contain 1,612,959 candidate rows before unknown-label filtering and 1,609,263 fitting rows afterwards. There are 187,005 positive and 1,422,258 negative fitting rows. Student testing has 169,191 candidate rows, 19,128 positives and 32 known-reference candidate misses. Neither partition has an observation with an empty final candidate set in this saved preparation. The training candidate recall ceiling is lower than the student-test ceiling; these splits contain different trip conditions.


## 5.2  Provisional supervision


A route is an ordered segment sequence, not one label per GPS row. The aligner uses route-occurrence states, so repeated visits to an arc remain distinct. Point-to-segment residual costs, monotonic occurrence progress, bounded backwards projection costs and skip penalties align observations within valid GPS blocks. Invalid observations and long temporal gaps reset alignment. Reference discontinuities remain diagnostics and affect progress costs.

Aligned residuals above 300 m become unknown labels, producing 415 excluded training observations. Near-equal alternative road distances generate ambiguity flags. These labels are derived references rather than certified physical truth; the organiser’s pointwise convention remains unresolved. Accuracy denominators include known labels only, while candidate misses and unmatched predictions count as incorrect on those known labels. Coverage and ambiguity are reported separately.


# 6  Feature engineering


All 22 features are computed identically at training and prediction from public track/network inputs. The stored schema fixes their order. IDs remain metadata; trip ID, arc/node ID, route membership, aligned progress, reference distances and absolute longitude/latitude are excluded. Learned scaling/imputation is fitted within training folds only. Nonfinite feature values receive deterministic finite replacements, while invalid input rows retain explicit statuses.

| Group | Feature | Definition and reason |

| --- | --- | --- |

| Geometry | distance_m | Perpendicular segment distance (m). |

| Geometry | distance_rank | Zero-based deterministic rank in final candidates. |

| Geometry | distance_delta_m | Distance above the nearest retained candidate (m). |

| Geometry | fraction | Clipped projection fraction along directed arc [0,1]. |

| Geometry | endpoint_distance_m | Along-segment distance to nearer endpoint (m). |

| Geometry | length_m | Projected directed arc length (m). |

| Direction | heading_cosine | Candidate/motion direction agreement × heading reliability. |

| Direction | signed_motion_m | Signed motion projected along candidate direction (m). |

| Direction | heading_reliability | Movement magnitude factor × 30-second straightness. |

| Motion/time | previous_displacement_m | Previous valid displacement; zero across invalid/large gaps. |

| Motion/time | next_displacement_m | Next valid displacement; offline feature. |

| Motion/time | speed_mps | Previous displacement / max(previous interval, 1). |

| Motion/time | previous_dt_s | Previous timestamp interval; zero at first observation. |

| Motion/time | next_dt_s | Next timestamp interval; zero at final observation. |

| Motion/time | gap_flag | No reliable prior interval or a subsequent large gap. |

| Motion/time | stationary | Previous displacement <2 m: a heuristic, not proof of parking. |

| Motion/time | straightness_30s | Net/path displacement within a time-based window and valid block. |

| Topology | start_in_degree | Incoming arc count at the candidate start node. |

| Topology | end_out_degree | Outgoing arc count at the candidate end node. |

| Topology | reverse_available | Whether the reverse endpoint pair exists. |

| Candidates | candidate_count | Final directed candidate count for the observation. |

| Candidates | radius_m | First radius with candidates (m). |

Heading confidence is reduced for low movement or low net displacement. GPS speed remains a noisy measurement, not verified vehicle speed. Timestamp windows do not cross declared large gaps. Feature groups provide road proximity, direction, motion reliability and local connectivity without geographic memorisation. No feature ablation or calibration gain is claimed: those experiments are not in the saved evidence.


# 7  Experimental design and metric definitions


Training uses 80 trip groups with fixed seed 42 and three GroupKFold splits. Candidate rows belonging to one trip remain together. A final model is refitted on all resolved training rows and then evaluated on the ten student-test trips. Fixed iteration/epoch budgets avoid using outer validation folds or student-test data for early stopping. These are development estimates used for model selection, not final reserved-test estimates [R2].

| Metric | Operational definition |

| --- | --- |

| Candidate accuracy / precision / recall | Binary target per retained candidate; threshold 0.5. Positive class means the candidate equals the aligned reference arc. |

| Candidate macro-F1 | Mean F1 of negative and positive classes. The consolidated CV value pools OOF candidate predictions; it is not the unweighted mean of fold F1 values. |

| Candidate binary log loss | Binary cross-entropy on clipped positive probabilities; affected by class imbalance and probability quality. |

| Mean trip observation Top-1 | Compute correct directed-arc fraction on known labels in each trip, then average trip fractions equally. |

| Pooled observation Top-1 | Total correct observations / total known observations across trips. Different denominator from mean trip Top-1. |

| Candidate miss rate | Known reference arc absent from final candidates, including nonempty wrong candidate sets. |

| Conditional observation loss | Within-observation normalized positive probability loss only where reference candidates exist; report alongside misses. |

| Directed continuity | Feasible directed travel under graph and movement/time bounds; report eligible, legal, gap and unmatched pair counts. |

| Route precision / recall / F1 | Overlap of arc-ID sets within each trip. Repeated visits do not increase set counts. |

| Normalized edit distance | Edit distance of ordered consecutive-duplicate-collapsed arc sequences / max(sequence lengths). Lower is better; repeated visits remain. |

Selection ranks models by training OOF mean-per-trip Top-1. Candidate macro-F1, conditional probability loss, trip-level variation and fit cost are tie-breakers. Student-test comparison is permitted development feedback; its repeated use must not be described as a pristine final holdout. Explicit saved fold trip lists are available for the newer XGBoost and MLP fits; older baseline provenance relies on their code, CV tables, OOF arrays and serialized estimators.


# 8  Model experimentation: five distinct families



## 8.1  Logistic regression — Linear and probabilistic baseline


Why considered. Suitable for a compact relative-feature representation and a simple interpretable reference. Fold-local scaling and median imputation make mixed-unit features usable.

Strengths and limitations. Convex logistic-loss fitting is efficient. Its linear log-odds surface cannot directly express all geometry/motion interactions, and balanced class weights alter the interpretation of raw positive probabilities.

Objective. Weighted logistic loss with L2 regularisation.

| Recorded configuration | Value |

| --- | --- |

| C | 1.0 |

| max_iter | 500 |

| solver | lbfgs |

| class_weight | balanced |

| n_jobs | None |

| OOF mean trip Top-1 | OOF macro-F1 | Student Top-1 | Student macro-F1 | Binary log loss |

| --- | --- | --- | --- | --- |

| 82.05% | 0.7830 | 84.39% | 0.8197 | 0.1986 |

Measured cost. Final full fit: 6.274 s; prediction over all ten cached student-test candidate matrices: 0.0534 s. These timings exclude candidate construction, decoding and HTTP overhead. They are not end-to-end request durations.

Interpretation / over- and underfitting. The lowest candidate macro-F1 among the five shows that this feature representation benefits from nonlinear scoring. That is not conclusive proof of underfitting: saved training-set accuracy or learning curves for this fit are unavailable.


## 8.2  Decision tree — Single nonlinear tree


Why considered. A regularised tree can learn threshold interactions between distance, heading reliability and topology without scaling.

Strengths and limitations. Readable nonlinear splits and fast prediction are useful. One tree has higher variance than an ensemble, and probability estimates can be coarse in small leaves.

Objective. Weighted Gini-impurity reduction; bounded depth and minimum leaf size.

| Recorded configuration | Value |

| --- | --- |

| class_weight | balanced |

| max_depth | 12 |

| min_samples_leaf | 20 |

| max_features | None |

| criterion | gini |

| OOF mean trip Top-1 | OOF macro-F1 | Student Top-1 | Student macro-F1 | Binary log loss |

| --- | --- | --- | --- | --- |

| 82.09% | 0.8332 | 84.11% | 0.8679 | 0.1553 |

Measured cost. Final full fit: 25.984 s; prediction over all ten cached student-test candidate matrices: 0.0082 s. These timings exclude candidate construction, decoding and HTTP overhead. They are not end-to-end request durations.

Interpretation / over- and underfitting. It improves candidate macro-F1 over the linear baseline, but mean trip Top-1 is slightly lower on student testing. This demonstrates that binary threshold quality and within-observation ranking are different objectives. No artificial epoch loss curve is shown.


## 8.3  Random forest — Bagged tree ensemble


Why considered. A forest averages multiple nonlinear trees and can reduce the instability of one decision tree.

Strengths and limitations. It captures interactions and provides strong observation ranking. Costs include a larger model and slower fit/inference; class weighting can shift threshold behaviour.

Objective. Averaged class probabilities from regularised Gini-split trees.

| Recorded configuration | Value |

| --- | --- |

| class_weight | balanced |

| max_depth | 16 |

| min_samples_leaf | 5 |

| n_estimators | 200 |

| max_features | sqrt |

| criterion | gini |

| n_jobs | -1 |

| OOF mean trip Top-1 | OOF macro-F1 | Student Top-1 | Student macro-F1 | Binary log loss |

| --- | --- | --- | --- | --- |

| 82.76% | 0.8748 | 86.71% | 0.8898 | 0.1257 |

Measured cost. Final full fit: 133.192 s; prediction over all ten cached student-test candidate matrices: 0.5884 s. These timings exclude candidate construction, decoding and HTTP overhead. They are not end-to-end request durations.

Interpretation / over- and underfitting. It ranks second by training mean-trip Top-1 and student-test Top-1. Its positive recall is high but precision is lower than XGBoost. Its long recorded fit duration makes the small ranking gap relevant to model choice.


## 8.4  XGBoost — Gradient-boosted trees


Why considered. Histogram boosting is appropriate for tabular geometry/motion features, nonlinear interactions and probability scoring.

Strengths and limitations. It balances ranking quality, candidate macro-F1 and log loss in this comparison. Search complexity, depth-related overfitting and uncalibrated scores remain limitations.

Objective. Binary logistic objective with log-loss evaluation; histogram tree construction.

| Recorded configuration | Value |

| --- | --- |

| max_depth | 6 |

| n_estimators | 300 |

| learning_rate | 0.05 |

| subsample | 0.8 |

| colsample_bytree | 0.8 |

| objective | binary:logistic |

| eval_metric | logloss |

| tree_method | hist |

| device | cuda |

| n_jobs | 2 |

| OOF mean trip Top-1 | OOF macro-F1 | Student Top-1 | Student macro-F1 | Binary log loss |

| --- | --- | --- | --- | --- |

| 83.20% | 0.8996 | 87.49% | 0.9159 | 0.0805 |

Measured cost. Final full fit: 3.679 s; prediction over all ten cached student-test candidate matrices: 0.2633 s. These timings exclude candidate construction, decoding and HTTP overhead. They are not end-to-end request durations.

Interpretation / over- and underfitting. It leads both training OOF mean-trip Top-1 and student-test mean-trip Top-1 and has the lowest student-test candidate log loss. It is therefore selected for decoding and service export. CUDA is recorded in the newer training metadata; inference is explicitly portable to CPU.


## 8.5  PyTorch MLP — Two-hidden-layer neural classifier


Why considered. A modest MLP learns nonlinear combinations of relative features without learning trip or absolute road IDs.

Strengths and limitations. Batched optimization supports GPU fitting and a compact network. Sensitivity to scaling, optimization and calibration is a trade-off. The saved CPU-portable estimator must retain its learned mean and standard deviation.

Objective. Binary cross-entropy with logits, Adam, fixed 20 epochs and weight decay.

| Recorded configuration | Value |

| --- | --- |

| hidden1 | 64 |

| hidden2 | 64 |

| epochs | 20 |

| batch_size | 4096 |

| lr | 0.001 |

| weight_decay | 0.0001 |

| dropout | 0.1 |

| device | cpu |

| OOF mean trip Top-1 | OOF macro-F1 | Student Top-1 | Student macro-F1 | Binary log loss |

| --- | --- | --- | --- | --- |

| 82.51% | 0.8948 | 85.50% | 0.9053 | 0.0881 |

Measured cost. Final full fit: 12.752 s; prediction over all ten cached student-test candidate matrices: 0.0261 s. These timings exclude candidate construction, decoding and HTTP overhead. They are not end-to-end request durations.

Interpretation / over- and underfitting. The recorded training loss decreases over 20 epochs. Candidate macro-F1 is competitive but student-test Top-1 is below the forest and XGBoost. No epoch-wise validation loss exists, so training-loss decrease alone does not establish generalization or the absence of overfitting.


![Figure 3: Actual fold/full-fit training losses. No validation curve is invented.](reports/assets/mlp.png)



# 9  Consolidated model comparison


| Model | OOF trip Top-1 | OOF macro-F1 | Test trip Top-1 | Test macro-F1 | Test log loss |

| --- | --- | --- | --- | --- | --- |

| XGBoost | 83.20% | 0.8996 | 87.49% | 0.9159 | 0.0805 |

| Random forest | 82.76% | 0.8748 | 86.71% | 0.8898 | 0.1257 |

| PyTorch MLP | 82.51% | 0.8948 | 85.50% | 0.9053 | 0.0881 |

| Decision tree | 82.09% | 0.8332 | 84.11% | 0.8679 | 0.1553 |

| Logistic regression | 82.05% | 0.7830 | 84.39% | 0.8197 | 0.1986 |

All five fits use 1,609,263 resolved training candidate rows from 80 trips. Student-test binary metrics use 169,191 candidate rows across all ten trips. The positive fraction is about 11.3%, so candidate accuracy alone is not an adequate selection metric. Test label coverage is 100%; the retained-candidate miss rate is 0.17%.

| Model | Candidate accuracy | Positive precision | Positive recall | Fit seconds | Classifier continuity |

| --- | --- | --- | --- | --- | --- |

| XGBoost | 96.73% | 0.8810 | 0.8214 | 3.679 | 95.49% |

| Random forest | 94.94% | 0.7061 | 0.9465 | 133.192 | N/A: not run |

| PyTorch MLP | 96.32% | 0.8617 | 0.8031 | 12.752 | N/A: not run |

| Decision tree | 93.66% | 0.6496 | 0.9534 | 25.984 | N/A: not run |

| Logistic regression | 90.55% | 0.5472 | 0.9532 | 6.274 | N/A: not run |

Continuity was executed for the selected XGBoost pipeline. It was not computed for the other four families; N/A values are unexecuted checks rather than zeros. Classifier inference durations exclude graph/features/decoding and should not be compared directly with later full-request timings.


![Figure 4: Actual OOF and student-test scores for all five models.](reports/assets/models.png)



![Figure 5: Recorded probability loss and full-fit durations. Hardware/execution paths differ by family.](reports/assets/loss_runtime.png)



![Figure 6: Actual student-test candidate confusion matrices; rows are reference classes, columns predictions.](reports/assets/confusion.png)



# 10  Bayesian hyperparameter optimisation


Tuning is useful for controlling the bias/variance trade-off of a selected model, but a gain must be measured rather than promised [R3]. The current generic workflow uses Optuna TPESampler, a fixed seed, persistent SQLite studies and trip-disjoint folds. It can tune whichever of the five families wins. Fifteen total trials and five startup trials are configured; failed/pruned trials remain visible [R4].

| Winning family | Implemented bounded search space |

| --- | --- |

| Logistic regression | C: 0.001–100 (log); solver: lbfgs / liblinear. |

| Decision tree | Depth: 4–20; minimum leaf: 5–100; criterion: Gini / entropy. |

| Random forest | Trees: 60–240, step 30; depth: 6–20; leaf: 3–40; feature fraction: 0.4–1.0. |

| XGBoost | Trees: 100–500; depth: 3–10; rate: 0.01–0.20 (log); child weight: 0.5–10 (log); row/column sampling: 0.6–1.0; alpha: 1e-8–10; lambda: 1e-5–100. |

| MLP | Widths: 32/64/128 and 32/64; learning rate: 1e-4–1e-2; weight decay: 1e-6–1e-2; dropout: 0–0.3; batch: 1024/2048/4096. |


## 10.1  Actual saved search record


The older study pathsnap_xgboost_tpe contains 10 completed trials and 1 entry marked RUNNING. Its recorded objective is mean-fold candidate macro-F1, different from the new pipeline’s primary mean-trip Top-1 objective. Best observed trial: 3, value 0.901578. The RUNNING marker is a database state, not evidence that a training process remains active.

| Best observed parameter | Recorded value |

| --- | --- |

| colsample_bytree | 0.6185801650879991 |

| learning_rate | 0.018187859051288217 |

| max_depth | 9 |

| min_child_weight | 2.333481883618462 |

| n_estimators | 282 |

| reg_alpha | 0.0029369981104377003 |

| reg_lambda | 0.00015619562520792737 |

| subsample | 0.836965827544817 |


![Figure 7: Partial legacy study extracted read-only from the saved SQLite database. The best recorded trial is not a deployed tuned model.](reports/assets/tuning.png)


The legacy startup configuration is not independently stored with each trial, and its study was not completed/exported. This report therefore does not claim a completed Bayesian tuning demonstration or a tuned student-test improvement from this record. The current selected classifier remains the untuned comparison winner. A final tuned-versus-untuned table, total completed-study cost and convergence claim require the user-run generic workflow to finish.


# 11  HMM/Viterbi and route reconstruction



## 11.1  States, emission potentials and recurrence


At each observation a state is an original directed candidate arc plus its projected fraction. Classifier positive probabilities are clipped and normalized within the candidate set. This creates discriminative emission potentials; it does not automatically produce calibrated generative observation likelihoods. The design follows the established HMM map-matching separation of local uncertainty and road-network feasibility [R1].


```text
e_t(j) = clip(p_t(j)) / sum_k clip(p_t(k))
V_t(j) = log e_t(j) + max_i [ V_(t-1)(i) + T_t(i,j) ]
T_t(i,j) = -abs(d_road(i,j) - d_GPS(t-1,t)) / beta
              - U_turn_penalty(i,j)
Unreachable or motion-infeasible transition: -infinity
```


Directed travel uses same-arc forward progress, source remainder plus destination prefix for adjacent arcs, or source remainder + directed shortest-path length + destination prefix for multi-hop travel. Small backwards projection noise is tolerated within 12 m; arbitrary wrong-way travel on a directed arc is not allowed. Graph-feasible reverse-road changes can occur with a modest U-turn penalty. Legal turn restrictions were not supplied.

| Parameter / implementation | Current value or rule |

| --- | --- |

| Emission convention | Clipped positive probability / within-observation sum. |

| Transition scale beta | 35 m. |

| Large-gap break | More than 60 seconds. |

| Movement proxy bound | 55 m/s × interval + 50 m slack; a feasibility proxy, not a supplied speed limit. |

| Backward projection tolerance | 12 m on a directed arc. |

| U-turn penalty | 1.0 when switching into the reverse endpoint pair. |

| Routing cache | Per-network bounded source search; rounded-up bucket plus exact caller cutoff; at most 2,000 source/bound entries. |

| Block handling | Invalid/empty candidates or all-impossible transitions finish/backtrack the prior block and restart explicitly. |


## 11.2  Reconstructed geometry


Backtracking retains one output/status per original observation. Selected arcs and predecessor-mapped intermediate arcs form actual directed route components. Gaps and infeasible transitions remain separate components; no straight line is drawn across disconnected blocks. Interpolated roads are predicted paths through unobserved travel, not proof of the exact journey. Duplicate arcs never have their lengths accidentally summed by sparse adjacency construction.


# 12  Decoder results and failure analysis


| Metric | Classifier only | Classifier + Viterbi |

| --- | --- | --- |

| Mean trip Top-1 | 87.49% | 91.01% |

| Pooled Top-1 | 86.27% | 90.01% |

| Mean trip directed continuity | 95.49% | 99.95% |

| Pooled continuity | 94.72% | 99.94% |

| Mean trip arc-set route F1 | 0.8619 | 0.9410 |

| Mean normalized edit distance ↓ | 0.4197 | 0.1859 |

| Legal / eligible observation pairs | 18,137 / 19,149 | 19,138 / 19,149 |

There is one temporal-gap pair excluded from these continuity denominators. Restarts are exposed separately; continuity is not inflated by simply omitting every restarted eligible pair. Mean-of-trip scores and pooled scores differ because trip lengths vary.

| Student-test trip | Classifier Top-1 | Decoded Top-1 | Change (points) | Route blocks |

| --- | --- | --- | --- | --- |

| 00000000 | 93.77% | 95.57% | +1.80 | 1 |

| 00000007 | 97.71% | 92.54% | -5.17 | 4 |

| 00000009 | 81.98% | 86.37% | +4.39 | 1 |

| 00000040 | 93.92% | 96.06% | +2.14 | 2 |

| 00000052 | 93.51% | 96.56% | +3.05 | 1 |

| 00000056 | 85.88% | 92.65% | +6.77 | 3 |

| 00000065 | 85.03% | 90.89% | +5.86 | 2 |

| 00000076 | 85.73% | 90.31% | +4.58 | 1 |

| 00000077 | 73.74% | 78.32% | +4.58 | 5 |

| 00000091 | 83.64% | 90.85% | +7.21 | 2 |


![Figure 8: Same classifier/candidates with and without decoding, and measured API durations. Trip suffixes identify all ten cases.](reports/assets/decoder.png)


Observed failure: trip 00000007 decreases from 97.71% to 92.54%, a -5.17-point change. This demonstrates that graph coherence can override a locally correct classifier choice. Possible causes include transition scale, GPS jitter, provisional alignment or road ambiguity; these hypotheses were not separately ablated and are not presented as proven causes. The case is retained rather than hidden by a universal “decoder improves accuracy” claim.


# 13  Geographical visualisation and applications



![Figure 9: Genuine rendered Folium view of student-test 00000052. GPS, public road context, decoded route and supplied development reference are distinct layers. Reference overlays are confined to report maps.](reports/assets/development_map.png)


The development report map is constructed from executed model output. It does not substitute the reference path for the prediction. The production/evaluator UI receives only public GPS/network inputs, predicted geometry and permitted aggregate scores. It cannot request a hidden reference overlay.


![Figure 10: Genuine local Streamlit UI screenshot captured with separate backend and frontend processes. This is a localhost demonstration, not a physical friend-laptop screenshot.](outputs/verification/ui.png)



## 13.1  Browser frontend from the supplied ZIP


The additionally supplied frontend-web.zip contains React 18, TypeScript, Vite and React-Leaflet. Its source is extracted to frontend-web/ while node_modules, local environment files and ZIPs are excluded from publication. It uses a same-origin /api development proxy to Laptop 1. Compatibility updates accept invalid GPS rows safely, display decoded/classifier/network geometry, show available scores, and bound per-request waits. The visual layout is retained. An optional online basemap and theme fonts can require internet; this client is distinguished from the fully self-contained Streamlit/Folium vector map.

Frontend build and browser integration evidence are included only where completed. The Streamlit offline-map browser check recorded zero external map requests. A production React static host must proxy /api to the backend; building static assets alone does not configure that proxy.

Executed React verification. The extracted frontend passed TypeScript checking and its Vite production build. A real-browser test loaded all ten trip options, submitted student-test 00000052 through the actual backend, displayed decoded geometry and its 96.56% derived-reference score, and repeated the action successfully. No browser page errors were recorded. This was a local two-process check, not a physical two-laptop test.


![Figure 11: Genuine React/Vite frontend screenshot after a successful saved-model request through its development proxy.](outputs/verification/react_ui.png)



# 14  API, deployment and repeated interaction


| Endpoint | Contract |

| --- | --- |

| GET /health | Loaded-artifact and data readiness, model version and mode; no private paths. |

| GET /trips | IDs from the configured development/evaluation allowlist. |

| POST /predict/{trip_id} | 202 and job ID; 404 unknown trip; 503 unavailable model/data; 429 another active prediction. |

| GET /jobs/{job_id} | Queued/running/completed/failed stage and safe error; expired jobs return 404. |

| GET /results/{job_id} | Completed predicted geometry, observation statuses, timing and permitted aggregate scores. |

A single backend process allows one active prediction and retains at most 32 recent jobs. In-memory jobs expire after restart. Exceptions mark failures and release UI controls; they do not leave a perpetual loading state. Logs record request/job/trip IDs, stage, model version and duration to stdout and rotating local files. Backend time limits are checked between stages; routing searches are bounded. This is not a hard real-time cancellation guarantee.


## 14.1  Private evaluation boundary


Prediction receives explicit .track/.nodes/.arcs inputs and does not scan for .route. Only the post-prediction scorer reads an authorised reference. Missing references yield reference_unavailable with absent metrics, not a fake zero. Evaluation-mode responses omit reference arcs, geometry, aligned labels, pointwise correctness and reference diagnostics. Reserved mode requires exactly ten complete trip entries from a configured root or manifest; student-test trips are never renamed as reserved trips.


## 14.2  Actual receiving-backend evidence



```text
2026-11-18 15:35:07,185 job_id=232c77401cd94bd79b3ddf99bb833ee6 trip_id=00000091
stage=viterbi status=running model=xgboost-8679132db4c1
2026-11-18 15:35:07,928 job_id=232c77401cd94bd79b3ddf99bb833ee6 trip_id=00000091
stage=completed status=completed model=xgboost-8679132db4c1
2026-11-18 15:35:07,928 job_id=232c77401cd94bd79b3ddf99bb833ee6
duration_seconds=0.924606 success=true
2026-11-18 15:35:08,825 job_id=14a21ed73ab843deadf637cfe376e469 trip_id=00000000
stage=viterbi status=running model=xgboost-8679132db4c1
2026-11-18 15:35:09,710 job_id=14a21ed73ab843deadf637cfe376e469 trip_id=00000000
stage=completed status=completed model=xgboost-8679132db4c1
2026-11-18 15:35:09,710 job_id=14a21ed73ab843deadf637cfe376e469
duration_seconds=1.384020 success=true
```


The block above is a typeset export of actual receiving-backend logs, not a fabricated terminal screenshot. Genuine source logs are retained in outputs/verification/http_server.log. Loopback addresses establish local HTTP integration only.


# 15  Testing, runtime and reproducibility


| Check | Executed result / evidence |

| --- | --- |

| Regression suite | 30 tests passed; report_test_results.txt. Five-family serialization tests fit tiny synthetic fixtures only. |

| All-ten API run | All ten development trips completed; each retained one match/status per observation. |

| Repeated interaction | A completed trip was submitted again successfully; browser repeat control passed. |

| Error paths | Unknown ID rejected; missing model returns unavailable readiness; failed inference reaches terminal safe status; active-job bound enforced. |

| Reference isolation | Prediction without a reference works; API snapshots were scanned for reference label/geometry fields. |

| Offline map | Bundled Leaflet script/style; browser reported zero external map requests for the offline vector map. |

| Backend package | ZIP CRC/checksums validated; extracted package loaded model and predicted in a fresh process. |

| Largest graph / longest track | Both separately executed below; observation correspondence verified. |

| Physical LAN / Windows / official scoring | Not executed in the accessible environment. Exact remaining steps are in the setup guides. |

| Benchmark | GPS rows | Directed arcs | Prediction + loading (s) | Blocks |

| --- | --- | --- | --- | --- |

| 00000013 | 2,580 | 307,392 | 2.550 | 6 |

| 00000054 | 19,313 | 116,916 | 10.134 | 348 |

Recorded development request durations range from 0.190 to 4.515 s in the saved all-ten run. These are observations of that execution, not guaranteed latency. Repeated-trip verification passed; the recorded timings do not constitute a controlled cold-versus-warm cache study.

| Environment / package | Observed value |

| --- | --- |

| Python | 3.12.3 |

| Host OS | Linux 6.8.0-149-generic |

| CPU | AMD Ryzen 7 7435HS (current host identification) |

| GPU evidence | XGBoost/MLP run.json record cuda; current sandbox could not re-query the NVIDIA driver. No GPU model/VRAM is invented. |

| numpy | 2.5.3 |

| pandas | 3.0.6 |

| scipy | 1.18.1 |

| scikit-learn | 1.9.1 |

| shapely | 2.2.0 |

| pyproj | 3.8.0 |

| xgboost | 3.4.1 |

| torch | 2.14.1 |

| optuna | 5.1.0 |

| fastapi | 0.163.2 |

| streamlit | 1.67.0 |

| folium | 1.0.0 |


# 16  Conclusions, limitations and next actions


The delivered project connects candidate learning to directed sequential reconstruction and actual applications. XGBoost is justified by the saved OOF selection rule and student-test comparison, not by training accuracy. The same saved classifier improves from 87.49% to 91.01% mean trip Top-1 after decoding, while arc-set F1 and ordered quality also improve on average. One trip worsens; explicit failure analysis is necessary. The largest graph and longest track execute with complete observation correspondence.

| Limitation / remaining work | Required next action |

| --- | --- |

| Provisional pointwise reference convention | Ask the organiser for the official scorer/alignment convention; retain derived-reference status until resolved. |

| Incomplete saved tuning study | Run the generic resumable 15-trial workflow, save search history and compare tuned/untuned results without promising a gain. |

| Candidate/heading ambiguities | Measure prediction-only radius/K or reliability refinements on training/development data; do not insert reference roads. |

| No feature ablation/calibration study | Do not claim a causal contribution for individual features or calibrated emission likelihoods. |

| Physical two-laptop demonstration | Run Laptop 2 against Laptop 1’s LAN IP and capture both sides; localhost success is insufficient. |

| Reserved evaluation | Freeze model/configuration, provision the ten authorised inputs, then evaluate once under organiser rules. |

| Historical provenance differences | Keep recorded estimator configurations and new run identities distinct; do not retroactively attribute old metrics to changed defaults. |

AI-assistance disclosure. AI assistance supported source implementation, debugging, report drafting and evidence assembly. Experimental claims in this report are linked to saved fits, predictions or executed checks. No metrics, screenshots, identities, citations or physical demonstrations are invented. Team members remain responsible for correctness and for explaining the code and conclusions to the mentor.


# 17  References and evidence index


[R1] Paul Newson and John Krumm (2009). Hidden Markov Map Matching Through Noise and Sparseness. ACM SIGSPATIAL GIS.<br/><link href="https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/" color="#007d87">https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/</link>

[R2] scikit-learn documentation: GroupKFold and grouped cross-validation.<br/><link href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html" color="#007d87">https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html</link>

[R3] XGBoost documentation: Notes on Parameter Tuning.<br/><link href="https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html" color="#007d87">https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html</link>

[R4] Optuna documentation: TPESampler and startup behaviour.<br/><link href="https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html" color="#007d87">https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html</link>

[R5] Shapely documentation: STRtree segment spatial indexing.<br/><link href="https://shapely.readthedocs.io/en/stable/strtree.html" color="#007d87">https://shapely.readthedocs.io/en/stable/strtree.html</link>

[R6] pyproj documentation: Transformer coordinate handling.<br/><link href="https://pyproj4.github.io/pyproj/stable/api/transformer.html" color="#007d87">https://pyproj4.github.io/pyproj/stable/api/transformer.html</link>

Official local sources: Main Guidelines.pdf; Dataset Exploration.pdf; Models and Training Guidelines.pdf; Deliverables.pdf; Assessment and Plagiarism Policy.pdf; UE24CS352A_ML hackathon_Instructions.docx. Team roster and project requirements are taken from these supplied materials. Web method references were checked on 18 November 2026.

| Evidence | Repository-relative location |

| --- | --- |

| EDA and raw-input hashes | outputs/audit/inventory.csv; audit_summary.json; data_manifest.json |

| Feature/label coverage | outputs/prepared/candidate_diagnostics.csv; reports/report_evidence.json |

| Fit/CV model records | outputs/experiments/models/<family>/ |

| Consolidated model comparison | outputs/experiments/student_test/model_comparison.csv and *_per_trip.csv |

| Partial tuning record | outputs/experiments/tuning/tuning.sqlite3 (local, excluded from public raw-data publication) |

| API/decoder run | outputs/verification/api_trip_metrics.csv; <trip_id>.json; summary.json |

| Browser and performance checks | outputs/verification/browser_checks.json; large_trip_benchmarks.json |

| Genuine screenshots | outputs/verification/ui.png; map.png; report development map in reports/assets/ |

| Report source and metadata | scripts/generate_report.py; reports/model_metadata.json; reports/report_evidence.json |


# Appendix A  Exact startup and pipeline commands



## A.1  Laptop 1 — backend



```text
cd "/path/to/PathSnap"
bash scripts/setup_backend.sh
export PATHSNAP_DATA_ROOT="/path/to/Student folder"
bash scripts/start_backend.sh
# Verify locally:
curl http://127.0.0.1:8000/health
# Find the LAN IPv4 address:
hostname -I
```


The data-root directory must directly contain train/ and student_test/. The server binds 0.0.0.0:8000; this is not the URL entered by Laptop 2. Use the reachable IPv4 address on a shared network. Paths above are deployment placeholders, not hardcoded host paths.


## A.2  Laptop 2 — React browser frontend



```text
cd "/path/to/PathSnap/frontend-web"
npm ci
# Linux/macOS shell:
export VITE_API_BASE_URL="http://<LAPTOP_1_LAN_IP>:8000"
npm run dev
# Open the Vite URL, normally http://localhost:5173
# Build check:
npm run build
```


On PowerShell set `$env:VITE_API_BASE_URL="http://&lt;LAPTOP_1_LAN_IP&gt;:8000"` before `npm run dev`. The development proxy forwards /api to Laptop 1. For static production hosting, configure that proxy on the host/reverse proxy explicitly.


## A.3  Laptop 2 — Streamlit alternative



```text
cd "/path/to/PathSnapFrontend"
bash scripts/setup_frontend.sh
export BACKEND_URL="http://<LAPTOP_1_LAN_IP>:8000"
bash scripts/start_frontend.sh
# Open http://localhost:8501
```


Both laptops must permit device-to-device connections. If connection fails, inspect IP/port, shared subnet, VPN/proxy, firewall and access-point isolation. A narrowly scoped inbound backend-port rule may be required; disabling the entire firewall is unnecessary. Frontend-only packages require no datasets, model weights or GPU.


## A.4  Retraining, tuning and reproducible loop



```text
cd "/path/to/PathSnap"
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python scripts/check_environment.py --role training --require-gpu
.venv/bin/python scripts/run_full_pipeline.py \
  --data-root "/path/to/Student folder" \
  --device cuda --trials 15 --resume
# Existing saved-model verification without fitting/tuning:
.venv/bin/python scripts/run_full_pipeline.py \
  --data-root "/path/to/Student folder" \
  --skip-training --skip-tuning --skip-audit --skip-prepare
# Regenerate this report from saved evidence:
.venv/bin/python scripts/generate_report.py
```


Use `--device cpu` for an explicit CPU training run. A requested CUDA run fails clearly if GPU execution is unavailable; it must not be silently reported as GPU training. The selected inference model is exported CPU-portable with its checksum, preprocessing and feature order. ZIP deployment archives can be generated locally using `scripts/package_dist.py`; supplied/deployment ZIPs are intentionally not published to GitHub.
