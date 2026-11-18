# PathSnap technical report — development draft

## Status and scope

Development report. Full retraining and the new tuning workflow are to be executed by the user. Saved-model inference and synthetic regression results are recorded separately. Physical Laptop 2 and official scoring remain pending.

## Dataset understanding and EDA

{"train": {"trips": 80, "observations": 208417, "nodes": 1933485, "arcs": 3876904, "reference_entries": 31336, "invalid_observations": 0, "zero_length_arcs": 100, "duplicate_endpoint_pairs": 1775, "reference_breaks": 13}, "student_test": {"trips": 10, "observations": 19160, "nodes": 90689, "arcs": 185041, "reference_entries": 3477, "invalid_observations": 0, "zero_length_arcs": 6, "duplicate_endpoint_pairs": 128, "reference_breaks": 1}}

## Preprocessing and features

The raw files are preserved. Per-trip WGS84-assumed azimuthal-equidistant projection, original arc/node IDs, long-segment-safe STRtree candidates, and 22 relative geometry/motion/topology features are used. See docs/FEATURE_AUDIT.md. Candidate budgets count directed arcs and ties break by arc ID. Radius expansion occurs only when no candidate is retained. This can miss an offset true road when a closer false road exists; final recall is measured, not presumed.

## Reference limitations

Labels use ordered route-occurrence dynamic programming, not a row-wise zip or route-membership target. Labels are derived references, not independently established physical truth. A pointwise organiser scoring convention remains unresolved. Reference residuals above 300 m are unknown; coverage and true candidate misses are reported separately.

## Model experimentation

Logistic regression uses fold-local imputation/scaling and logistic loss. Decision trees use regularized depth/leaf size and Gini impurity; random forests average bounded-depth trees. XGBoost uses binary logistic loss and histogram boosting. The two-hidden-layer PyTorch network uses batched BCE-with-logits and fold-local scaling. Strengths: linear baseline interpretability; tree nonlinear interactions; forest variance reduction; boosting flexible ranking; MLP learned nonlinear interactions. Limitations: linear feature interactions; individual tree variance; forest size/runtime; boosting search/overfit risk; MLP scale/sensitivity. Configurations and genuine loss histories are saved with each new training run. Existing CV and student-test results are development estimates. Training accuracy alone is not evidence of generalization.

## Bayesian tuning

The repeatable workflow selects a family from grouped training OOF mean-per-trip Top-1; candidate macro-F1, hit-conditional observation loss, trip variation and fit cost break ties. TPE tunes whichever of the five families wins, with 15 total trials and five startup trials. Studies are persistent and keyed to feature/model/search identity. Tuned/untuned student-test comparison and best parameters are emitted by the user-run command. No new full-data tuning gain is claimed in this report.

## HMM and Viterbi

States are directed arc IDs plus projected fractions. Emission potentials are clipped positive classifier probabilities normalized within each observation, a hybrid discriminative approximation rather than a calibrated generative likelihood. The log-space recurrence maximizes previous score minus absolute directed travel/GPS displacement discrepancy divided by beta, plus log emission. Directed shortest paths use edge-aware adjacency, bounded source caches and original arc reconstruction. Same-arc backwards jitter is bounded by 12 m. Gaps over 60 seconds, missing candidates and impossible transitions create independently backtracked blocks. Intermediate arcs are predictions; no geometry bridges disconnected blocks. Turn restrictions were not supplied.

## API, UI and repeated pipeline

Laptop 1 serves a checksummed model without retraining. /health, /trips, /predict/{trip_id}, /jobs/{job_id}, /results/{job_id} implement bounded single-process jobs and safe failure states. Private scoring runs only after prediction. Laptop 2 contains Streamlit, an HTTP client and a self-contained Folium/Leaflet vector map. It displays actual decoded/classifier/network/GPS layers, scores, stage/status and preserved results across reruns. A bounded cycle runner joins audit, preparation, five-model fitting, comparison, tuning, export, tests, HTTP verification, report and packaging.

## Verification and limitations

Synthetic regression tests cover direction, duplicate edges, zero lengths, projection, gaps, missing candidates, decoder restarts, candidate misses, model CPU serialization, API failure/readiness/concurrency, and offline map dependencies. Actual development API verification is in outputs/verification/. A localhost test does not prove physical two-laptop connectivity. Reserved evaluator references were not accessed for development. Newly changed training/tuning code is not represented as a completed full-data experiment.

## Reproducibility and AI assistance

Run the exact commands in README.md and docs/TWO_LAPTOP_DEMO.md. AI assistance was used for implementation, debugging and report generation. Tables and figures are generated from saved experimental outputs and executed checks; team members must verify and be able to explain the work. Team identities are not inferred.

## References

Official supplied Main Guidelines, Dataset Exploration, Models and Training Guidelines, Deliverables, Assessment and Plagiarism Policy PDFs and the hackathon DOCX were reviewed. Method references: https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/ ; https://shapely.readthedocs.io/en/stable/strtree.html ; https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html ; https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html ; https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html

Actual model comparison:

model,cv_mean_trip_top1,cv_macro_f1,cv_conditional_observation_loss,cv_top1_std,fit_seconds,student_trips,candidate_rows,candidate_accuracy,candidate_precision,candidate_recall,candidate_macro_f1,candidate_binary_log_loss,confusion_matrix,mean_trip_observation_top1,pooled_observation_top1,candidate_miss_rate,label_coverage,prediction_seconds
xgboost,0.8319535277953294,0.8996125152816722,0.4165934763349236,0.1486949635459711,3.6788990959976218,10,169191,0.967261851989763,0.8809643958508551,0.821413634462568,0.9158855159208622,0.0804648846387863,"[[147940, 2123], [3416, 15712]]",0.8749108980605994,0.8626826722338204,0.001670146137787,1.0,0.2633111770119285
random_forest,0.8276099822817663,0.8748477703152953,0.4945183714697601,0.1534619405773981,133.1915538840003,10,169191,0.9494122027767434,0.706095706095706,0.9465181932245922,0.8898339927648664,0.1257372954641207,"[[142527, 7536], [1023, 18105]]",0.8671270692245692,0.8577244258872652,0.001670146137787,1.0,0.588449049006158
pytorch_mlp,0.8251499929500559,0.894827690696429,0.4613717796826421,0.1535606498486406,12.751876763002656,10,169191,0.9631599789586917,0.8616704998036686,0.8030635717273108,0.9053285505374148,0.0881250500679016,"[[147597, 2466], [3767, 15361]]",0.8549646196336868,0.8455114822546973,0.001670146137787,1.0,0.026104769000085
decision_tree,0.82088359810495,0.833153009577329,0.6715486585862368,0.1523952680218869,25.984381529997336,10,169191,0.9365982824145492,0.6496277562070316,0.9534190715181932,0.8679491183819383,0.1552555673546665,"[[140227, 9836], [891, 18237]]",0.8411342134562162,0.8251565762004175,0.001670146137787,1.0,0.0082160370002384
logistic,0.8204869901853431,0.783022628404526,0.7184902650247758,0.1502677130248162,6.27362119800091,10,169191,0.9055446211677928,0.5472253068819593,0.9532099539941448,0.8197010328445142,0.198613554239273,"[[134977, 15086], [895, 18233]]",0.8439035827789387,0.829749478079332,0.001670146137787,1.0,0.0533901729977515


Executed API comparison:

trip_id,status,wall_seconds,reported_seconds,blocks,classifier_top1,classifier_continuity,classifier_edit_distance,decoded_top1,decoded_continuity,decoded_edit_distance
0,completed,1.6697538789994724,1.426343809998798,1,0.937674790251698,0.9768185451638688,0.4879518072289157,0.9556532161406311,1.0,0.2009345794392523
7,completed,1.1300065309987986,0.5825450740012457,4,0.9771048744460856,0.9830007390983,0.2389937106918239,0.9254062038404728,0.9977827050997784,0.0692307692307692
9,completed,1.8647467119990324,1.541056979996938,1,0.8198284080076264,0.9627862595419848,0.3288364249578415,0.8636796949475691,1.0,0.1059322033898305
40,completed,1.985916465000628,1.8323040979994405,2,0.9392320534223706,0.968937875751503,0.4584070796460177,0.9606010016694492,0.9996659986639946,0.1868421052631579
52,completed,0.5300696140002401,0.1901291200010746,1,0.9351145038167938,0.9808429118773946,0.1157894736842105,0.965648854961832,1.0,0.0210526315789473
56,completed,5.116290533002029,4.515330635000282,3,0.8587670540677109,0.9191306545362647,0.2871116225546605,0.926478019201617,0.9994945665908516,0.074074074074074
65,completed,1.2301163640004231,0.7927830240005278,2,0.8502604166666666,0.93733681462141,0.452,0.9088541666666666,1.0,0.1369047619047619
76,completed,3.408002785003191,3.1316739629983203,1,0.857335127860027,0.9508417508417508,0.495225102319236,0.9030955585464334,1.0,0.2940038684719536
77,completed,4.347958349000692,3.892754544998752,5,0.7374073127077474,0.921227621483376,0.8123682361208714,0.7831756583993863,0.9989769820971868,0.5663430420711975
91,completed,1.3227640400000382,0.9246062120000716,2,0.8363844393592678,0.9484536082474226,0.52,0.908466819221968,0.9988545246277204,0.2041522491349481
