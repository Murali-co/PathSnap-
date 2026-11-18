# Scoring assumptions

No organiser-supplied pointwise label convention or scorer was found in the accessible workspace. PathSnap therefore uses a documented provisional ordered-occurrence route-alignment DP, never zips track and route rows.

- Candidate rows are `(trip_id, observation_id, original_arc_id)` metadata.
- Only candidates geometrically aligned to the derived route label are positive.
- Reference route entries do not cause candidate insertion.
- Unknown/uncertain labels are retained for coverage reporting and excluded from fitting where indicated.
- Observations with labels and candidates absent count as incorrect/missing.
- Official reserved evaluation compatibility remains pending organiser-provided data and/or a scorer.
