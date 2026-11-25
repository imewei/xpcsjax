# codex position — round 2
## R4-A3: RATIFY
## data-B9: RATIFY
## core-B3: RATIFY
## core-B3 open point (check inside model_func): DROP — Validate both time coordinates once at the adapter boundary, where their meaning and effective `dt` are known. A check tied to cache misses and concrete NumPy arrays would leave traced calls uncovered.
## Anything the plan missed
- For masked fits with nonpositive reporting DOF, report reduced chi-squared as undefined. The shared helper floors the denominator at one, so handle this in the three joint result builders to preserve all-positive parity and CMA-ES gating. Retain the averaged path's expanded parameter count for reduced chi-squared and its fitted parameter count for covariance.
- Read the same coordinate aliases as `_flatten_xpcs_data` (`t1`/`t1_2d` and `t2`/`t2_2d`) for the adapter check. Cover both cache settings and the case where only `t2` exceeds the endpoint tolerance.
END_OF_POSITION
