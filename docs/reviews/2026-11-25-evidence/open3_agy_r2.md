# agy position — round 2
## R4-A3: RATIFY
## data-B9: RATIFY
## core-B3: RATIFY
## core-B3 open point (check inside model_func): DROP — All in-package adapter calls already pass through `_build_model_function` outside any JAX trace, so checking there closes the bug with minimal risk and avoids putting eager array checks inside a potentially traced closure. Direct callers of `get_or_create_model` can rely on a brief docstring note pointing to the `[0, (GRID_SIZE - 1) * dt]` fallback bound.
## Anything the plan missed
- `R4-A3`: Codex noted that `noise_normalized_reduced_chi2` (`xpcsjax/optimization/nlsq/heterodyne_data_prep.py:363`) floors DOF at `1` when `n_valid <= n_params`; confirm whether non-positive reporting DOF should return `nan` (matching the all-NaN covariance) or keep the existing `max(dof, 1)` floor so callers like `classify_quality_flag` stay consistent.
- `core-B3`: In `NLSQAdapter._build_model_function`, resolve time arrays via the same `t1`/`t1_2d` and `t2`/`t2_2d` fallback `_flatten_xpcs_data` uses (`xpcsjax/optimization/nlsq/adapter.py:1082-1088`), guard only when `dt_val is not None` (`compute_g1_batch` already raises on `dt=None`), and guard `min(t1, t2) < -1e-12 * dt` rather than bare `< 0` if tiny negative float noise is possible.
END_OF_POSITION
