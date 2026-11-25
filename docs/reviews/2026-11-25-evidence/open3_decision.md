# Round 2 — ratify the merged decision

Round 1 was unanimous (claude, codex, agy): R4-A3 = A, data-B9 = B, core-B3 = B. Your round-1
files are `$WT/.review/open3_codex_r1.md` and `$WT/.review/open3_agy_r1.md` (read the other
reviewer's now). Below is the merged implementation plan, taking conditions from both of you.
`$WT` = `/home/wei/Documents/GitHub/xpcsjax/.claude/worktrees/codex-review-loop`. Same rules as the
brief (`$WT/.review/open3_brief.md`): READ ONLY except your own file, no filesystem search.

## R4-A3 (A)
1. In each heterodyne joint path, count contributing rows ONCE at build time by applying the
   kernel's own off-diagonal row selection to `weights_batch > 0`.
2. Generalise `_drop_penalty_rows_from_dof` into one rule: with `n_rows_solver` = rows the solver
   saw (data rows + penalty rows) and `n_valid` = contributing rows, return the inputs unchanged
   when `n_rows_solver == n_valid`; all-NaN covariance/uncertainties when `n_valid - p <= 0`;
   otherwise scale covariance by `(n_rows_solver - p) / (n_valid - p)` and uncertainties by its
   square root. It therefore also applies with L3 off.
3. Apply it to the individual (`_build_joint_result`), averaged (`_fit_joint_averaged_multi_phi`)
   and constant (`heterodyne_constant_mode.py`) paths, and use the same `n_valid` for
   `noise_normalized_reduced_chi2` and `diagnostics["covariance_dof"]`.
4. NOT changed: the CMA-ES warm-start auto-skip criterion (it steers the solver trajectory).
5. Unweighted / all-positive-weight fits must be bit-identical.

## data-B9 (B)
1. Remove `cache_directory` from the four templates, from `ConfigManager._DATA_PATH_KEYS` and from
   the manager's default config.
2. Templates document `cache_file_path` as an optional, commented example and say it is a
   DIRECTORY (default: the data folder).
3. Loader: `self.exp_config.get("cache_file_path") or data_folder` (a null value must not crash).
4. Loader logs a WARNING when `cache_directory` is present and `cache_file_path` is unset: the key
   is not used; caches are in the data folder; set `cache_file_path` to choose a directory.

## core-B3 (B)
1. Hoist the fallback grid size to one module-level constant in `xpcsjax/core/jax_backend.py`,
   used by both kernels and imported by the adapter.
2. In `NLSQAdapter._build_model_function` (before the cached/uncached split, homodyne modes only),
   raise `ValueError` naming `use_adapter=False` when max(t1, t2) exceeds
   `(GRID_SIZE - 1) * dt * (1 + 1e-12)` with the effective `dt`, or when min(t1, t2) is negative.
3. Open point (the two of you differed): agy also wants the same check inside the cached
   `model_func` on an `_xdata_cache` miss so direct callers of `get_or_create_model` cannot clip
   silently. Claude's view: NOT in this change — `xdata[:, 0:2]` there may be time VALUES or
   something else depending on the caller, the function is JIT-traced on some paths, and the only
   in-package callers go through `_build_model_function`. Say whether you insist.

## What to write
Your file (`$WT/.review/open3_codex_r2.md` or `$WT/.review/open3_agy_r2.md`), exactly:

# <codex|agy> position — round 2
## R4-A3: RATIFY | OBJECT — <reason if OBJECT>
## data-B9: RATIFY | OBJECT — <reason if OBJECT>
## core-B3: RATIFY | OBJECT — <reason if OBJECT>
## core-B3 open point (check inside model_func): DROP | INSIST — <one or two sentences>
## Anything the plan missed
<bullets, or `Nothing.`>
END_OF_POSITION
