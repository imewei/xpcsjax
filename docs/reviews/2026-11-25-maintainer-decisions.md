# Maintainer-decision items from the 2026-11-24 review — dispositions

Follow-up to `2026-11-24-claude-codex-review-loop.md`, whose "Maintainer decisions" section listed
the items deliberately left unchanged in PR #80. Branch `fix/maintainer-decisions-2026-11-25`,
based on merged `main` (`fde41b4`). Item IDs are defined in `2026-11-24-evidence/`.

## Summary

| Disposition | Count | Items |
|---|---|---|
| Fixed | 20 | core-B1, opthet-B11, R4-A5, opthet-B10, optstrat-B9, optstrat-B10, optstrat-B11, optstrat-B12, optescape-B3, R2-A4, data-B1, gui-B2, optwrap-B4, optrest-B3, opthet-B13, cli-B10, R4-C2, R3-A1, opthet-B8, core-B9 |
| Closed as intended | 12 | optrest-B4, optescape-B1, opthet-B6, config-B2, config-B9, core-B2, optescape-B5, cli-B9, ioviz-B7, ioviz-B8, optstrat-B7, optstrat-B13 |
| Still open | 3 | R4-A3, data-B9 (`cache_directory`), core-B3 |

Four choices were made by the maintainer before the work started (recorded below where they apply):
honour `regularization.enable` and keep template behaviour; apply the DOF reporting fixes and
regenerate a golden only if it moves on uncertainties alone; leave the three solver-trajectory
items; disable the dead streaming paths rather than repair them.

## Recommendation

Review the three default-visible changes below before merging, then decide the three open items.
`R4-A3` is the only open item that affects a reported number.

## Fixed

Wrong results today:

- **core-B1** — heterodyne time axis when `end_frame` exceeds the file. The loader now reports
  `data["frames_clamped_at_end"]` (cold load and warm cache hit), and `sync_time_axis` drops that
  many trailing model points before treating any remaining shortfall as leading-frame exclusion.
  Normal and frame-0-excluded loads are unchanged. A bare NPZ with no window metadata carries no
  clamp information and keeps the old behaviour.
- **opthet-B11 / R4-A5** — the engine route solves unweighted. Non-unit uniform weights under a
  robust loss now raise `NotImplementedError` there and take the existing fallback to the weighted
  `fit_nlsq_multi_phi`. Unit weights, and uniform weights under `loss="linear"`, stay on the
  engine route (dropping them is exact).

Default-visible changes (read these):

- **opthet-B10 / optstrat-B9** — `regularization.enable: false` now switches Layer 3 off (shared
  controller, laminar streaming layers, heterodyne config). The four shipped templates now say
  `enable: true`, so template users keep Layer 3 on those paths. **Exception:** heterodyne hybrid
  streaming already honoured the flag, so Layer 3 was off there for template users and is now on.
  Anyone with their own config containing `enable: false` now gets Layer 3 off.
- **optwrap-B4** — averaged-mode covariance (the default `auto` at `n_phi >= 3`) is rescaled onto
  the expanded DOF the module docstring documents: variances grow by `(n - p_solver)/(n - p_expanded)`
  (about 1.4% for 300 points and 3 angles). Parameters and chi-squared unchanged.
- **cli-B10** — `xpcsjax.cli` no longer exports `main` (it resolved to the submodule). Use
  `from xpcsjax.cli.main import main`. Entry points are unaffected.

Reporting only:

- **optrest-B3** — out-of-core reduced chi-squared under `constant` counts the scaling slots the
  loop actually fitted.
- **opthet-B13** — heterodyne sequential fallback reduced chi-squared uses `n_phi * n_physics`.
- **R2-A4** — laminar CMA-ES results report `convergence_status` from a new honest
  `diagnostics["converged"]`; `CMAESResult.success` keeps its eligibility meaning, so no fit
  selection changes.
- **core-B9** — `compute_fraction` returns the smooth-clipped fraction the kernel uses.
- **R3-A1** — duplicate phi angles: stratified LS labels the shared slots (sorted-unique angles);
  the engine route declines duplicates explicitly and falls back.

Behaviour-identical or failure-path only:

- **optstrat-B10/B11/B12, optescape-B3** — the fallback chain names the unsupported generic
  hybrid-streaming tier instead of dying on an `AttributeError`; `refinement_workflow: streaming`
  is skipped with a warning; `StreamingExecutor` documents its full-batch-only contract. The
  streaming solver is NOT activated.
- **data-B1 / gui-B2** — a per-row q list is sliced with the angles kept by phi filtering.
- **R4-C2** — the HDF5 allocation guard budgets the largest candidate shape and dtype.
- **opthet-B8** — the heterodyne gradient preflight probes every parameter before failing.

## Closed as intended

- **optrest-B4** (recovery overrides `x_scale`), **optescape-B1** (L2 convergence after one sweep),
  **opthet-B6** (adapter reports success after `max_nfev`): maintainer decision 2026-11-25 to leave
  the solver trajectory unchanged. optrest-B4 is a verbatim upstream port and moving it would move
  the `rtol=1e-10` goldens.
- **config-B2** — equal bounds stay in the optimizer vector: the stated convention
  (`tests/optimization/test_fixed_parameters_integration.py`, module docstring).
- **config-B9** — null `initial_parameters.values` returns physical midpoints only: documented at
  `config/manager.py` (`get_initial_parameters`), and no consumer reads per-angle slots from it.
- **core-B2** — 64 substeps in the scattered quadrature are deliberate and more accurate than the
  grid trapezoid (`tests/core/test_physics_nlsq_degenerate_grid.py`).
- **optescape-B5** — multistart workers left alive after the 1800 s timeout: documented as accepted
  in `multistart.py`.
- **cli-B9** — activation scripts keep a caller-supplied XLA device count, matching
  `xpcsjax/__init__.py`.
- **ioviz-B7** — logging `propagate` handling is pinned by `tests/test_logging_quality_gate.py`.
- **ioviz-B8** — the annotation panel's scalar mean is the documented summary.
- **optstrat-B7** — pseudo-inverse for per-angle sequential covariance is the documented choice.
- **optstrat-B13** — `optimization.max_iterations` (full out-of-core passes) and
  `optimization.nlsq.max_iterations` count different things.

## Still open

- **R4-A3** — heterodyne joint covariance DOF still counts rows zeroed by zero weights. Fixing it
  needs the weight mask threaded into both result builders with the same row selection the residual
  uses; counting exact-zero residuals instead would be fragile. Affects reported uncertainties only
  when weights contain zeros.
- **data-B9 (`cache_directory`)** — the templates ship `cache_directory: "./cache"` but the loader
  ignores it and caches next to the data file. Honouring it relocates every template user's cache
  (a one-time regeneration, old caches orphaned) and makes a relative path depend on the working
  directory. Needs a decision: honour it, or remove the key from the templates.
- **core-B3** — the opt-in adapter path does not pass a time grid, so times beyond `10000 * dt`
  clip. The kernel docstring records why this was deferred: the grid must come from the full
  experiment axis outside any traced closure, which needs an API change to `get_or_create_model`.

## Verification status

See the pull request for the test and golden-parity results on the final commit.
