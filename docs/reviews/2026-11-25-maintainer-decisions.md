# Maintainer-decision items from the 2026-11-24 review — dispositions

Follow-up to `2026-11-24-claude-codex-review-loop.md`, whose "Maintainer decisions" section listed
the items deliberately left unchanged in PR #80. Branch `fix/maintainer-decisions-2026-11-25`,
based on merged `main` (`fde41b4`). Item IDs are defined in `2026-11-24-evidence/`.

## Summary

| Disposition | Count | Items |
|---|---|---|
| Fixed | 23 | R4-A3, data-B9, core-B3, core-B1, opthet-B11, R4-A5, opthet-B10, optstrat-B9, optstrat-B10, optstrat-B11, optstrat-B12, optescape-B3, R2-A4, data-B1, gui-B2, optwrap-B4, optrest-B3, opthet-B13, cli-B10, R4-C2, R3-A1, opthet-B8, core-B9 |
| Closed as intended | 12 | optrest-B4, optescape-B1, opthet-B6, config-B2, config-B9, core-B2, optescape-B5, cli-B9, ioviz-B7, ioviz-B8, optstrat-B7, optstrat-B13 |
| Still open | 0 | — |

Four choices were made by the maintainer before the work started (recorded below where they apply):
honour `regularization.enable` and keep template behaviour; apply the DOF reporting fixes and
regenerate a golden only if it moves on uncertainties alone; leave the three solver-trajectory
items; disable the dead streaming paths rather than repair them.

## Recommendation

Review the default-visible changes below before merging. No item is left open: the last three
were decided in a three-way round (Claude, Codex, Antigravity) and implemented; see
"Decided by the three-way round".

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

## Decided by the three-way round

Claude, Codex and Antigravity each took a position on the last three items from one brief, then
ratified a merged plan. Both rounds were unanimous. Record: `2026-11-25-evidence/` (brief, merged
plan, two position files per reviewer).

- **R4-A3 — zero-weight rows are not observations (option A).** A zero-weight pixel keeps a
  residual row that is always 0. The heterodyne joint fits counted it as an observation, so
  `s² = cost / (rows − p)` was too small and uncertainties too tight on masked data.
  - The row count now comes from the weight mask on the kernel's own support, and is used for the
    covariance DOF, the reported `covariance_dof.n_valid` and reduced chi-squared on the individual,
    averaged and constant joint paths (the paths a masked fit below 1 M points takes, since the
    engine route declines non-uniform weights).
  - The same count is applied to the ≥ 1 M stratified-LS path (covariance and reduced chi-squared)
    and to hybrid streaming's reduced chi-squared.
  - With no data DOF left after masking, covariance, uncertainties and reduced chi-squared are NaN.
  - Unweighted and all-positive-weight fits are unchanged. Parameters and chi-squared never change.
  - Not changed: the CMA-ES warm-start skip test (it steers the solve), and the hybrid-streaming
    covariance scale, which comes from that optimizer's own estimate.
- **data-B9 — `cache_directory` removed (option B).** The loader never read it; caches go to
  `cache_file_path`, defaulting to the data folder, and still do. The key is gone from the four
  templates and the manager defaults; templates show `cache_file_path` (a directory) as a commented
  option. A null `cache_file_path` now means the data folder instead of crashing.
  **Default-visible:** configs made from the old templates still carry `cache_directory`, so each
  load logs one warning that the key is unused.
- **core-B3 — the adapter refuses times its fixed grid would clamp (option B).** `NLSQAdapter`
  (opt-in) raises a `ValueError` naming `use_adapter=False` when `t1` or `t2` leaves
  `[0, 10000·dt]`. Through `fit_nlsq_jax` that triggers the existing logged fallback to
  `NLSQWrapper`, which uses a real time grid, so the user gets a warning and a correct fit instead
  of a silently clamped one. Passing a real grid through `get_or_create_model` remains a possible
  enhancement; both reviewers judged it disproportionate (a non-uniform time axis would integrate
  wrongly).

## Review of PR #81 (Codex + Antigravity)

Both reviewed the whole diff independently (`2026-11-25-evidence/pr81_codex.md`,
`pr81_agy.md`). Codex reported 6 must-fix findings, Antigravity 5; merged, eight distinct issues.
Each was reproduced with a failing test first (`tests/review_r1/test_pr81_review.py`). Neither
disputed any item closed as intended.

| Finding (reviewer IDs) | Outcome |
|---|---|
| Engine route accepted all-zero or mask-like weights under `loss="linear"` and fitted unweighted (c-F1, a-F4) | Fixed: any non-positive or non-finite weight goes to the weighted solver |
| A fully masked in-memory fit crashed building its result (a-F1) | Fixed: reduced chi-squared NaN, `quality_flag="unknown"` |
| Stratified LS and the shared builder reported a finite reduced chi-squared at zero data DOF; NaN adapter covariance was not flagged as a placeholder (c-F2, a-F2) | Fixed |
| Duplicate phi rows picked up each other's mask on the stratified and streaming paths (c-F3) | Fixed: sigma is taken per measurement when angles repeat |
| Hybrid streaming's SSR still summed masked pixels (c-F4, a-F3) | Fixed: masked pixels are excluded from the SSR as well as the DOF |
| Layer 3 off switch: a flat `regularization_mode` overrode nested `enable: false`; controller callbacks still carried L3 groups; streaming builders ran a penalty for `mode: "none"` (c-F5, a-F5) | Fixed. Explicit nested `enable: false` now wins, with a warning |
| NumPy integer `end_frame` lost the tail-clamp metadata (c-F6) | Fixed |
| The unsupported-streaming test only searched source text; `config=None` skipped the guard (c-F7, a-F7); cache metadata fallback ignored a `None` window (a-F6) | Fixed |

Not changed, with the reason:

- **Laminar streaming's flat `enable_group_variance_regularization: true` with nested
  `regularization.enable: false`** (part of c-F5). That flat flag switches on a different, simpler
  mechanism inside the streaming optimizer, set explicitly by the user; the nested switch governs
  the adaptive Layer 3 regularizer only.

Two of these fixes change behaviour for non-default configs: `regularization.mode: "none"` now
means Layer 3 off on both hybrid-streaming paths (it used to apply an absolute-variance penalty),
and a weighted fit with repeated phi angles on the large-data paths now uses each measurement's own
weights.

Closing check (`2026-11-25-evidence/pr81_codex_closing.md`, `pr81_agy_closing.md`): both reviewers
confirmed every finding fixed and accepted the one sub-point left unchanged. Codex found one
regression from the fixes — on heterodyne hybrid streaming an explicit `regularization.enable:
null` had always meant off and was turned on by the shared gate — which is fixed with a test.

## Verification status

See the pull request for the test and golden-parity results on the final commit.
