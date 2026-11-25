# Claude + Codex review/debug loop — 2026-11-24

Branch `worktree-codex-review-loop` (base `bd0a549`, v0.1.8). Codex reviewed the whole package
module by module; Claude verified every claim by reproduction, fixed the confirmed ones, and Codex
adversarially re-reviewed each fix round. The loop ended with Codex's closing verdict
`AGREE_NO_BUGS`.

## Summary

| Round | Codex raised | Fixed | Left to the maintainer | Rejected |
|---|---|---|---|---|
| 1 (11 shards, whole codebase) | 126 claims | 89 | 29 | 8 |
| 2 (review of round-1 fixes) | 13 fix problems, 6 counters, 4 new | 20 | 3 | 0 |
| 3 | 11 follow-ups, 2 new | 12 | 1 | 0 |
| 4 | 5 follow-ups, 3 new | 5 | 3 | 0 |
| Closing check | 0 | — | — | — |

Round-1 counts are by claim ID after all later reclassifications (two pairs of claims shared a root
cause; data-B9 is counted as fixed for its `use_caching` half). Regression tests for every fix are
in `tests/review_r1/` (127 tests).

Classification rule used throughout: a claim was fixed only if it is reachable from an in-package
caller or the documented public surface **and** its fix does not change fitted parameters or
chi-squared on a default path. Otherwise it is a maintainer decision (behaviour as claimed) or
rejected (not a defect / unreachable).

## Recommendation

Run `make test-full-local` on this branch before merging (the final full-suite and forced-parity
runs did not complete; see below). Then decide the eleven higher-impact maintainer items, starting
with core-B1 and opthet-B11, which can produce wrong results today.

## Verification status

- Passed after the final commit: `tests/review_r1` (127), `ruff`, `mypy` (204 files), and the
  affected domain suites run per round (data, cli, viz, service, config, gui, heterodyne, targeted
  optimization files).
- Forced golden parity (`XPCSJAX_RUN_CHARACTERIZATION=1 XPCSJAX_RUN_ENGINE_PARITY=1`, `tests/parity`)
  passed 134/134 on the untouched tree and again midway through round 1.
- **Not completed on the final tree:** `make test-all-parallel` and the forced parity run. Both were
  stopped by the system for low memory during round 1 (no failure at that point: full suite 98%
  with 2840 passed, parity 51/134) and were not re-run. Run before merging:
  `make test-full-local`.
- Codex's review was static plus single-file probes; it did not run the full suite either.

## Fixes that change reported uncertainties or DOF

Fitted parameters and chi-squared are unchanged by these; reported covariance, uncertainties or
reduced chi-squared change:

- Fixed parameters no longer consume a degree of freedom (standard, sequential, adapter,
  stratified-LS, CMA-ES and out-of-core result paths).
- Masked rows (`t1 == t2`, `sigma <= 1e-10`) are excluded from the stratified-LS and out-of-core
  covariance dof.
- L3 penalty rows no longer inflate heterodyne joint covariances; zero data dof yields a
  placeholder (NaN) covariance.
- An indefinite Hessian on the streaming L2 paths yields a flagged placeholder covariance.

## Behaviour changes users can see

- `--threads N` no longer injects `--intra_op_parallelism_threads` into `XLA_FLAGS` (it aborted
  XLA on jaxlib 0.11.1).
- An explicit `experimental_data.file_path` wins over a conflicting legacy folder/filename pair.
- `NLSQ_MEMORY_FRACTION` and `optimization.nlsq.memory_fraction` now reach strategy routing
  (precedence: env, config, default).
- Unknown heterodyne `active_parameters` / `fixed_parameters` names raise; string `vary` flags are
  rejected; empty or non-mapping config files are rejected.
- `use_caching: false` disables the NPZ cache; a cache built for a different frame window is
  regenerated; malformed HDF5 bins (shape/dtype mismatch) raise instead of broadcasting/truncating.
- Saved parameter names follow the solver's canonical order; `diagnostic_covariance` moved from
  the result JSON to the NPZ.
- Data whose phi angles are not ascending: per-angle scaling, sigma and theory are now paired with
  the right angle in fits and plots (identity for ascending phi).

## Maintainer decisions

Behaviour is as Codex claimed; each needs a choice that was not Claude's to make.

Higher impact:

1. **core-B1 — heterodyne time axis when `end_frame` exceeds the file.** The loader clamps with a
   warning, then `sync_time_axis` trims the model's LEADING frames, shifting absolute times. Choose:
   reject the short window, or carry clamp provenance through loader, cache and dispatcher.
2. **opthet-B11 / R4-A5 — uniform non-unit weights on the engine route.** The engine drops them as
   a pure rescale, which is not true under `soft_l1`. Choose: weight the engine objective, or route
   weighted fits to `fit_nlsq_multi_phi`.
3. **opthet-B10 / optstrat-B9 — `regularization.enable` is ignored.** The shipped two_component
   template has `enable: false` with `mode: "relative"`, so L3 is effectively on. Pinned by
   `test_l3_banner_always_on_even_when_regularization_disabled`.
4. **optstrat-B10/B11/B12, optescape-B3 — streaming tiers that never run.** The fallback chain's
   STREAMING tier always raises and falls to CHUNKED; the exported `StreamingExecutor` and the
   CMA-ES `refinement_workflow: streaming` are likewise non-functional. Repair or remove.
5. **R2-A4 — `CMAESResult.success` is true after an unconverged search plus an unconverged
   refinement.** Callers gate candidate eligibility on it, so an honest flag needs a
   candidate/convergence split. `diagnostics["refinement_converged"]` now carries the real verdict.
6. **optescape-B1 — L2 hierarchical "converges" after one sweep** (it compares physics before
   Stage 1 with after Stage 2, which never touches them). Fixing it adds sweeps and changes
   individual-mode results.
7. **data-B1 / gui-B2 — `wavevector_q_list` is not sliced with phi filtering**, so the fit's
   `q_list[0]` can belong to a filtered-out row (matters for APS-old per-row q).
8. **optwrap-B4 — averaged-mode covariance s² uses the compressed dof**; rescaling changes every
   default `auto -> averaged` uncertainty by about 1%.
9. **optrest-B4 — recovery overrides the configured `x_scale`** (verbatim upstream port; honouring
   it moves the rtol=1e-10 goldens).
10. **R3-A1 — heterodyne fits with duplicate phi angles** report inconsistent per-angle diagnostics
    and mismatched engine x0/bounds (pre-existing). Choose one duplicate-angle policy.
11. **R4-A3 — zero-weight rows still count in the heterodyne joint covariance dof.**

Lower impact: optrest-B3 (out-of-core DOF under `constant`), config-B2 (equal bounds stay in the
optimizer vector), config-B9, core-B2 (64-substep quadrature), core-B3 (time-grid wiring, already
deferred), core-B9, optescape-B5 (multistart workers after timeout), cli-B9, cli-B10
(`from xpcsjax.cli import main` returns the module), ioviz-B7, ioviz-B8, data-B9 (`cache_directory`
is ignored), opthet-B6, opthet-B8, opthet-B13, optstrat-B7, optstrat-B13, R4-C2 (mixed-shape HDF5
bins in the allocation guard).

Rejected by both agents: config-B11, core-B4, optescape-B7, gui-B14, optwrap-B6, ioviz-B9,
data-B14, data-B15.

## How it was run

- Round 1: eleven one-shot `codex exec --sandbox read-only` reviewers (one per shard), eleven
  Claude verifiers that turned each claim into a failing test, then fixes by Claude plus five
  file-scoped fixer agents whose diffs Claude reviewed.
- Rounds 2-4 and the closing check: the live Codex pane (`t1k`) reviewing each fix commit.
- Three fixes Claude made were themselves wrong and were caught by Codex (the heterodyne
  angle-order restore with duplicate angles and with a shared weight slab; a half-fix for the
  streaming refinement, since reverted).
