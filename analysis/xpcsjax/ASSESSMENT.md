# xpcsjax — Modernization Assessment

Scope: entire repository at `/home/wei/Documents/GitHub/xpcsjax` (commit `a669fee`, branch `main`, clean tree).
Tooling: `scc`/`cloc`/`lizard` not installed on this host — used `find`+`wc -l` for LOC and a
decision-keyword grep (`if`/`for`/`while`/`except`/`elif`) as a complexity proxy. `pip-audit` was
available for the dependency scan. Note this so figures are reproducible with different tooling.

## Executive Summary

xpcsjax is a 138K-LOC, actively maintained Python 3.12+ / JAX scientific package (X-ray Photon
Correlation Spectroscopy NLSQ fitting) with a CLI, a PyQt/PySide6 GUI, and 247 test files gated by
CI (ruff + mypy + pytest). It is **not legacy code** — uv-managed, single dependency-injection
lazy-export surface, zero import cycles, 0 critical/high security findings, and a documented,
deliberate scope boundary (NLSQ-only, Bayesian/MCMC explicitly out of scope and actively guarded
against). The headline recommendation is **incremental refactor-in-place**, not a rewrite: pay down
three god-files at the optimization engine's core seams, resolve one deprecated dependency
(`jaxopt`), and close a handful of documented-but-deferred parity gaps — none of it blocks
continued feature work.

## Remediation Status (2026-07-20)

**PR review addendum:** a 4-agent review (code-reviewer, pr-test-analyzer, silent-failure-hunter,
comment-analyzer) of the remediation commit itself found — and this PR then fixed — 4 real issues
introduced by the remediation, all independently confirmed by 2-3 agents each:
1. **Bug:** the Debt #3 retry loop compounded its per-attempt config multipliers quadratically
   instead of applying them fresh from the original config each attempt (e.g. by the 3rd retry the
   learning rate had collapsed to 0.5⁶ instead of the documented 0.5³). Fixed by snapshotting base
   config values and assigning (not `*=`) each retry.
2. **Regression:** the retry loop's `try` block was too broad — it wrapped result-extraction/
   info-building in addition to the optimizer call, so an unrelated post-processing bug would get
   mistaken for an optimizer failure and retried up to 3x against a large dataset. Fixed by
   narrowing the `try` to only the `optimizer.fit()` call.
3. **False documentation claim:** the jaxopt→optimistix CLAUDE.md note (written for this PR's own
   Doc Gap 1) claimed `optimistix` was "already an xpcsjax dependency" — verified false
   (`import optimistix` fails, absent from `pyproject.toml`/`uv.lock`). Corrected in both CLAUDE.md
   and this file.
4. **Comment referencing a non-existent method:** `adapter_base.py`'s shared-constant comment named
   `NLSQWrapper.fit_nlsq_wrapper()`, which doesn't exist (the real method is `fit()`). Corrected.

Also added: 4 new regression tests closing coverage gaps the same review surfaced (config-magnitude
assertion, non-recoverable-exception passthrough, retry-count/error-context assertion in the
exhaustion test, and a test for the Debt #1 `AttributeError` fallback path that had zero coverage
either before or after the original fix). Softened one CLAUDE.md historical claim (god-files
extraction precedent) per a LOW-confidence comment-accuracy finding.


All findings below were triaged and actioned in a follow-up pass. Status per item:

| # | Finding | Status |
|---|---|---|
| Debt #1 | Silent bare `except Exception` in `heterodyne_result_builder.py` | **Fixed** — narrowed to `AttributeError` + `logger.debug` |
| Debt #2 | `jaxopt` deprecated, no migration timeline | **Documented** (CLAUDE.md) — migration to `optimistix` is real work, not mechanical; not attempted here |
| Debt #3 | Unimplemented T030 retry in hybrid-streaming optimizer | **Fixed** — wired `HybridRecoveryConfig`'s progressive-recovery retry (3 tests added) |
| Debt #4 | Duplicated `per_angle_scaling=False` rejection message | **Fixed** — extracted `PER_ANGLE_SCALING_REMOVED_MSG` to `adapter_base.py` |
| Debt #5 | Three-way joint-fit result-assembly split (`TODO(C3)`) | **Documented** (CLAUDE.md) — real architectural convergence work, not attempted here |
| Debt #6 | Three god-files at the engine's central seams | **Documented** (CLAUDE.md) — assessment itself says this is not a blanket-refactor candidate |
| Debt #7 | `core.py` `type: ignore` repetition (46 sites) | **Partially fixed** — removed one dead symbol (`WrapperOptimizationResult`, never referenced anywhere), chain-assigned the two genuinely multi-symbol blocks (MultiStart, CMA-ES). A generic dynamic-import helper was considered and rejected: it would route through `importlib`/`globals()`, breaking the static import-graph analysis this repo's own tooling (graphify, mypy) depends on. |
| Debt #8 | Golden parity gate never runs in CI | **Investigated, not automated** — the gated test files themselves document (verified 2026-06-07) that the `rtol=1e-10` value-compare is CPU-microarch-specific and fails on every GitHub-hosted Ubuntu runner. Adding a hosted nightly job would be permanently red with zero signal. Documented in CLAUDE.md as a maintainer-local pre-release check instead; only revisit if a self-hosted runner pinned to the goldens' recording machine becomes available. |
| Debt #9 | Broad `except Exception` (93 sites) | **Not attempted** — assessment explicitly flags this as not a blanket fix; a lint-rule policy change affecting 93 call sites needs its own scoped review, not a drive-by edit |
| Debt #10 | `fourier` mode permanently excluded from shared engine | **Documented** (CLAUDE.md) — flagged as needing an explicit owner decision, no code change |
| Security | `mistune`/`pillow`/`setuptools` transitive CVEs | **Fixed** — floors bumped in `pyproject.toml`, `uv.lock` regenerated, `pip-audit` now reports zero known vulnerabilities |
| Doc gap 1 | No `jaxopt` migration note | **Fixed** — added to CLAUDE.md |
| Doc gap 2 | No owner/decision record for `fourier` gap | **Fixed** — added to CLAUDE.md |
| Doc gap 3 | Three-way consolidation not documented | **Fixed** — added to CLAUDE.md |
| Doc gap 4 | `graphify-out/wiki/index.md` referenced but missing | **Not a bug** — re-checked; the repo's own `CLAUDE.md` already phrases it conditionally ("If `graphify-out/wiki/index.md` exists, navigate it") — the original finding overstated this |
| Doc gap 5 | No extraction-convention note for god-files | **Fixed** — added to CLAUDE.md |

## System Inventory

| Metric | Value |
|---|---|
| Total LOC (py+yaml+toml+md+rst+sh, excl. graphify-out/.venv/.git) | ~189,900 |
| Python LOC | 138,131 across 471 files |
| Docs (md+rst) | 47,000 LOC across 160 files |
| Config (yaml/yml/toml) | 4,590 LOC across 13 files |
| Test files | 247 (`tests/{benchmarks,characterization,cli,config,core,data,gui,heterodyne,integration,optimization,parity,property,runtime,service,viz}`) |
| CI workflows | `ci.yml`, `codeql.yml`, `docs.yml`, `mirror.yml`, `release.yml` |
| Package manager | `uv` (`uv.lock` is source of truth) |
| Python requirement | `>=3.12` |

**Tech fingerprint:**
- **Language/runtime:** Python 3.12+, no other languages in the source tree.
- **Compute core:** JAX 0.8.2+ (`jax`, `jaxlib`), `jaxopt` (L-BFGS warm-start — **deprecated
  upstream**, see Technical Debt #2), `evosax` (CMA-ES), `interpax` (JIT-safe interpolation),
  `nlsq>=0.6.10,<1.0` (trust-region solver, external).
- **Data:** `h5py` (HDF5 correlation data), `numpy`, `scipy`, `scikit-learn` (GaussianMixture in the
  anti-degeneracy controller).
- **GUI:** PyQt/PySide6, decoupled from the JAX-bearing fit worker via a picklable event contract
  (`xpcsjax/service/events.py`) — the GUI process itself never imports JAX.
- **Build/manifest:** `pyproject.toml` (uv-managed), `Makefile` (test/lint/format/verify targets),
  `uv.lock` pinned.
- **Integration points:** none external (no network service, no DB, no message queue) — this is a
  local single-user CLI/GUI scientific tool; the fit worker communicates with the GUI over an
  in-process/subprocess event contract, not a network boundary.
- **Test signal:** strong — 247 test files, domain-sharded (`make test-core`, `test-optimization`,
  `test-heterodyne`, etc.), a synthetic golden-parity suite (`rtol=1e-10`) pinning the shared
  optimization engine seam, and a property-test suite (Hypothesis).

## Architecture-at-a-Glance

12 functional domains, verified against real import statements (not the graph tool's 471
auto-generated low-cohesion communities, which were consulted for orientation but too fine-grained
to use directly for domain grouping — see `graphify-out/GRAPH_REPORT.md`'s own God-Nodes and
Hyperedges sections for corroboration: `ConfigManager` 182 edges, `AnalysisMode` 161,
`OptimizationResult` 142).

| Domain | Key files | Responsibility | Depends on |
|---|---|---|---|
| Config & Parameter Registry | `config/manager.py`, `parameter_registry.py`, `parameter_manager.py`, `templates/*.yaml` | Single source of truth for parameter names/bounds/physics constraints per `AnalysisMode` | core, utils |
| Physics Core (kernels) | `core/jax_backend.py`, `homodyne_model.py`, `heterodyne_jax_backend.py`, `heterodyne_physics_kernel.py` | JAX g1/g2/C2 kernels for `HomodyneModel`/`HeterodyneModel` | config, utils |
| Data Ingestion & QC | `data/xpcs_loader.py`, `preprocessing.py`, `quality_controller.py`, `memory_manager.py` | Load/normalize APS correlation data, diagonal correction, quality gating, memory-aware chunking | core, utils |
| NLSQ Optimization Engine | `optimization/nlsq/{core,wrapper,adapter,model_adapter}.py`, `strategies/*.py` | Strategy layer around upstream NLSQ's trust-region solve: memory routing, chunking, multistart, CMA-ES | config, core, device |
| 5-Layer Anti-Degeneracy Defense | `optimization/nlsq/{per_angle_mode,hierarchical,adaptive_regularization,gradient_monitor,shear_weighting}.py` | Feature-gated degeneracy mitigation shared by homodyne/heterodyne; L5 is `laminar_flow`-only | NLSQ Optimization Engine |
| Heterodyne Optimization Subsystem | `optimization/nlsq/heterodyne_{core,adapter,engine_route,stratified_ls}.py` | Two-component fit dispatch, per-angle-mode resolution, joint global escapes | NLSQ Engine, Anti-Degeneracy, Physics Core |
| Device / Runtime Environment | `device/{config,cpu}.py`, `xpcsjax/__init__.py` (XLA env pre-JAX-import), `runtime/shell/*` | CPU threading/XLA flags, concurrency-aware worker detection, shell-completion generation | utils |
| CLI | `cli/{main,args_parser,commands,config_generator,optimization_runner,plot_dispatch}.py` | Argparse entry points, config generation, headless load→fit→save→plot orchestration | config, optimization (via service), viz |
| Service Layer (process boundary) | `service/{events,fit,data,config,plots,persist}.py` | Decouples the JAX-bearing fit worker subprocess from the JAX-free GUI/CLI processes via a picklable event contract | optimization, data, config |
| GUI | `gui/{app,views/main_window}.py`, `gui/ipc/*`, `gui/project/*` | PyQt/PySide workbench: project management, IPC to fit-worker subprocess, result presentation | service |
| I/O & Result Persistence | `io/{json_utils,nlsq_writers}.py` | JSON-safe serialization, NLSQ result writers | utils |
| Visualization | `viz/{nlsq_plots,diagnostics,datashader_backend}.py` | Fit/residual/diagnostic plotting; `generate_nlsq_plots` is the one viz symbol re-exported top-level | config, core, io, utils |

Diagram: see `ARCHITECTURE.mmd`. **Zero import cycles detected** (both by the graph tool and by
direct verification). One dangling reference flagged: `core/homodyne_model.py:27` contains a
`>>> from xpcsjax.viz import ...` **docstring example**, not a real import — `core` has no runtime
dependency on `viz`.

## Production Runtime Profile

No telemetry available — this is a local CLI/GUI scientific tool with no APM/observability
integration and no supplied batch-job logs. Skipped per Step 4.

## Technical Debt

Ranked by remediation value; full evidence trail in the underlying analyst reports.

1. **Silent bare `except Exception` degrades a user-facing statistic with no logging.**
   `xpcsjax/optimization/nlsq/heterodyne_result_builder.py:667-670` — DOF calculation falls back to
   `_n_physics = None` on *any* exception, not just the expected `AttributeError`, with no log call.
   Fix: narrow to `except AttributeError` + add `logger.debug(...)`.
2. **Deprecated, unmaintained dependency (`jaxopt`) with a globally-suppressed warning.**
   `pyproject.toml:14,123-126`, `xpcsjax/optimization/nlsq/hierarchical.py:56-61` — used for L2's
   bounded L-BFGS warm-start; upstream is unmaintained. Planned replacement is `optimistix`, but
   (corrected during PR review — the original claim that it's "already a dependency" was false,
   verified by `import optimistix` failing) it is **not currently installed**; migration would need
   to add it as a new dependency first. No tracking issue exists for this migration.
3. **Unimplemented retry logic left as a TODO on the hot-path streaming optimizer.**
   `strategies/hybrid_streaming.py:276` (`# T030: TODO - Implement 3-attempt retry`) — any transient
   failure falls back immediately to the slower plain streaming optimizer (10x+ slower, may miss
   shear parameters) instead of retrying.
4. **Verbatim-duplicated rejection message across two call paths.**
   `optimization/nlsq/adapter.py:1263-1266` vs `wrapper.py:850-859` — the legacy
   `per_angle_scaling=False` rejection guard (intentional, keep it) is copy-pasted rather than
   shared, so wording/behavior can silently drift.
5. **Function-local import flagged by the code's own TODO as intentionally unconsolidated.**
   `optimization/nlsq/heterodyne_core.py:3427-3434` — three joint-fit code paths (`constant`,
   `individual`, `averaged`) haven't converged onto one shared result-assembly helper yet; tracked
   as the concrete next unit of the documented "procedural parity" work in `CLAUDE.md`.
6. **Three god-files at the optimization engine's central seams.**
   `heterodyne_core.py` (4,606 lines, ~340 branch points), `wrapper.py` (4,174 lines, ~427),
   `core.py` (2,684 lines, ~281) — together ~11,464 lines / ~621 decision points. Any new mode,
   escape path, or streaming tier touches these. **Not** a wholesale-refactor candidate — the
   `rtol=1e-10` golden parity tests make broad restructuring high-risk for low near-term payoff;
   recommend incremental, parity-gated extraction one PR at a time (same pattern already used for
   `heterodyne_stratified_ls.py`/`heterodyne_result_builder.py`).
7. **`core.py` carries the densest concentration of `# type: ignore` suppressions (46).**
   `core.py:60-203` — ~20 repetitions of the identical "optional dependency → `None` sentinel →
   `type: ignore[assignment]`" idiom. Legitimate pattern, but a shared `_optional_import()` helper
   would collapse the repetition without behavior change.
8. **The strict-numeric CPU-microarch-sensitive parity gate never runs in default CI.**
   `tests/parity/test_homodyne_engine_preservation.py:69-72` — only forced via
   `XPCSJAX_RUN_ENGINE_PARITY=1` (`make test-full-local`), which is a manual local target. No
   scheduled CI job exercises it, so a kernel-legitimacy regression in the shared engine seam is
   only caught if a developer remembers to run it.
9. **Broad `except Exception` is the dominant error-handling idiom (93 occurrences across
   data/optimization/core/config/service/cli).** Most carry a "best-effort, must never break a fit"
   rationale — a deliberate resilience choice for a long-running fit pipeline — but few emit a
   diagnostic on the swallow path, so a genuine bug is structurally indistinguishable from an
   expected "feature unavailable" condition. Not a blanket fix; recommend a lint rule requiring
   every bare `except Exception` to either narrow the type or carry a one-line rationale comment.
10. **Homodyne/heterodyne procedural convergence is intentionally incomplete for one sub-mode.**
    The `two_component` in-memory path shares the `StratifiedResidualFunctionJIT` engine with
    homodyne for most per-angle modes, but `fourier` mode is permanently excluded
    (`NotImplementedError`) per `CLAUDE.md`. Not a bug — flagged so it doesn't silently become
    permanent by default without an explicit owner decision.

## Security Findings

No CWE-078 (injection), CWE-089 (SQLi), CWE-502 (insecure deserialization), CWE-798 (hardcoded
credentials), or CWE-862 (missing auth) findings. This is a local single-user tool with no network
listener — YAML config loading uses `yaml.safe_load` only, path-save operations route through a
previously-hardened `path_validation.py` (symlink-aware containment checks, `..`/null-byte
rejection), the sole `subprocess.run` call has no shell/no user-controlled argv, and worker-pool
IPC uses `multiprocessing.spawn` between trusted same-machine processes (not untrusted
deserialization). `cloudpickle` is a declared dependency but appears unused anywhere in the tree —
hygiene note, not a vulnerability.

| CWE | Severity | Location | Description | Remediation | Status |
|---|---|---|---|---|---|
| CWE-1104 | Low | `pyproject.toml` (docs extra → `mistune`) | `mistune==3.2.1` had 10 known advisories, fixed in `3.3.0`; docs-build-only, never parses attacker-supplied Markdown at runtime | Bump to `mistune>=3.3.0` | **Fixed** — floor added, resolved to `3.3.3` |
| CWE-1104 | Low–Medium | `pyproject.toml` (`matplotlib` → transitively `pillow`) | `pillow==12.2.0` had 8 advisories, fixed in `12.3.0`; xpcsjax only writes images via matplotlib, never decodes untrusted images | Bump `pillow>=12.3.0` | **Fixed** — floor added, resolved to `12.3.0` |
| CWE-1104 | Low | `pyproject.toml` (dev extra) | `setuptools==82.0.1` had 1 advisory, fixed in `83.0.0`; build-time only | Bump floor | **Fixed** — floor added, resolved to `83.0.0` |

`pip-audit` now reports **zero known vulnerabilities** against the locked environment.

No credentials found in source; `analysis/.gitignore`/`SECRETS.local.md` were not created since
nothing needed quarantining.

## Documentation Gaps

Top 5 undocumented-or-under-documented behaviors a new engineer would need explained, beyond the
already-extensive `CLAUDE.md`:

1. **No migration plan or timeline for the `jaxopt` deprecation** (Technical Debt #2) — `CLAUDE.md`
   documents many other deliberate v0.1 scope cuts in detail but is silent on this one, even though
   the dependency itself is unmaintained upstream.
2. **The `fourier` per-angle-mode engine-route gap (Technical Debt #10) has no named owner or
   decision record** — `CLAUDE.md` documents it as a known exclusion but not whether it's permanent
   or pending.
3. **The three-way joint-fit result-assembly split (`constant`/`individual`/`averaged`,
   Technical Debt #5) isn't called out as an in-progress consolidation anywhere in `CLAUDE.md`** —
   only discoverable by reading the `TODO(C3)` comment in `heterodyne_core.py`.
4. **`graphify-out/wiki/index.md` is referenced by CLAUDE.md's graphify section but does not
   exist** — only `graphify-out/GRAPH_REPORT.md` is present, so the documented navigation path is
   stale/broken for anyone following it literally.
5. **No CI-level enforcement narrative for the god-files (Technical Debt #6)** — `CLAUDE.md`
   documents *what* each of `wrapper.py`/`heterodyne_core.py`/`core.py` do in exhaustive behavioral
   detail, but nothing documents the size/complexity risk or the incremental-extraction convention
   already used for `heterodyne_stratified_ls.py`/`heterodyne_result_builder.py`, so a new
   contributor has no signal that further extraction is the expected pattern rather than "leave it
   alone."

## Relative Scale

- **Python source:** 138.1 KSLOC across 471 files.
- **COCOMO-II basic index** (nominal scale factors, computed as `2.94 × KSLOC^1.10` since `scc` was
  unavailable): **2.94 × 138.131^1.10 ≈ 665**.

This is a **relative complexity/scale index only**, useful for ranking this system against others
in a portfolio — **it is not a timeline, cost, or person-month estimate.** The formula assumes
traditional human-team productivity curves, which agentic transformation does not follow. No
schedule or dollar figure should be derived from it.

## Recommended Modernization Pattern

**Refactor (in-place)** — routes to `/modernize-uplift`.

xpcsjax does not fit the legacy-modernization mold this command is normally aimed at: it is already
on a current, actively-supported stack (Python 3.12+, JAX 0.8+), has strong CI gates, zero critical
findings, and a deliberately scoped architecture with documented intentional cuts. The debt found
(§ Technical Debt) is ordinary maintenance-grade: one deprecated dependency, three large-but-cohesive
files at natural extraction seams, a handful of TODO-tracked gaps, and error-handling hygiene. None
of it requires a same-stack version bump, a cross-stack rewrite, or a greenfield rebuild — the right
unit of work is a sequence of small, parity-gated refactor PRs against the existing test suite
(the golden `rtol=1e-10` parity tests already provide the safety net this kind of incremental
extraction needs). `/modernize-uplift` is the closest-fitting downstream command for tracking that
kind of bounded, in-place cleanup; `/modernize-transform` (cross-stack rewrite) and
`/modernize-reimagine` (greenfield rebuild) are not warranted.
