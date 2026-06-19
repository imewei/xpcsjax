# xpcsjax GUI Analysis Workbench — Design

**Date:** 2026-06-18 (revised 2026-06-18 — dedup, F1 root-cause correction, IPC/error-handling gaps closed, v1 scope decisions)
**Status:** Approved design (pre-implementation). No code written.
**Scope:** Design only — this document is the spec that a subsequent implementation
plan (`writing-plans`) will decompose. Nothing here is implemented yet.

---

## At a glance

- **What:** a PySide6 desktop workbench over the existing JAX-native, **NLSQ-only**
  XPCS engine — load HDF5 → config → fit → live-monitor → results → export.
- **Three processes, one-way dependency (Qt → never → JAX):** a JAX-free **GUI
  process**, **spawned worker processes** that run fits, and a shared headless
  **core-service** (`xpcsjax/service/`) called by both the CLI and the GUI worker
  (§2). Workers stream `FitEvent`s over a bounded `multiprocessing.Queue`; results
  go to disk, so only paths cross the boundary.
- **The load-bearing constraint:** JAX must never import in the GUI process — and
  this is **not free today**: `config/__init__.py`'s eager re-exports pull JAX
  (§1). A Phase-1 lazy-import fix + a submodule-level import-graph test enforce it.
- **Six shippable phases** (§10): 1 core-service extraction + JAX-boundary fix →
  2 single-dataset happy path + full IPC robustness contract → 3 rich live
  diagnostics → 4 project model + 2-run comparison → 5 interactive plots →
  6 distributable hardening + packaging.
- **v1 cut-line:** the Config form-editor, Data HDF5/preview tab, Fit
  resolved-settings, and the separate Inspector dock are **deferred to Plan I**
  (§5); v1 comparison is a **2-run parameter/χ² table**, not interactive overlays.
- **11 implementation plans** are drafted against this spec (1A, 1B, B2, C, D, E,
  E2, F, G, H, I); §14 records the cross-plan coherence review.

---

## 1. Purpose & decisions

A desktop **full analysis workbench** for xpcsjax: browse/load HDF5 XPCS data,
build & validate configs, launch NLSQ fits, watch them with rich live
diagnostics, explore results interactively, and export publication figures —
all over the existing JAX-native NLSQ-only engine.

Locked-in decisions (from brainstorming):

| Decision | Choice |
|---|---|
| **GUI scope** | Full analysis workbench (load → config → fit → monitor → results → export) |
| **Live monitoring** | Rich live diagnostics **and** a structured log tail during fits |
| **Audience** | Maintainer + collaborators first; architected to harden into a distributable later (phased) |
| **Session model** | Project of multiple datasets — sidebar list, queued fits, side-by-side comparison |
| **Execution model** | **Out-of-process worker streaming events** (Approach B) over a `multiprocessing.Queue` |
| **Reuse boundary** | Extract a **headless core-service layer** from the CLI's orchestration; CLI and GUI worker both call it |

Non-negotiable constraints inherited from the project:

- **NLSQ-only.** No Bayesian/MCMC/CMC pathway is added or wired (out of scope by charter).
- **Qt is view-only.** Numerical logic stays in JAX / the core-service; widgets hold no business logic.
- **JAX never imported in the GUI process.** XLA env setup, cold-compile cost, and OOM risk live only in worker processes. **Empirically verified gap (2026-06-18 probe, reconfirmed in §14):** importing `xpcsjax.config` (or *any* of its submodules), `xpcsjax.cli.config_generator`, `xpcsjax.cli.data_pipeline`, or `xpcsjax.data` *today* pulls `jax`+`jaxlib` into the process. **Root cause: `xpcsjax/config/__init__.py`'s eager re-exports** — importing *any* `config` submodule runs the package `__init__` first, whose unconditional re-exports (six submodules: `manager`, `parameter_manager`, `parameter_registry`, `parameter_space`, `physics_validators`, `types`) pull in `parameter_manager`, whose module-level `parameter_manager.py:21` (`from xpcsjax.core.physics import …`) drags in JAX. So the leak fires on **every** submodule import (even an innocent `import xpcsjax.config.types`), not only `import xpcsjax.config`. Bare `import xpcsjax` stays JAX-free thanks to the top-level lazy `__getattr__`, but direct submodule imports bypass it. This boundary is therefore **not free** — the Phase-1 fix must make the `config/__init__.py` re-exports lazy (and/or defer the `:21` import); whether deferring `:21` alone fully closes it must be **proven by a submodule-level import-graph test** (other re-exported submodules may carry their own transitive JAX edges), see §3, §9, §10. Until then, in-process config validation is impossible JAX-free and must run in a worker.
- **Parity oracles untouched.** The `rtol=1e-10` homodyne baselines and the no-worse heterodyne contracts must remain green; the core-service extraction is behavior-preserving.

---

## 2. Architecture & layering

Three processes, three layers, one-directional dependencies (Qt → never → JAX):

```
┌─────────────────────────────────────────────────────────────┐
│  GUI process  (PySide6, NO jax import)                        │
│  xpcsjax/gui/                                                 │
│   ├─ views/       panels & widgets (pure Qt, view-only)       │
│   ├─ models/      Qt models, project/session state            │
│   ├─ controllers/ glue: launch fits, route events             │
│   └─ ipc/         worker handle + event-stream reader thread  │
└───────────────▲───────────────────────┬─────────────────────┘
        Qt signals (UI thread)   spawn + Queue (events/results)
                │                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Worker process(es)  (imports jax/xpcsjax)                    │
│  xpcsjax/gui/worker.py — thin: calls core-service + emits     │
│        events to the Queue via an EventEmitter callback       │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Headless core-service  (NO argparse, NO Qt)                  │
│  xpcsjax/service/ — typed orchestration reused by CLI + GUI   │
│   load_data() · run_fit(on_event=...) · save · make_plots()   │
└─────────────────────────────────────────────────────────────┘
```

**Keystone:** the numerical path has exactly one implementation. The CLI's
`commands.py` / `main.py` become thin argparse→service adapters, mirroring the
GUI worker as a thin IPC→service adapter. The existing CLI test suite therefore
guards the refactor.

---

## 3. Headless core-service (`xpcsjax/service/`)

Lifts the argparse-free essence out of `xpcsjax/cli/`. The CLI orchestration is
already cleanly separated; the extraction is mostly **decoupling functions from
`argparse.Namespace`** into typed parameters.

| Service module | Public function (typed) | Wraps / lifts from |
|---|---|---|
| `service/data.py` | `load_dataset(config, *, phi_subset=None) -> XpcsDataset` | `cli/data_pipeline.py::load_and_validate_data`, `resolve_phi_angles` |
| `service/fit.py` | `run_fit(dataset, config, *, overrides, on_event=None) -> OptimizationResult \| list` | `cli/optimization_runner.py::run_nlsq`, `apply_cli_overrides` |
| `service/persist.py` | `save_results(...)` (re-export) | `cli/result_saving.py::save_results` |
| `service/plots.py` | `generate_plots(result, dataset, config, out_dir, backend) -> list[Path]` | `cli/plot_dispatch.py::_generate_post_fit_plots` |
| `service/config.py` | `load_config` (Phase 1); build / validate / template facade (Plan I) | `config/ConfigManager`, `cli/config_generator.py` |

- `on_event: Callable[[FitEvent], None] | None` is the **live-diagnostics seam**.
  Default `None` reproduces today's behavior exactly.
- **Refactor discipline:** behavior-preserving extraction — only the parameter
  *shape* changes (typed args instead of `Namespace`), not the values passed or
  the numerics. Parity oracles stay untouched.
- **Equivalence tests (F8):** the parity oracle proves *numerics*, not *input
  resolution*. Add adapter/equivalence tests that assert the resolved service
  inputs (merged config dict, applied overrides + precedence, normalized paths,
  default fill-in) match what the current CLI produces for the same argv — the
  argparse→typed-params change must not silently shift a default or override
  order.
- **Event schema is JAX-free (F1/F3):** the `FitEvent` dataclasses live in their
  own dependency-light module (e.g. `xpcsjax/service/events.py`) that imports
  **no** `jax`, `xpcsjax.core`, Qt, or h5py — the GUI imports it directly, so it
  must stay on the JAX-free side of the boundary. Only plain
  dataclasses/paths/primitives are defined there.
- **Phase-1 JAX-boundary fix:** to make in-process *config* validation feasible
  JAX-free, the enforceable requirement is that **direct imports of the config
  submodules are JAX-free** — the `__init__` re-export chain must not pull JAX on
  *any* config submodule import. The submodule-level import-graph test (§9)
  **proves** the boundary is actually closed; that test, not a specific mechanism,
  is the contract. The defer pattern is lazy-inside-function (mirroring the
  existing `manager.py:467` pattern) or moving `ValidationResult`/
  `validate_parameters_detailed` to a JAX-free module; Phase 1 also audits the
  `xpcsjax.data` loader for its own JAX edges. If a closed boundary proves
  infeasible within Phase 1, the fallback (per §8) is to run validation in a
  short-lived worker.
  > **Implementation note (2026-06-19 audit):** the boundary was closed by
  > deferring the transitive `core.physics` import (`parameter_manager.py:21`),
  > **not** by making `config/__init__.py` lazy/minimal — its re-exports stay
  > eager but no longer transitively pull JAX, and the §9 import-graph test
  > confirms `xpcsjax.config` and its submodules import without `jax` in
  > `sys.modules`. The original "must make `__init__` lazy" wording over-specified
  > one mechanism; the test-proven outcome is what matters and is satisfied.
- **`service/config.py` build/validate/template facade is deferred to Plan I.**
  Phase-1 ships only `load_config` (mode + output-dir overrides); the
  build/validate/template surface listed in the table above is **not** delivered
  in Phase 1 (tracked follow-on — Plan I's config form editor depends on it, so
  it lands there, not silently).

---

## 4. Process & IPC model

- **Spawn, not fork** (JAX + fork → deadlock). One worker process per fit.
- **Worker pool** bounds concurrency; default low (these fits are RAM-heavy —
  mirrors the project's OOM serial-routing lesson). Configurable.
- **Event channel:** a **bounded** `multiprocessing.Queue`. Every event carries
  `run_id` + a monotonic per-run `seq` (F5) — multiple workers share one reader,
  so events are demultiplexed by `run_id` and **cross-run global ordering is not
  assumed** (only per-run ordering, via `seq`). Worker pushes `FitEvent`
  dataclasses:

  | Event | Payload (all also carry `run_id`, `seq`) |
  |---|---|
  | `Started` | mode, resolved settings summary |
  | `Iteration` | `n`, `ssr`, `chi2` (live convergence curve) |
  | `LayerStatus` | L1–L5 active/inactive + mode |
  | `Banner` | `text`, `kind` (CMA-ES escape, gradient collapse, …) |
  | `LogLine` | `level`, `msg` (forwarded from `logging`) |
  | `Finished` | `result_path` |
  | `Failed` | `traceback` |
  | `Died` (synthetic) | `exit_code`/`signal` — emitted by the **parent**, not the worker (F5) |

- **Log capture:** a `logging.Handler` in the worker forwards records as
  `LogLine` events — structured, not scraped (what Approach C could not do).
- **Result transport:** worker calls `save_results` (NPZ/JSON), which writes to a
  temp path then **atomically renames** into place (F9 — implemented in
  `service/persist.py` per Plan B2 Step 2b, so it is path-independent), and emits
  `Finished` only after the rename — so the GUI never reads a half-written file. Returns the
  **path**; the GUI loads it lazily. Large two-time arrays never cross the pickle
  boundary.
- **Cancellation lifecycle (F4):** not a bare `terminate()`. The parent runs
  `terminate()` → `join(timeout)` → `kill()` if still alive, then **tears down the
  GUI-side reader `QThread` first** — `requestInterruption()` + `wait()` so the
  reader is no longer blocked in `queue.get()` — *before* calling
  `queue.cancel_join_thread()` on the (now-suspect) channel (calling
  `cancel_join_thread()` while another thread is mid-`get()` is undefined and can
  leave the reader spinning or reading garbage). It then reaps the process and
  removes any partial output dir. Workers run in their own **process group** so a
  fit that spawned children is torn down wholesale. The reader thread is likewise
  joined on `shutdown()` / app-close, so Qt never destroys a still-running
  `QThread`. GUI marks the fit `cancelled`.
- **GUI-side reader:** a `QThread` drains the queue and re-emits Qt signals on
  the UI thread (widgets only ever touched on the UI thread).

### IPC robustness contract

- **Pickling contract (F3):** with `spawn`, everything crossing the boundary must
  be top-level-picklable. **In:** the worker entrypoint (module-level function),
  a plain config dict/path, a data path, typed overrides. **Out:** `FitEvent`
  dataclasses + result paths. **Never crossed:** Qt objects, bound methods,
  open HDF5 handles, live JAX arrays, `logging.LogRecord` objects (forward
  `level`+`msg` strings instead), or any GUI-side callback.
- **Backpressure (F2):** the queue is **bounded**. High-rate `Iteration` and
  `LogLine` events are **coalesced/throttled** (e.g. keep-latest + count, ~10–20 Hz
  to the UI) so a lagging reader can never balloon memory or block the solver.
  Terminal events (`Finished`/`Failed`/`Died`) are **never dropped** — they go
  through a path that does not coalesce.
- **Worker-death detection (F5):** the parent watches the process
  sentinel/exit-code rather than inferring death from a missing `Finished`. A
  non-zero exit or kill-signal synthesizes a `Died` terminal event (mapped to
  "killed — likely out of memory" in §8), so the GUI never waits forever.
- **Matplotlib backend (F6):** the worker calls `matplotlib.use("Agg")` before
  importing `pyplot`/`xpcsjax.viz`, so plot generation in the child never tries to
  grab a Qt backend. (The GUI-side export path is constrained the same way — see §6.)
- **Worker XLA env (F11):** the worker imports `xpcsjax` (whose `__init__.py`
  sets the XLA flags) **before** the first `jax` import, so env setup is not
  raced. v0.1 is CPU-only (`NLSQ_SKIP_GPU_CHECK=1`), so GPU preallocation knobs
  are moot now; the note is for forward-compat.
- **Cold-compile cost (F10):** spawning a fresh process per fit re-pays JAX import
  + XLA cold-compile each time. Accepted for v1 (correct + isolated). **Do not**
  reach for `JAX_COMPILATION_CACHE_DIR` — it is a documented dead-end here (the
  persistent cache writes 0 entries because data-in-closure blocks it). The only
  real mitigation is a **reusable/sequential daemon worker**, deferred as an
  optional later optimization, not a v1 requirement.
  **User-facing expectation:** the first fit after launch (and every fresh spawn)
  shows a multi-second pause while JAX imports + XLA cold-compiles — this
  dominates wall-time (per the project's perf findings) and is **expected, not a
  hang**. The UI must surface a "compiling…/starting…" state on `Started`-pending
  so a cold spawn never reads as a frozen app.

### Instrumentation seam (graceful degradation)

`run_fit(on_event=...)` threads a callback into the engine. Per-iteration
`Iteration` events depend on the solver exposing an iteration callback. If
`nlsq.CurveFit` does not expose one initially, **the early phase degrades
gracefully** to `LogLine` + `LayerStatus` + `Banner` events (already emitted as
log banners today); per-iteration SSR curves arrive in a later phase once the
seam is added. **No engine change is on the critical path for the first GUI
release.**

---

## 5. UI structure (PySide6 main window, dockable)

```
┌──────────────────────────────────────────────────────────────────┐
│ menubar · toolbar (New project · Add data · Run · Cancel · Export) │
├────────────┬─────────────────────────────────────┬───────────────┤
│ PROJECT    │   CENTER (tabbed)                    │  INSPECTOR    │
│ (dock,L)   │   ┌─Data──Config──Fit──Results──┐    │  (dock,R)     │
│ • dataset1 │   │                              │    │  params table │
│   └ fitA ✓ │   │  active tab content          │    │  + uncert.    │
│   └ fitB ⏳ │   │                              │    │  + diagnostics│
│ • dataset2 │   └──────────────────────────────┘    │  summary      │
├────────────┴─────────────────────────────────────┴───────────────┤
│ FIT MONITOR (dock, bottom): live SSR curve · L1–L5 chips · log tail│
└──────────────────────────────────────────────────────────────────┘
```

- **Project sidebar (dock, left):** tree of datasets → fits with status icons
  (queued / running / done / failed / cancelled). Multi-select → comparison view.
  **v1 comparison = a 2-run side-by-side parameter/χ² table** (Plan F);
  interactive run-vs-run *overlay* plots are deferred to a later phase (see §6, §10).
- **Center tabs** (per active dataset/fit):
  - *Data* — file picker + HDF5 metadata + two-time C₂ preview. The preview is
    **display-only downsampling**, explicitly labeled as such (F12) — it is
    rendering decimation, categorically distinct from the project's prohibited
    *analysis* downsampling; full resolution is available via the datashader
    fast-path (§6). No fit ever consumes the decimated preview.
  - *Config* — form editor over `parameter_registry` + mode templates, live
    validation (reuses `config_generator`); raw-YAML toggle for power users.
  - *Fit* — launch controls + resolved-settings summary; mirrors into Fit Monitor.
  - *Results* — interactive plots (residual map, per-angle views) + export button.
    **v1 fit-vs-fit comparison is the 2-run parameter/χ² table** (Plan F); linked
    run-A-vs-run-B *overlay* plots (shared color scales, linked axes) are a
    later-phase add (§10), not v1.
- **Inspector (dock, right):** parameter values + uncertainties + the structured
  anti-degeneracy diagnostics block.

- **Fit Monitor (dock, bottom):** the rich-live-diagnostics surface — live
  SSR/convergence curve (PyQtGraph), L1–L5 status chips, escape/collapse
  banners, scrolling log tail. Per-dataset tabs so a queued project shows each
  running fit.

> **Surface→plan mapping & v1 scope (added 2026-06-18 after holistic plan review).**
> The implementation plans build these surfaces incrementally; not all are v1:
> *Results* interactive plots → Plan G; *Fit Monitor* (status/log/SSR/chips/banners)
> → Plans D+E; project sidebar/comparison → Plan F. **Deferred to a dedicated
> "Plan I — workbench surfaces" (Phase 5/6 follow-on), NOT v1:** the *Config* tab's
> **form editor + live validation + raw-YAML toggle** (v1 uses the toolbar
> "Open Config" = pick an existing YAML), the *Data* tab's **HDF5-metadata +
> two-time C₂ preview** (needs a JAX-free metadata reader + the Plan-G `rasterize`
> helper), the *Fit* tab's **resolved-settings summary**, and the separate
> right-hand **Inspector dock** (v1 shows parameters + uncertainties in the Results
> panel — `ResultSummary` now retains `uncertainties` — and diagnostics in the Fit
> Monitor). This keeps the spec from promising surfaces no current plan builds;
> Plan I is the explicit home for them.

System-aware light/dark theming (per project UI guidelines).

---

## 6. Plotting stack

Split by purpose (per project UI guidelines):

- **PyQtGraph** — everything interactive/live: convergence curve, pan/zoom
  two-time maps, residual maps, per-angle overlays. Embeds natively in Qt.
- **Display rasterization for huge maps (resolved in Plan G):** large two-time /
  residual maps are decimated for display by a **numeric block-mean** downsample
  (`gui/views/raster.py`), preserving numeric values so the PyQtGraph `ImageItem`
  applies its own interactive colormap/levels. (The Datashader colored-image
  fast path in `xpcsjax/viz/datashader_backend.py` returns an RGB image — suited
  to the *static* publication figures below, not the interactive numeric maps —
  so it is not used for the in-app maps.)
- **JAX-free C₂ preview reader needs a shared layout contract (Plan I).** The
  Data-tab two-time preview reads the array directly with `h5py` (bypassing
  `xpcsjax.data` to stay JAX-free), so it cannot lean on the loader's axis
  normalization. A strided hyperslab that hardcodes `(n_phi, t, t)` will silently
  show the wrong axis — or always blank — for layouts that differ across the
  `aps_old` / `aps_u` formats and homodyne vs heterodyne files. The reader must
  consult a **single shared layout descriptor** (dataset path + axis order keyed
  by `data_type`) instead of duplicating layout knowledge, and must **return
  "preview unavailable"** rather than render a mislabeled map when the layout is
  unrecognized.
- **Matplotlib publication figures (resolved in Plan G):** the **fit worker**
  renders the publication PNG/PDF *during the fit* via `service.plots.generate_plots`
  (Agg-forced, using `xpcsjax.viz`; Plan B2/F6), writing them under
  `<result_dir>/plots`. **"Export figure" copies** those already-rendered
  artifacts to a user-chosen directory (`gui/export.py`, `shutil`, JAX-free) — it
  does **not** re-render (the GUI has no full-`OptimizationResult` loader; a
  re-render-at-new-params feature would add one, out of scope). The Agg /
  JAX-free boundary (F6) is satisfied because rendering happens only in the
  worker, never in the GUI process.
- **Export must respect the JAX/backend boundary (F6):** if export runs *in the
  GUI process*, it must force the `Agg` backend and `xpcsjax.viz` must reach it
  without importing `jax`/`xpcsjax.core` (verify via the import-graph test, §9);
  otherwise export is delegated to a short-lived worker (same isolation as a
  fit). Decision deferred to implementation, but the constraint is fixed here.

---

## 7. Project / session state

- A `Project` model (in-memory; serializable to a `.xpcsproj` JSON in a later
  phase) holding `Dataset` entries, each with `config`, `data_path`, and a list
  of `FitRun` (status, result-path, diagnostics, timestamp).
- Qt models (`QAbstractItemModel`) wrap it for the sidebar/inspector — clean
  view/model separation, no business logic in widgets.
- Fits are **append-only runs** per dataset, so re-fitting with tweaked config
  keeps history for comparison.
- **Stable identity from Phase 4 (F13):** datasets and runs get stable IDs and
  relative-path handling **designed in Phase 4** (when comparison views first
  depend on identity), even though `.xpcsproj` save/load only *lands* in Phase 6.
  Comparison must key off stable IDs, not list position or absolute paths.
- **Round-trip integrity for `.xpcsproj` (Phase 6):** save/load must persist —
  and restore — the dataset **`label`** (a user rename must survive; do not
  silently re-derive it from the config-filename stem on reload), each run's
  **status + `result_dir` + timestamp**, and the stable IDs. Per-run
  **diagnostics/summary are *not* duplicated into the project file** — they are
  reloaded from `result_dir` on open (the result NPZ/JSON is their source of
  truth). A reload that drops `label` is a regression (round-trip test); a
  `result_dir` that has since moved or been deleted flags the run **"result
  missing"** rather than raising (§8 dead-path).

---

## 8. Error handling & robustness

- **Worker failure ≠ app crash:** `Failed{traceback}` → fit marked failed in the
  tree, traceback in a detail pane; GUI stays alive (the point of out-of-process).
- **OOM/SIGKILL:** the parent's process-watcher synthesizes a `Died` event on a
  non-zero/kill-signal exit (F5) → reported as "killed (likely out of memory)"
  with a hint. Detection does **not** depend on a missing `Finished`, so the GUI
  never hangs on a silently-killed worker (echoes documented OOM reality).
- **Validation (corrected per the F1 probe):** config/data validation surfaces
  inline in the Config/Data tabs *before* launching a fit — but it is **not**
  free to run in-process, because the config/data modules pull in JAX today
  (§1). Two honest options, decided in Phase 1: **(A)** land the
  `config/__init__.py` lazy-import fix (§3) so *config* validation is genuinely
  JAX-free in-process; **(B)** run validation in a short-lived worker. Either
  way, no doomed full-fit spawn — but the spec no longer assumes free in-process
  validation.
- **Missing or moved artifacts (dead paths):** a `.xpcsproj` references configs,
  data, and `result_dir`s by (relative-resolved) path; any may have moved or been
  deleted since save. Load **does not** fail wholesale — it resolves each
  reference *eagerly at load* and surfaces a dead one as a clearly-flagged
  "missing" dataset/run in the tree, never as a deferred `FileNotFoundError`
  thrown by a lazy summary/preview read far from the load call. Likewise
  **"Export figure" on a run whose `<result_dir>/plots` is empty** (fit failed,
  was cancelled/killed, or plot generation errored) reports "no figures to
  export" rather than silently copying nothing.
- **Zombie/orphan cleanup (F7):** the GUI registers `atexit` + Qt `closeEvent`
  handlers that terminate all active workers (by process group), so closing or
  crashing the GUI never leaves JAX fits consuming CPU/RAM in the background.
- **Phased robustness:** maintainer phase tolerates rough edges; distributable
  hardening (friendly messages, guard rails, no tracebacks-in-face) is a later
  phase the architecture already supports.

---

## 9. Testing strategy

- **Core-service:** behavior-preserving extraction covered by the *existing* CLI
  tests (now exercising the service through the thin CLI adapter) + a few direct
  service tests. Parity oracles untouched.
- **Input-equivalence tests (F8):** assert resolved service inputs (merged config,
  override precedence, normalized paths, defaults) equal the CLI's for matched
  argv — guards the argparse→typed-params change against silent default/precedence
  drift the parity oracle would not catch.
- **Import-graph regression test (F1):** a test that imports the GUI package (and
  the `FitEvent` schema module) in a subprocess and asserts `"jax" not in
  sys.modules` — fails loudly if anyone reintroduces a JAX edge into the GUI/event
  layer. **It must assert the submodule-direct imports the GUI actually uses**
  (e.g. `import xpcsjax.config.parameter_registry`, `xpcsjax.config.types`), not
  merely `import xpcsjax.config` — because `config/__init__.py` runs on *every*
  submodule import, a test that only checks the top-level package can pass while a
  submodule-direct path (the way Plan I validates configs) still leaks JAX. A
  companion test pins that the Phase-1-fixed config-validation path stays JAX-free
  (or, under option B, that it is never imported in the GUI process).
- **IPC/event layer:** unit-test the event schema and worker→queue protocol
  **headlessly** (no Qt) — run a tiny fit in a worker, assert the event sequence.
- **GUI logic:** `pytest-qt` for controllers/models (launch → events → tree
  state) driven by a fake event stream — no real fits in UI tests. Widgets stay
  logic-free.
- **CI:** GUI tests run under `QT_QPA_PLATFORM=offscreen` (already required by CI
  facts). Heavy real-fit worker tests stay out of `-n auto` per the OOM
  serial-routing rule.

---

## 10. Dependencies, packaging & phasing

- **New deps** under an optional extra `xpcsjax[gui]` (headless installs stay
  lean): `PySide6`, `pyqtgraph`. Datashader already exists under `[viz-fast]`.
- **Entry point:** `xpcsjax-gui` console script. Maintainer phase:
  `uv run xpcsjax-gui`. Distributable packaging uses **PyInstaller** (the v1
  choice — `briefcase` is not pursued), deferred to Phase 6; the architecture is
  ready for it. The PyInstaller hidden-import list **must be guarded against drift
  from the `pyproject` deps** by an automated check (see Phase 6 + §12), because a
  frozen build that silently omits a `jax`/`numba`/`llvmlite` hidden import fails
  only at runtime, in the user's hands.

### Phasing (each phase shippable)

1. **Core-service extraction** (no GUI) — pure refactor, green CLI tests +
   input-equivalence tests (F8). Includes the **JAX-boundary fix** (make
   `config/__init__.py`'s re-exports lazy, not just defer the
   `parameter_manager.py:21` import; audit `xpcsjax.data`) and the
   **import-graph regression test** (F1, asserting submodule-direct imports).
   De-risks everything.
2. **Skeleton app + single-dataset happy path** — load → config → run
   (out-of-process, spinner + log tail) → results, with Matplotlib-export plots.
   Establishes the full IPC robustness contract (§4): bounded queue, pickling
   contract, hardened cancellation, process-watcher `Died` events, atomic writes.
3. **Rich live diagnostics** — SSR curve, L1–L5 chips, banners; add the engine
   iteration seam if needed.
4. **Project model** — multi-dataset sidebar, fit queue, comparison views.
   **Stable dataset/run IDs designed here** (F13).
5. **Interactive PyQtGraph plots** + datashader fast-path in-app.
6. **Distributable hardening** — friendly errors, project save/load (`.xpcsproj`
   round-trip with `label` + diagnostics, §7), **PyInstaller** packaging + the
   hidden-import drift guard. The later-phase comparison *overlay* plots (deferred
   from v1, §5/§6) land here or in Phase 5.

**Calibration verdict (b) — keep order 4 → 5.** The project model (Phase 4)
stays *before* rich interactive plots (Phase 5). The single-dataset loop is
already usable earlier — Phase 2 gives static Matplotlib result export and Phase
3 gives a minimal live SSR curve — so there is no need to pull rich *exploration*
forward. Rich result-exploration plot/widget APIs (linked axes, run-A-vs-run-B
overlays, shared color scales) are better shaped *after* the dataset/run
comparison model exists; building them first would force rework once comparison
lands.

---

## 11. Explicit YAGNI cuts (out of scope)

Deliberately excluded; do not add without a new decision:

- No plugin system.
- No remote / cluster / HPC execution from the GUI.
- No embedded Jupyter / scripting console.
- No real-time data-acquisition streaming.
- No Bayesian / MCMC / CMC anything (out of scope by project charter).

---

## 12. Key risks & mitigations

| Risk | Mitigation |
|---|---|
| Core-service extraction drifts numerics **or input resolution** | Behavior-preserving refactor gated by existing CLI + parity oracles **+ input-equivalence tests (F8)**; Phase 1 ships before any GUI. |
| Solver exposes no per-iteration callback | Graceful degradation to log/banner events; per-iteration curve deferred to a later phase. Not on critical path. |
| Worker OOM / SIGKILL | Out-of-process isolation + parent process-watcher synthesizes a `Died` event (F5) + explicit "killed (likely OOM)" reporting; low default pool size. |
| Large arrays over IPC | Results go to disk (atomic write, F9); only paths cross the boundary; previews are display-only decimated / datashaded (F12). |
| **GUI/JAX coupling — confirmed real today** | Empirically, `xpcsjax.config`/`xpcsjax.data` pull JAX (root cause: `config/__init__.py`'s eager re-exports pull in `parameter_manager.py:21`'s `core.physics` import on *every* submodule import). Mitigation: Phase-1 lazy-ify `config/__init__.py` re-exports + JAX-free `FitEvent` module + subprocess import-graph regression test asserting submodule-direct imports (F1). |
| Queue corruption / reader hang on cancel | Hardened cancellation lifecycle: terminate→join→kill + `cancel_join_thread` + process-group teardown (F4); bounded queue never drops terminal events (F2). |
| Cold-compile re-paid every spawn | Accepted for v1; `JAX_COMPILATION_CACHE_DIR` is a known dead-end here; reusable daemon worker is an optional later optimization (F10). UI shows a "compiling…/starting…" state so a cold spawn never reads as a hang (§4). |
| Frozen build omits a hidden import (runtime-only failure) | PyInstaller hidden-import list guarded against `pyproject`-dep drift by an automated check + a freeze-safety smoke test (Phase 6, §10). |
| `.xpcsproj` references a moved/deleted artifact | Paths resolved eagerly at load; dead references flagged "missing" in the tree, not thrown as a lazy `FileNotFoundError` (§8); round-trip persists `label` + diagnostics (§7). |

---

## 13. Review provenance & finding dispositions (2026-06-18)

This spec was reviewed by two independent external agents (`codex` and `agy`)
plus an empirical Claude verification pass. Their findings were consolidated to
13 items (F1–F13), adversarially triaged, and the load-bearing one (F1) was
**empirically confirmed by running the actual import graph**. Dispositions:

| # | Finding | Disposition | Addressed in |
|---|---|---|---|
| F1 | GUI imports JAX via config/data/event modules | **REAL — empirically confirmed + root-caused** (`config/__init__.py` eager re-exports pull `parameter_manager.py:21`'s `core.physics` edge on any submodule import) | §1, §3, §8, §9, §10, §14 (Phase 1) |
| F2 | Queue backpressure/coalescing | REAL | §4 (robustness contract) |
| F3 | Spawn/pickling contract | REAL | §4 (pickling contract) |
| F4 | Queue corruption on `terminate()` + cancel lifecycle | REAL | §4 (cancellation lifecycle) |
| F5 | Worker-death detection + `run_id`/ordering | REAL | §4 (event table + death detection), §8 |
| F6 | Matplotlib `Agg` in worker; export boundary | REAL | §4, §6 |
| F7 | Zombie workers on GUI exit | REAL | §8 |
| F8 | Extraction not purely a pure refactor | REAL | §3, §9 (input-equivalence tests) |
| F9 | Atomic result write | REAL — implemented in `service/persist.py` (temp + `os.replace`), Plan B2 Step 2b; `run_worker` inherits it transitively (emits `Finished` only after `save_results` returns) | §4 (result transport), Plan B2 |
| F10 | Cold-compile per spawn | REAL (documented tradeoff) | §4, §12 |
| F11 | Worker XLA env pinning | PARTIAL (mostly handled; GPU bits moot CPU-only) | §4 |
| F12 | Downsampled preview vs no-downsampling rule | DONE (viz≠analysis; `DataPanel` carries a visible "display-only downsampling" label) | §5 |
| F13 | Stable dataset/run IDs early | REAL | §7, §10 (Phase 4) |

**Calibration outcomes:** (a) phasing order retained, with Phase 1/2 scope
sharpened above; (b) **keep Phase 4 before Phase 5** (see §10 verdict); (c) **all
YAGNI items stay out** (§11) — both reviewers agreed.

*Note on method:* the final triage triangulated the two independent external
reviews + the empirical F1 probe + maintainer engineering judgment. F1 — the one
finding that could have been a false positive about the architecture — was the
one empirically proven, by running the actual import graph.


---

## 14. Holistic cross-plan review (2026-06-18)

After all 10 plans (1A, 1B, B2, C, D, E, E2, F, G, H) were drafted and individually review-hardened, a holistic pass (codex + a 4-agent Claude cross-plan workflow) checked the *set* for interface/edit-composition coherence. Verdict: **COHERENT-WITH-FIXES** — all applied:

- **E↔F live-diagnostics break (CRITICAL):** `FitController` (D/E) was replaced by `FitQueueController` (F), which dropped `Iteration`/`LayerStatus`/`Banner`/`LogLine`. F's queue now re-emits all per-run signals (`iteration_received`/`layer_status_received`/`banner_received`/`log_received`/`run_failed`, each `(run_id, …)`), and `MainWindow` routes them to the Plan-E/D monitor gated on the active run. F now lists E as a prerequisite.
- **Cancel→killed mislabel (HIGH):** D's `FitController` and F's queue track user-cancelled runs and map the resulting synthetic `Died` to `cancelled` (no spurious OOM dialog).
- **Lost `result_path` (HIGH):** the queue carries `result_path` on `run_finished`; `MainWindow` stores `run.result_dir` even when summary-load fails (viz/export/restore survive).
- **Spec surface gap (HIGH):** §5 now maps each surface to its plan and **routes the Config form editor, Data HDF5/preview tab, Fit resolved-settings, and the separate Inspector dock to "Plan I — workbench surfaces" (Phase 5/6)**; v1 uses toolbar Open-Config + params/uncertainties in Results.
- **MEDIUM:** worker passes `output_dir` to `load_config` (C); `EventEmitter` coalesces telemetry to ~20 Hz (C, §4); `ResultSummary` retains `uncertainties` (D); §6 export realigned to copy-artifacts; G's `write_viz_bundle` insertion pinned + worker-integration tested.
- **LOW/NIT:** G self-review wording, E2 worker.py file-list, make_plots gating note — all corrected.

**Plan I (now drafted — `plans/2026-06-18-gui-phase6-plan-i-workbench-surfaces.md`):** Config form editor + live validation + raw-YAML toggle (over a `service/config.py` validate/template facade), the Data tab (JAX-free HDF5-metadata reader + two-time preview via Plan-G `rasterize`), the Fit tab resolved-settings summary, and the Inspector dock. It was reviewed (codex NEEDS-REWORK) and reworked; its CRITICAL finding strengthened Plan 1A (config/__init__ must be JAX-free, not just parameter_manager.py:21).
