# Decision round: close the 3 remaining open items

You are one of three participants deciding how to close three open items in the xpcsjax repo:
`claude` (lead implementer, will implement whatever is decided), `codex` and `agy` (independent
reviewers). This is a DESIGN DECISION round, not a bug hunt. Give your own position; you will see
the others' positions in the next round and can then change yours.

## Rules
- READ ONLY. Do not edit any file except your own position file (named in the message that sent
  you here). Do not run git commands that change state. Do not run the test suite or `make`
  (another long test run is in progress and the machine is memory-limited). Small read-only probes
  with `uv run --no-sync python -c ...` from `$WT` are fine.
- Do NOT search the filesystem. Everything you need is at the exact paths below. Use `rg`/`grep`
  only inside `$WT`.
- `$WT` = `/home/wei/Documents/GitHub/xpcsjax/.claude/worktrees/codex-review-loop`
  (branch `fix/maintainer-decisions-2026-11-25`; your pane's own directory is the main checkout and
  does NOT have this branch's code — read code only under `$WT`).

## Context
- Project conventions and deliberate design decisions: `$WT/CLAUDE.md`.
- What was already decided for the other 32 items: `$WT/docs/reviews/2026-11-25-maintainer-decisions.md`.
- Standing rules that constrain the options:
  1. Fitted parameters and chi-squared on default paths are guarded by `rtol=1e-10` golden tests
     (`$WT/tests/parity/_golden/`). A fix that moves them needs the maintainer.
  2. No silent data loss or silent fallbacks; fail loudly at I/O boundaries.
  3. Smallest change that fully solves the problem; no speculative abstraction.
  4. The maintainer already chose: reporting-only DOF fixes are acceptable (uncertainties may
     change if parameters and chi-squared do not).

---

## Item 1 — R4-A3: zero-weight rows in the heterodyne joint covariance DOF

**Problem.** The heterodyne joint fits take the data degrees of freedom from the residual vector
LENGTH. A pixel with weight 0 still occupies a row (its residual is exactly 0), so it is counted as
an observation. With masked pixels, `s² = cost / (n_rows - p)` is too small and uncertainties are
too tight; with enough masked pixels a fit with zero true data DOF still reports a finite
covariance.

**Facts.**
- Both joint paths now share `_drop_penalty_rows_from_dof(covariance, uncertainties, *, n_valid,
  n_params, n_penalty_rows)` in `$WT/xpcsjax/optimization/nlsq/heterodyne_core.py` (search for
  that name). Callers pass `n_valid=int(data_only_residual.size)`.
- In `_build_joint_problem` (same file, search `weights_batch = jnp.stack`) the weights are a
  concrete `(n_phi, N, N)` array before the closure is built; unweighted fits use ones.
- The residual rows are selected inside `compute_multi_angle_residuals`
  (`$WT/xpcsjax/core/heterodyne_jax_backend.py`); the same selection is what a row count must use.
- The averaged path (`_fit_joint_averaged_multi_phi`) and the constant path build their own
  residuals; check whether they have the same issue.
- Homodyne stratified LS already solved the analogous problem with
  `_n_contributing_rows` in `$WT/xpcsjax/optimization/nlsq/strategies/stratified_ls.py`.

**Options.**
- A. Count contributing rows at problem-build time by applying the kernel's own row selection to
  the weight array (`weight > 0`), store it (e.g. `prob.meta["n_data_rows"]`), and use it as
  `n_valid` in every heterodyne joint result builder. Reduced chi-squared would use the same count.
- B. Count non-zero entries of the final data-only residual vector. One line, but an exact-zero
  true residual would be miscounted.
- C. Leave as is and document: zero weights are a mask, uncertainties are conditional on the row
  count.
- D. Something else (say what).

**Claude's recommendation:** A, applied to the individual, averaged and constant joint paths, and
used for BOTH the covariance DOF and reduced chi-squared so the two cannot disagree. Unweighted
fits (weights all ones, the default) are bit-identical.

---

## Item 2 — data-B9: `experimental_data.cache_directory` is ignored

**Problem.** All four shipped templates contain
`cache_directory: "./cache"   # Where to read/write npz caches. MUST be writeable.`
(`$WT/xpcsjax/config/templates/xpcsjax_laminar_flow.yaml`, search `cache_directory`), and
`ConfigManager` lists it as a known key with that default (`$WT/xpcsjax/config/manager.py`). The
loader never reads it: it uses `cache_file_path`, defaulting to the DATA folder
(`$WT/xpcsjax/data/xpcs_loader.py`, search `cache_folder = self.exp_config.get`). So caches are
written next to the data file regardless of what the template says.

**Options.**
- A. Honour `cache_directory` as documented. Every template user's cache moves to `./cache`
  (relative to the working directory, unless anchored to the config file's directory like other
  relative paths); existing caches next to the data are orphaned and rebuilt once; a read-only or
  missing `./cache` becomes a new failure mode.
- B. Remove `cache_directory` from the templates and the manager defaults, and document
  `cache_file_path` (the key that works) instead. No behaviour change.
- C. Treat `cache_directory` as an alias of `cache_file_path` ONLY when the user set it to
  something other than the template default; otherwise keep today's location.
- D. Honour it with a read fallback: look in `cache_directory` first, then next to the data file;
  write to `cache_directory`.
- E. Something else.

**Claude's recommendation:** B. The data folder is where every existing cache lives, the working
key already exists, and A/D change where files are written for every user to honour a comment.
Option C makes behaviour depend on whether a value equals a default, which is surprising.

---

## Item 3 — core-B3: the opt-in adapter path clips times beyond `10000 * dt`

**Problem.** `NLSQAdapter` (opt-in, `use_adapter=True`; not the default path) calls
`model.compute_g1_batch(...)` without a `time_grid`
(`$WT/xpcsjax/optimization/nlsq/adapter.py`, search `compute_g1_batch(`). The kernel then falls
back to a fixed `arange(10001) * dt` grid and clamps indices
(`$WT/xpcsjax/core/jax_backend.py`, search `_FALLBACK_GRID_SIZE`), so any time beyond `10000 * dt`
is silently evaluated at the last grid point. The kernel docstring
(`$WT/xpcsjax/core/models.py`, `compute_g1_batch`, the `time_grid` parameter) records that wiring
it was deferred because the grid must be the full, uniformly dt-spaced experiment axis computed
ONCE outside any traced closure, not derived from a batch.

**Facts.**
- The default path (`NLSQWrapper`) and the CMA-ES path already pass a proper `time_grid`.
- `get_or_create_model(analysis_mode, phi_angles, q, per_angle_scaling, config, enable_jit, t, dt, L)`
  builds and caches the closure; the cache key is `ModelCacheKey` (same file).
- `_build_model_function(data, ...)` in the adapter has the data object, hence the full `t1` axis.

**Options.**
- A. Wire it: add a `time_grid` argument to `get_or_create_model`, built once in
  `_build_model_function` from the data's full unique time axis; put a compact signature of it
  (length, first value, dt) in `ModelCacheKey`; pass it to both `compute_g1_batch` call sites.
- B. Fail loudly instead of clipping: on the adapter path, raise a clear `ValueError` when the
  data's largest time exceeds the fallback grid, telling the user to use the default path.
  No silent wrong result, no API change.
- C. Leave deferred (documented), since the adapter is opt-in.
- D. Something else.

**Claude's recommendation:** B now. It removes the silent wrong result (standing rule 2) with a
few lines and no cache-key or signature change; A stays a possible later enhancement. If you
prefer A, say exactly what goes in the cache key and how non-uniform or duplicated time axes are
handled.

---

## What to write (your position file, exactly this structure)

# <codex|agy> position — round <n>
## R4-A3
- Choice: <A|B|C|D>
- Why: <2-5 sentences, citing code as path:line where it matters>
- Risks / conditions: <what must hold, what to test>
## data-B9
- Choice: ...
- Why: ...
- Risks / conditions: ...
## core-B3
- Choice: ...
- Why: ...
- Risks / conditions: ...
## Disagreements with Claude's recommendation
<list, or `None.`>
END_OF_POSITION
