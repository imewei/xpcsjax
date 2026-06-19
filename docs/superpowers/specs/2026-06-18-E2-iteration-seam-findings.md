# Plan E2 — Task 1 Confirmation Spike: Findings

**Status:** GO (live SSR is feasible at the default-path seam, byte-identical when `on_iteration=None`).
**Scope:** homodyne / laminar_flow standard in-memory path. Other paths degrade gracefully to
`LogLine` + `LayerStatus` + `Banner` (no per-iteration SSR), exactly as the spec's
graceful-degradation clause allows.

> Reconstructed post-implementation (the spike deliverable required by Plan E2 Task 1,
> Step 4 was not committed at the time). Every claim below is re-verified against the
> merged code on `main` (HEAD `4936b92`) with `file:line` evidence.

## Q1 — Does `nlsq.CurveFit.curve_fit` expose a per-iteration callback?

**Yes.** Its signature accepts `callback: Callable | None = None`:

```
nlsq.CurveFit.curve_fit(self, f, xdata, ydata, p0=None, sigma=None, ...,
                        callback: 'Callable | None' = None,
                        compute_diagnostics=False, ...)
```

(`uv run python -c "import nlsq, inspect; print(inspect.signature(nlsq.CurveFit.curve_fit))"`.)

The callback is invoked per solver iteration as `callback(iteration, cost)`, where `cost` is
the current SSR. xpcsjax already used this same `callback` slot for the L4 gradient-collapse
monitor, so the contract was known-good before the spike.

## Q2 — Chosen seam point

The single function where `on_iteration(iteration, cost)` is surfaced is the NLSQ wrapper's
`curve_fit` callback shim, which is also where the L4 monitor callback is built:

- `xpcsjax/optimization/nlsq/wrapper.py:326-332` — when `on_iteration` is set and L4 is **off**,
  a callback is installed solely to forward `on_iteration(int(iteration), float(cost))`.
- `xpcsjax/optimization/nlsq/wrapper.py:376-384` — when L4 **is** on, the existing L4 callback
  additionally calls `on_iteration(int(iteration), float(cost))` (`cost == SSR`).

So the observer rides the **existing** `CurveFit(..., callback=...)` slot
(`xpcsjax/optimization/nlsq/wrapper.py:2117`); no new solver argument is introduced.

## Q3 — Threading list (functions that gained the `on_iteration` keyword, in order)

1. `fit_nlsq(data, config, *, on_iteration=None)` — `xpcsjax/optimization/nlsq/__init__.py:462,466`;
   forwards on the homodyne/laminar path only (`:529`, `:563-564`).
2. `fit_nlsq_jax(..., on_iteration=None)` — `xpcsjax/optimization/nlsq/wrapper.py:299`.
3. core solve entry — `xpcsjax/optimization/nlsq/core.py:251` (param), `:546` (forward).
4. (streaming) `xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py:1800` — accepted; the
   ≥1 M stratified tier remains a documented non-wired follow-up.
5. service boundary — `xpcsjax/service/fit.py:172-182` builds an `on_iteration(n, ssr)` closure
   that emits an `Iteration` event, and passes it to `fit_nlsq` **only when an `on_event` sink is
   supplied** (default `None` → legacy 2-positional `fit_nlsq(data, cm)` call shape preserved).

## Q4 — GO / NO-GO: is a byte-identical-when-`None` change feasible?

**GO.** Passing `on_iteration=None` is structurally inert — the wrapper installs no extra callback
on that path, so the solver trajectory is unchanged. This is pinned by parity tests:

- `tests/parity/test_iteration_seam_parity.py::test_on_iteration_none_is_byte_identical` —
  `fit_nlsq(data, config)` vs `fit_nlsq(data, config, on_iteration=None)` are byte-identical.
- `tests/optimization/test_iteration_callback_seam.py` — exercises the observer firing with a
  real callback (count / monotone-SSR sanity), and the default-off no-op.

## Decision

Implement live per-iteration SSR on the homodyne/laminar standard in-memory path via the
`on_iteration` observer described above; everything else degrades to log/banner/layer events.
No engine numerics change on the default path (parity-gated at `rtol=1e-10` byte-identity).
