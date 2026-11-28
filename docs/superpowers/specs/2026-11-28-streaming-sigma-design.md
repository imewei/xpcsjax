# Fits with `sigma` on the streaming tier

**Scope.** `two_component` masked and equal-weight fits stream. `laminar_flow` keeps
its stratified least-squares results and now reports the fallback.

## Problem
NLSQ's `AdaptiveHybridStreamingOptimizer.fit` rejects `sigma`. `two_component` raised
for masked or weighted pixels. `laminar_flow` always passes `sigma` (0.01 filled in),
threaded into the plain branch by PR #15, and fell back to stratified least squares
without saying so. The data loader, CLI and GUI produce neither `sigma` nor `weights`.

## Approaches
- A: fit equal `sigma` as plain least squares; decline uneven `sigma`.
- B: fold uneven weights into data and model. Supports them, costs memory.
- C: do not stream when `sigma` is present.

## Decision
A was chosen first, for both models. On a tiled, noise-free 1.08M-point
`laminar_flow` dataset it was then measured against `main`:

| Run | Status | Chi-squared | Uncertainties |
|---|---|---|---|
| `main` stratified least squares | converged | 111.72 | finite |
| streaming, template limits | failed | 111.80 | withheld |
| streaming, Gauss-Newton limit 1000 | failed | 111.72 | withheld |

The streaming solver stops short of its 1e-6 relative-gradient test (6.8e-4 at 1000).
The maintainer therefore chose **A for `two_component` only; C for `laminar_flow`**.

## Design
- `observed_equal_sigma_rows` (data_prep.py): mask of observed rows; `ValueError` if
  none; `NotImplementedError` if unequal (exact comparison).
- `two_component` plain branch: compact only the arrays given to `optimizer.fit`;
  withhold covariance for sigma-bearing fits until a rank check exists (#86 work).
- Masked pixels stay out of the quantile scaling estimate and the Layer 4 monitored
  loss: their stored values changed the fit. The estimate sits in the shared pointwise
  builder, so the Layer 2 starting point of a masked fit changes too. The engine route
  builds without weights and stratified least squares discards the builder's scaling.
- `laminar_flow`: solver code unchanged. The route records `device_info`
  `fallback_occurred` / `fallback_reason`; core.py merges it with the adapter's.
- Layer 2 sigma handling unchanged.

## Non-goals
Uneven weights; `laminar_flow` streaming convergence; Layer 5 acting only in warmup;
MemoryError falling back to a larger solver; keeping the reason when stratified least
squares also fails; the reduced chi-squared denominator counting zeroed diagonal rows;
the same unmasked scaling estimate in `two_component` stratified least squares.
