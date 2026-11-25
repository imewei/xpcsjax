# Codex review — shard optrest

## Files read

xpcsjax/optimization/batch_statistics.py — 186 — full  
xpcsjax/optimization/exceptions.py — 172 — full  
xpcsjax/optimization/nlsq/adapter_base.py — 37 — full  
xpcsjax/optimization/nlsq/anti_degeneracy_logging.py — 117 — full  
xpcsjax/optimization/nlsq/covariance.py — 230 — full  
xpcsjax/optimization/nlsq/data_prep.py — 377 — full  
xpcsjax/optimization/nlsq/fallback_chain.py — 518 — full  
xpcsjax/optimization/nlsq/fit_computation.py — 538 — full  
xpcsjax/optimization/nlsq/jacobian.py — 201 — full  
xpcsjax/optimization/nlsq/memory.py — 494 — full  
xpcsjax/optimization/nlsq/model_adapter.py — 231 — full  
xpcsjax/optimization/nlsq/nlsq_settings.py — 49 — full  
xpcsjax/optimization/nlsq/parallel_accumulator.py — 840 — full  
xpcsjax/optimization/nlsq/parameter_index_mapper.py — 353 — full  
xpcsjax/optimization/nlsq/parameter_utils.py — 944 — full  
xpcsjax/optimization/nlsq/progress.py — 194 — full  
xpcsjax/optimization/nlsq/recovery.py — 577 — full  
xpcsjax/optimization/nlsq/result_builder.py — 470 — full  
xpcsjax/optimization/nlsq/result_helpers.py — 81 — full  
xpcsjax/optimization/nlsq/results.py — 350 — full  
xpcsjax/optimization/nlsq/transforms.py — 497 — full  
xpcsjax/optimization/nlsq/validation.py — 785 — full  
xpcsjax/optimization/nlsq/wrapper_out_of_core_route.py — 169 — full  
xpcsjax/optimization/numerical_validation.py — 257 — full

## Bugs

### optrest-B1: Out-of-core worker timeouts can hang the fit during shutdown

- Location: xpcsjax/optimization/nlsq/parallel_accumulator.py:831
- Severity: Required
- Failure scenario: An out-of-core worker stalls during JIT compilation or chunk evaluation. The accumulation deadline expires, then cleanup waits indefinitely for that same worker. Shared-memory cleanup is also delayed.
- Root cause: `compute_accumulators` and `compute_chi2` bound the wait in `as_completed`, but `OOCComputePool.shutdown()` then calls `shutdown(wait=True, cancel_futures=True)`. Cancelling queued work leaves running work active.
- Checked: `strategies/out_of_core.py` calls `shutdown()` in `finally` before cleaning up shared memory. The real pool test in `test_parallel_accumulator.py` covers completed jobs and idempotent shutdown. A standalone probe with a running future confirmed that the timeout path requests a blocking join.
- Red test:
  ```python
  def test_ooc_timeout_does_not_join_running_worker(monkeypatch):
      from concurrent.futures import Future
      from types import SimpleNamespace
      from unittest.mock import Mock

      import numpy as np
      import pytest

      from xpcsjax.optimization.nlsq import parallel_accumulator as pa

      running = Future()
      assert running.set_running_or_notify_cancel()
      executor = SimpleNamespace(submit=lambda *args: running, shutdown=Mock())
      monkeypatch.setattr(pa, "_batch_timeout", lambda _: 0)
      pool = pa.OOCComputePool.__new__(pa.OOCComputePool)
      pool._executor, pool._n_chunks, pool._shutdown = executor, 1, False

      with pytest.raises(TimeoutError):
          with pool:
              pool.compute_accumulators(np.ones(1))

      assert running.done() or all(
          call.kwargs.get("wait") is False for call in executor.shutdown.call_args_list
      )
  ```
- Confidence: high

### optrest-B2: A successful retry produces a failed public result

- Location: xpcsjax/optimization/nlsq/recovery.py:316
- Severity: Required
- Failure scenario: The first solve hits its iteration limit and the perturbed retry returns valid, changed parameters and finite covariance. The final result has `success == False`, its message says the optimization failed, and the CLI warns about nonconvergence.
- Root cause: Recovery emits `converged_with_recovery` after a successful retry. That value is outside `ConvergenceStatus`, and `OptimizationResult.success` recognizes only `converged`. The retry history already travels separately in `recovery_actions`.
- Checked: `fallback_chain.py` passes the retry status through, and `NLSQWrapper._create_fit_result` preserves it. The standard public homodyne path enables recovery, and the CLI and service persistence consume `result.success`. The fallback regression tests exercise escalation on failed attempts. A standalone probe confirmed `success == False` after a successful retry.
- Red test:
  ```python
  import logging
  from unittest.mock import Mock

  import numpy as np

  from xpcsjax.optimization.nlsq.fallback_chain import OptimizationStrategy
  from xpcsjax.optimization.nlsq.recovery import execute_with_recovery
  from xpcsjax.optimization.nlsq.results import OptimizationResult

  x = np.arange(3.0)
  solver = Mock(side_effect=[
      RuntimeError("max iterations"),
      (np.array([2.0]), np.array([[4.0]])),
  ])
  p, cov, _, actions, status = execute_with_recovery(
      lambda x, a: a * x, x, 2 * x, np.array([1.0]), None,
      OptimizationStrategy.STANDARD, logging.getLogger(__name__),
      "linear", "jac", None, solver, None,
  )
  result = OptimizationResult(
      p, np.sqrt(np.diag(cov)), cov, 0.0, 0.0, status, 1, 0.0, {},
      recovery_actions=actions,
  )
  assert solver.call_count == 2 and actions
  assert result.success
  ```
- Confidence: high

### optrest-B3: Out-of-core reduced chi-squared counts configured frozen scaling even when scaling was fitted

- Location: xpcsjax/optimization/nlsq/wrapper_out_of_core_route.py:139
- Severity: Required
- Failure scenario: A laminar fit with five angles requests `per_angle_mode: constant` and routes to out-of-core. The accumulator fits 17 parameters, including ten scaling parameters, but the result subtracts only seven from the observation count. For 100 supplied rows and χ² of 83, the reported reduced χ² is about 0.892 instead of 1.0 on the same row-count basis.
- Root cause: Result assembly derives the fitted parameter count from the requested mode. The out-of-core kernel receives a dense vector and differentiates with respect to its scaling slots. Its covariance parameter-count helper already recognizes that a dense vector still fits scaling under a `constant` request.
- Checked: Both wrapper call sites supply dense vectors. `out_of_core.py` performs no scaling-mode reduction. `test_out_of_core_route.py::test_laminar_mode_constant_token_is_not_pinned` currently expects the seven-parameter denominator even though its stub returns 17 parameters. The real-solver covariance contract test checks dense expansion but does not assert reduced χ².
- Red test:
  ```python
  from tests.optimization.test_out_of_core_route import _run
  from xpcsjax.config.parameter_registry import AnalysisMode

  result = _run(
      per_angle_mode="constant", analysis_mode=AnalysisMode.LAMINAR_FLOW,
      n_phi=5, n_physical=7,
  )
  assert result.parameters.size == 17
  assert result.reduced_chi_squared == 130.0 / (1_000_000 - 17)
  ```
- Confidence: high

### optrest-B4: Standard recovery discards the configured parameter scales before the first solve

- Location: xpcsjax/optimization/nlsq/recovery.py:228
- Severity: Required
- Failure scenario: A standard fit supplies a valid per-parameter scale of `[99.0]` for a parameter initialized to `1.0`. With recovery enabled, the first solve receives `[1.001]`, so the configured trust-region scaling has no effect.
- Root cause: The standard recovery branch always sets `x_scale` from `abs(current_params) + 1e-3`. It discards the `x_scale_value` passed by the wrapper even on attempt zero.
- Checked: `NLSQWrapper.fit` resolves `x_scale`, `x_scale_map`, and `trust_region_scale` before forwarding the result through the fallback chain. `fit_nlsq_jax` enables recovery on the default wrapper path. The L4 observational test identifies this as the live standard solver call but does not assert scale forwarding. A standalone probe captured `[1.001]`.
- Red test:
  ```python
  import logging
  from unittest.mock import Mock

  import numpy as np

  from xpcsjax.optimization.nlsq.fallback_chain import OptimizationStrategy
  from xpcsjax.optimization.nlsq.recovery import execute_with_recovery

  x = np.arange(3.0)
  solver = Mock(return_value=(np.array([2.0]), np.array([[4.0]])))
  scale = np.array([99.0])
  execute_with_recovery(
      lambda x, a: a * x, x, 2 * x, np.array([1.0]), None,
      OptimizationStrategy.STANDARD, logging.getLogger(__name__),
      "linear", scale, None, solver, None,
  )
  np.testing.assert_array_equal(solver.call_args.kwargs["x_scale"], scale)
  ```
- Confidence: high

### optrest-B5: Homodyne strategy routing ignores `NLSQ_MEMORY_FRACTION`

- Location: xpcsjax/optimization/nlsq/memory.py:373
- Severity: Required
- Failure scenario: With 16 GiB available and `NLSQ_MEMORY_FRACTION=0.25`, an eight-million-point, 14-parameter fit is estimated to need about 5.42 GiB. Routing selects the standard solver under a 12 GiB threshold despite the configured 4 GiB threshold.
- Root cause: The selector defaults its argument to `0.75` and forwards it as an explicit override. `get_adaptive_memory_threshold` reads the environment only when that argument is `None`.
- Checked: The wrapper and its stratified recheck omit the fraction. The static-anisotropic template documents the environment override. The memory-routing and concurrency tests exercise explicit fractions or defaults, and a standalone probe confirmed that the selector uses 12 GiB with the override set.
- Red test:
  ```python
  def test_implicit_strategy_fraction_reads_environment(monkeypatch):
      from xpcsjax.optimization.nlsq import memory as mem

      monkeypatch.setenv("NLSQ_MEMORY_FRACTION", "0.25")
      monkeypatch.setattr(mem, "detect_available_system_memory", lambda: 16 * 1024**3)
      decision = mem.select_nlsq_strategy(8_000_000, 14, concurrency=1)
      assert (decision.threshold_gb, decision.strategy) == (4.0, mem.NLSQStrategy.OUT_OF_CORE)
  ```
- Confidence: high

### optrest-B6: Per-angle initialization pools observations from neighboring angle bins

- Location: xpcsjax/optimization/nlsq/parameter_utils.py:641
- Severity: Minor
- Failure scenario: Separate angle bins at `0°` and `0.05°` have true contrasts of `0.2` and `0.6`, with a shared offset of `1.0`. The initializer returns `0.4` for both contrasts.
- Root cause: The initializer enumerates distinct angle values, then builds each regression using a `0.1°` tolerance. Both regressions include both bins. The chunked branch at line 630 applies the same tolerance.
- Checked: The wrapper uses this initializer for laminar fits with more than three angles and no scaling override. Stratification retains distinct angle values, and the wrapper applies no minimum angular spacing. The analytical test in `test_parameter_utils.py` uses angles 45° apart. A standalone probe confirmed the pooled contrasts.
- Red test:
  ```python
  from types import SimpleNamespace

  import numpy as np

  from xpcsjax.optimization.nlsq.parameter_utils import compute_consistent_per_angle_init

  t1, t2 = (a.ravel() for a in np.meshgrid(np.arange(4.0), np.arange(4.0), indexing="ij"))
  g1_sq = np.exp(-np.abs(t1 - t2))
  data = SimpleNamespace(
      phi_flat=np.repeat([0.0, 0.05], t1.size),
      t1_flat=np.tile(t1, 2), t2_flat=np.tile(t2, 2),
      g2_flat=np.concatenate([1 + 0.2 * g1_sq, 1 + 0.6 * g1_sq]),
      q=1.0, L=1.0, dt=1.0,
  )
  contrasts, _ = compute_consistent_per_angle_init(
      data, np.array([1.0, 0.0, 0.0]), ["D0", "alpha", "D_offset"],
  )
  np.testing.assert_allclose(contrasts, [0.2, 0.6])
  ```
- Confidence: high

## Non-blocking notes

- Possible Duplicated Code. `fallback_chain.handle_nlsq_result` and `result_builder.normalize_nlsq_result` implement the same dict, tuple, and object dispatch. A shared normalizer would give result metadata one owner.
- Verification used source inspection and standalone probes at `bd0a549`. The test suite was skipped as requested. The timeout probe used a simulated running future.

## Verdict

BUGS_FOUND

END_OF_REPORT