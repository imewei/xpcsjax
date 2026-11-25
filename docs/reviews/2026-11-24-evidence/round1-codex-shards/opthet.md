# Codex review — shard opthet

## Files read

xpcsjax/optimization/nlsq/heterodyne_adapter.py — 968 — full  
xpcsjax/optimization/nlsq/heterodyne_adapter_base.py — 60 — full  
xpcsjax/optimization/nlsq/heterodyne_config.py — 1384 — full  
xpcsjax/optimization/nlsq/heterodyne_constant_mode.py — 642 — full  
xpcsjax/optimization/nlsq/heterodyne_core.py — 5011 — full  
xpcsjax/optimization/nlsq/heterodyne_data_prep.py — 367 — full  
xpcsjax/optimization/nlsq/heterodyne_engine_route.py — 803 — full  
xpcsjax/optimization/nlsq/heterodyne_logging.py — 346 — full  
xpcsjax/optimization/nlsq/heterodyne_memory.py — 323 — full  
xpcsjax/optimization/nlsq/heterodyne_multistart.py — 161 — full  
xpcsjax/optimization/nlsq/heterodyne_result_builder.py — 847 — full  
xpcsjax/optimization/nlsq/heterodyne_results.py — 217 — full  
xpcsjax/optimization/nlsq/heterodyne_stratified_data.py — 229 — full  
xpcsjax/optimization/nlsq/heterodyne_stratified_ls.py — 1415 — full  
xpcsjax/optimization/nlsq/heterodyne_views.py — 109 — full

## Bugs

### opthet-B1: The in-memory engine swaps angle-specific scaling on unsorted input

- Location: xpcsjax/optimization/nlsq/heterodyne_engine_route.py:581
- Severity: Required
- Failure scenario: With input `phi=[75, 0]` and true offsets `[1.1, 0.9]`, an `individual` fit can reach an engine SSR of approximately `5e-31` and report `chi_squared=8.8`. In `constant` mode the mismatch occurs during the solve, where the frozen offsets are applied to the opposite angles.
- Root cause: `StratifiedResidualFunctionJIT` builds its angle grid with `np.unique`, which sorts the angles. The route reads individual optimizer slots as if they already followed the input order when it scores and returns the result. Conversely, `_quantile_frozen_scaling` returns input-order arrays that `_build_engine` passes straight into the sorted engine.
- Checked: `_fit_nlsq_heterodyne` preserves the supplied angle and C2 order. The full-physics and weight guards permit this case. The engine result-contract tests use sorted angles or uniform scaling and do not exercise this permutation. Confirmed with a solver stub and the real residuals.
- Red test: Use a real 12-frame, two-angle model `m`. Stub `NLSQAdapter.fit` to return its physics seed with the exact sorted-angle scaling head, then check the engine’s public score.

  ```python
  phi = np.array([75.0, 0.0])
  c2 = np.stack([
      np.asarray(m.compute_correlation(
          phi_angle=p, angle_idx=i, contrast=0.3, offset=o
      ))
      for i, (p, o) in enumerate(zip(phi, [1.1, 0.9]))
  ])

  def exact_fit(self, *, initial_params, **kwargs):
      p = initial_params.copy()
      p[:4] = [0.3, 0.3, 0.9, 1.1]
      return NLSQResult(p, self._parameter_names, True, "ok")

  monkeypatch.setattr(NLSQAdapter, "fit", exact_fit)
  r = fit_two_component_via_engine(
      m, c2, phi, NLSQConfig(per_angle_mode="individual")
  )
  assert r.chi_squared < 1e-12
  ```

- Confidence: high

### opthet-B2: Stratified-LS results label sorted scaling and per-angle chi-squared with input-order angles

- Location: xpcsjax/optimization/nlsq/heterodyne_stratified_ls.py:1409
- Severity: Required
- Failure scenario: A stratified `individual` fit on `phi=[75, 0]` returns offsets `[0.9, 1.1]` alongside diagnostic angles `[75, 0]`, although the fitted offsets for those angles are `[1.1, 0.9]`. Downstream plots select the wrong offset. Unequal per-angle chi-squared values are likewise attributed to the wrong angles.
- Root cause: The pointwise residual, scaling head and per-angle SSR accumulation use the sorted `phi_unique` indices. The result assembly forwards those values and covariance in solver order while passing the original `phi` array as their labels.
- Checked: The stratified dispatcher forwards angles unchanged, and the deduplication guard checks count only. `_unpack_heterodyne_scaling` and the plotting caller index by the original data angle order. The stratified-LS fixtures use sorted angles. Confirmed with the real result builder and a solver stub.
- Red test: Reuse the model, data and `exact_fit` stub from opthet-B1.

  ```python
  r = fit_heterodyne_stratified_least_squares(
      model=m, c2=c2, phi=phi, weights=None,
      config=NLSQConfig(per_angle_mode="individual"),
  )
  np.testing.assert_allclose(r.parameters[2:4], [1.1, 0.9])
  ```

- Confidence: high

### opthet-B3: Stratified fitting turns zero-weight observations back on

- Location: xpcsjax/optimization/nlsq/heterodyne_stratified_data.py:167
- Severity: Required
- Failure scenario: Set the weight of a retained off-diagonal pixel to zero and add `1.0` to that observation. The stratified residual still contributes `-1.0` at the generating parameters. The masked outlier can move the fitted parameters and inflate chi-squared.
- Root cause: The weight-to-sigma conversion assigns `sigma=1` to zero weights. `build_joint_pointwise_residual` then multiplies the residual by `1/sigma`, giving the masked pixel full weight. The direct heterodyne kernel multiplies by `sqrt(weight)` and correctly zeros it.
- Checked: The public dispatcher accepts and forwards zero weights. `compute_weights` can produce zero weights, and `prepare_fit_data` explicitly treats them as excluded points. Existing stratified-LS tests use `weights=None` in the pointwise objective checks. Confirmed at the actual stratified residual.
- Red test: With a sorted two-angle stack generated at contrast `0.3` and offset `1.0`, run the following.

  ```python
  c2[0, 1, 2] += 1.0
  w = np.ones_like(c2)
  w[0, 1, 2] = 0.0
  d = build_heterodyne_stratified_data(model, c2, phi, w)
  residual, x, _, p0, _ = build_joint_pointwise_residual(
      model=model, stratified_data=d, per_angle_mode="individual",
      init_scaling=np.array([0.3, 0.3, 1.0, 1.0]),
  )
  assert np.asarray(residual(p0))[(x == [0, 1, 2]).all(axis=1)].item() == 0.0
  ```

- Confidence: high

### opthet-B4: Individual-mode regularization produces a non-finite Jacobian at the default seed

- Location: xpcsjax/optimization/nlsq/heterodyne_core.py:3399
- Severity: Required
- Failure scenario: A two-angle direct `individual` fit with `regularization_mode="adaptive"` starts with equal contrasts and equal offsets. The penalty residual values are finite, but their Jacobian rows contain NaNs at the first solver evaluation.
- Root cause: `_build_joint_problem` broadcasts the first scaling pair into every angle. The penalty then differentiates `jnp.std` at exactly zero variance. The mean-denominator guard leaves the singular derivative of `sqrt(variance)` intact. The stratified `_per_angle_cv` at `heterodyne_stratified_ls.py:394` has the same zero-variance derivative when a scaling group becomes uniform.
- Checked: Non-uniform weights and fixed physics both reach the direct path through the engine guards. `test_heterodyne_joint_problem.py` checks finite residual values with regularization disabled. `test_defense_layers.py` checks activation and penalty values, without testing the seed Jacobian. Confirmed with `jax.jacfwd` on the actual residual.
- Red test:

  ```python
  model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
  cfg = NLSQConfig(per_angle_mode="individual", regularization_mode="adaptive")
  p = _build_joint_problem(model, c2, phi, cfg, weights=None)
  assert np.isfinite(jax.jacfwd(p.joint_residual_fn)(jnp.asarray(p.x0))).all()
  ```

- Confidence: high

### opthet-B5: Joint individual global searches fall back when any physics parameter is fixed

- Location: xpcsjax/optimization/nlsq/heterodyne_core.py:2023
- Severity: Required
- Failure scenario: Fix `D0_ref`, leave 13 physics parameters varying, and request two-angle `individual` CMA-ES with warm-start auto-skip disabled. The warm-start residual raises a 14-to-13 broadcast error before CMA-ES runs. The caller receives a plain local fit. Flat multistart feeds the same expanded vector to the reduced residual at line 2315.
- Root cause: `_fit_joint_multi_phi` expands its public `OptimizationResult.parameters` to all 14 physics slots. Both escapes reuse that public vector as an optimizer-coordinate vector. The residual and bounds were constructed for varying physics only.
- Checked: The dispatcher honors the escape flags, and the fixed-physics engine guard affects only the engine route. `test_heterodyne_joint_escapes.py` exercises the escape with all physics varying. The tied-result assembly tests cover expansion on plain fits. Confirmed that the CMA-ES backend is never called in the fixed-parameter case.
- Red test:

  ```python
  model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
  model.param_manager.set_vary("D0_ref", False)
  cfg = NLSQConfig(
      per_angle_mode="individual", enable_cmaes=True,
      cmaes_warmstart_auto_skip=False,
  )
  backend = Mock(return_value=SimpleNamespace(success=False, parameters=None))
  monkeypatch.setattr(heterodyne_core, "fit_with_cmaes", backend)
  heterodyne_core.fit_nlsq_multi_phi(model, c2, phi, cfg)
  assert backend.called
  assert backend.call_args.kwargs["p0"].size == 17
  ```

- Confidence: high

### opthet-B6: The single-angle JAX adapter reports convergence after the solver exhausts its budget

- Location: xpcsjax/optimization/nlsq/heterodyne_adapter.py:551
- Severity: Required
- Failure scenario: Fit `x² − 4` from `x=0.3` with `max_nfev=2`. `CurveFit` returns `x=0.6` with `success=False` and status `0`. `fit_jax` returns `success=True` with the message “Optimization converged.” `_fit_local` can then accept and persist this unfinished solution.
- Root cause: `build_result_from_nlsq` preserves the backend verdict in `base.success`. `fit_jax` replaces it with `_assess_convergence`, which treats any finite, moved solution below its chi-squared threshold as converged.
- Checked: `_fit_local` trusts `result.success` before updating the model. The single-angle direct path is reachable through `fit_nlsq_jax` and through the public fallback for fixed physics. The adapter tests check heuristic inputs and budget forwarding separately. Confirmed with the installed `CurveFit`.
- Red test:

  ```python
  r = NLSQAdapter(["x"]).fit_jax(
      lambda _, x: x*x - 4 + np.zeros(3),
      np.array([0.3]), (np.array([-9.0]), np.array([9.0])),
      NLSQConfig(loss="linear", max_nfev=2), n_data=3,
  )
  assert not r.success
  ```

- Confidence: high

### opthet-B7: The engine route sends bounded `lm` fits into a backend that rejects them

- Location: xpcsjax/optimization/nlsq/heterodyne_engine_route.py:555
- Severity: Required
- Failure scenario: Request `method="lm"` on a small, unweighted, all-varying two-angle fit. The engine uses finite physics and scaling bounds. `CurveFit` rejects the combination, and public dispatch returns the failed engine result without trying the bounded `trf` solve used by the direct joint path.
- Root cause: The engine copies `config.method` directly into its bounded joint configuration. `NLSQAdapter.fit` converts the backend `ValueError` into a failed result, so the public dispatcher’s exception fallback never runs.
- Checked: `NLSQConfig` accepts `lm`. The direct joint builders explicitly convert it to `trf`. The engine’s mode, weights and physics guards allow the scenario, and its result tests use `trf`. Confirmed the installed backend rejects bounded `lm` through the adapter.
- Red test:

  ```python
  model, c2, phi = make_synthetic_two_component(n_phi=2, n_t=12)
  r = fit_two_component_via_engine(
      model, c2, phi,
      NLSQConfig(method="lm", loss="linear", per_angle_mode="individual"),
  )
  assert "doesn't work when there are bounds" not in r.nlsq_diagnostics["message"]
  ```

- Confidence: high

### opthet-B8: The stratified gradient check rejects fits with sensitivity in other parameters

- Location: xpcsjax/optimization/nlsq/heterodyne_stratified_ls.py:815
- Severity: Required
- Failure scenario: Set `f1=0` and vary only `f2` and `f3` in an averaged fit. Changing `f2` has no effect because it occurs inside `f1*(t−f2)`. The preflight raises “Gradient sanity check FAILED” before evaluating the sensitivity to `f3` or the scaling parameters. At production sizes the exception triggers the in-memory fallback.
- Root cause: The preflight perturbs only the first varying physics parameter and treats the absolute sum of its residual changes as a test of the entire Jacobian. It also permits changes of opposite sign to cancel within that one column.
- Checked: Fixed physics parameters are supported on the stratified route. The seed-clipping guard does not change this valid setup. The existing gradient tests check the perturbed index and the out-of-bounds seed case. Confirmed that the real preflight raises before the adapter is called.
- Red test: Generate a three-angle, 12-frame stack from a model with `f1=0`, then run the stratified driver with an adapter stub that records entry.

  ```python
  for name in list(model.param_manager.varying_names):
      if name not in {"f2", "f3"}:
          model.param_manager.set_vary(name, False)
  fit_heterodyne_stratified_least_squares(
      model=model, c2=c2, phi=phi, weights=None, config=NLSQConfig(),
  )
  assert adapter_spy.called
  ```

- Confidence: high

### opthet-B9: Zero-valued regularization rows shrink the reported direct-fit covariance

- Location: xpcsjax/optimization/nlsq/heterodyne_core.py:1674
- Severity: Required
- Failure scenario: Turn on regularization in a direct averaged fit with 330 data residuals and 16 optimizer parameters. Its two appended zero rows leave the objective and Jacobian information unchanged, yet the returned covariance is multiplied by `314/316`. The diagnostics also say the solver received 330 rows.
- Root cause: The averaged residual adds two synthetic, identically zero observations. `CurveFit` divides the covariance estimate by the number of residual rows minus the parameter count. The result assembly forwards that covariance unchanged and reports the data-only length as the solver length.
- Checked: The direct averaged route is reachable through non-uniform weights, fixed physics or global-escape dispatch. The installed `CurveFit` counts the passed rows, and the engine’s existing `rescale_covariance_dof` call confirms the project convention for synthetic rows. The covariance parity tests do not exercise the direct averaged penalty path. Confirmed with an adapter stub reproducing the backend’s row-count scaling.
- Red test: On a three-angle fixture, stub `NLSQAdapter.fit` to return `initial_params` and `np.eye(p)/(len(residual_fn(initial_params))-p)` as covariance, with `p=len(initial_params)`.

  ```python
  def run(mode):
      m, c2, phi = make_synthetic_two_component(n_phi=3, n_t=12)
      return fit_nlsq_multi_phi(
          m, c2, phi, NLSQConfig(regularization_mode=mode)
      )

  np.testing.assert_allclose(run("adaptive").covariance, run("none").covariance)
  ```

- Confidence: high

### opthet-B10: Nested configuration enables regularization even when `enable` is false

- Location: xpcsjax/optimization/nlsq/heterodyne_config.py:1028
- Severity: Required
- Failure scenario: A user sets `anti_degeneracy.regularization.enable: false` and leaves the mode present, as in the shipped two-component template. A direct individual fit still appends regularization penalties. The stratified execute-layers path can likewise attempt L3 despite the explicit opt-out.
- Root cause: `NLSQConfig.from_dict` consumes the nested `mode` but ignores the nested `enable` flag. The direct and stratified consumers use `regularization_mode != "none"` as their activation gate.
- Checked: `_fit_nlsq_heterodyne` passes the nested NLSQ configuration to this parser. The template contains `enable: false` alongside `mode: "relative"`. Config tests cover nesting a mode without an `enable` flag, and the execute-layers tests use a flat mode.
- Red test:

  ```python
  cfg = NLSQConfig.from_dict({
      "anti_degeneracy": {
          "regularization": {"enable": False, "mode": "tikhonov", "lambda": 1.0},
      },
  })
  assert cfg.regularization_mode == "none"
  ```

- Confidence: high

### opthet-B11: Uniform non-unit weights change the robust objective but the engine discards them

- Location: xpcsjax/optimization/nlsq/heterodyne_engine_route.py:479
- Severity: Required
- Failure scenario: Pass an all-`4.0` weight stack with `loss="soft_l1"` and data containing unequal residuals. The direct objective applies the robust loss to `2r`. The engine applies it to `r`, changing the relative influence of larger residuals and potentially changing the fitted parameters. It then reports chi-squared using the supplied weights.
- Root cause: The uniformity guard assumes a common residual multiplier leaves the minimizer unchanged. That holds for ordinary least squares with a positive multiplier. With a fixed robust-loss scale, the multiplier changes where residuals are downweighted. The engine builds unit-sigma data and keeps the same robust-loss settings.
- Checked: The public dispatcher forwards the weight stack. The engine guard accepts uniform weights and `soft_l1` is the default. The direct kernel applies `sqrt(weight)` before the robust solver sees the residual. Engine objective parity tests exercise the unit-weight case.
- Red test: For a sorted fixture with nonzero residuals, spy on `NLSQAdapter.fit` and capture the residual at the seed for inputs `weights=None` and `weights=np.full_like(c2, 4.0)` with `loss="soft_l1"`.

  ```python
  # If weighted engine input is unsupported, raising NotImplementedError
  # also satisfies the public fallback contract.
  try:
      weighted = capture_solver_residual(weights=np.full_like(c2, 4.0))
  except NotImplementedError:
      return
  unweighted = capture_solver_residual(weights=None)
  np.testing.assert_allclose(weighted, 2.0 * unweighted)
  ```

- Confidence: high

### opthet-B12: Stratified budget exhaustion is graded as a hard failure

- Location: xpcsjax/optimization/nlsq/heterodyne_result_builder.py:765
- Severity: Required
- Failure scenario: A stratified adapter returns a finite, usable fit with `success=False` after exceeding `max_nfev`. Even with reduced chi-squared near 1, the result is graded `failed` and `poor` instead of `max_iter` with quality based on the fit.
- Root cause: The stratified driver forwards the adapter’s termination text. The builder matches only `"Maximum function evaluations reached"`, while the installed `CurveFit` returns `"The maximum number of function evaluations is exceeded."`.
- Checked: `NLSQAdapter.fit` preserves the backend reason when its finite/movement checks pass. The heterodyne result-builder regression uses the legacy text. `test_stratified_max_iter_grading.py` covers the installed backend’s wording on the laminar path.
- Red test: In `test_build_hybrid_streaming_result_max_nfev_is_max_iter_not_failed`, use the actual backend reason and retain its assertions.

  ```python
  info["convergence_reason"] = "The maximum number of function evaluations is exceeded."
  r = build_hybrid_streaming_result(
      model=model, popt=popt, pcov=pcov, info=info, phi_angles=phi,
  )
  assert r.convergence_status == "max_iter"
  assert r.quality_flag == "good"
  ```

- Confidence: high

### opthet-B13: Sequential reduced chi-squared subtracts the size of the reporting vector

- Location: xpcsjax/optimization/nlsq/heterodyne_core.py:720
- Severity: Required
- Failure scenario: A one-angle sequential fit with 14 varying physics parameters and 110 retained observations divides its SSR by `110−16` before noise normalization. Only 14 parameters were fitted, so the denominator should be `110−14`. For three independently fitted angles the same code subtracts 20 instead of 42.
- Root cause: The aggregator uses the length of `[mean_physics | fixed contrasts | fixed offsets]` as the fitted parameter count. The SSR comes from the separate per-angle solves, each of which fits its own varying physics and holds both scaling terms fixed.
- Checked: `fit_nlsq_multi_phi` explicitly reaches this path for one angle, and the public fixed-physics engine guard can route a one-angle fit there. `_fit_local` confirms that scaling is frozen. `test_individual_no_config_falls_back_to_sequential` currently repeats the reporting-vector formula in its expected value.
- Red test:

  ```python
  model, c2, phi = make_synthetic_two_component(n_phi=1, n_t=12)
  r = fit_nlsq_multi_phi(model, c2, phi, NLSQConfig())
  assert r.chi_squared > 0
  expected = noise_normalized_reduced_chi2(
      r.chi_squared, c2, 110, model.param_manager.n_varying,
  )
  np.testing.assert_allclose(r.reduced_chi_squared, expected, rtol=1e-12)
  ```

- Confidence: high

## Non-blocking notes

- Reviewed `bd0a549` with targeted in-memory probes. The test suite and the remaining proposed regressions were not run.

## Verdict

BUGS_FOUND

END_OF_REPORT