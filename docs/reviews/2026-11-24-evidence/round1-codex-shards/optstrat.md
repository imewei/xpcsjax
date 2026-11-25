# Codex review — shard optstrat

## Files read

xpcsjax/optimization/nlsq/strategies/__init__.py — 65 — full  
xpcsjax/optimization/nlsq/strategies/chunking.py — 1242 — full  
xpcsjax/optimization/nlsq/strategies/executors.py — 490 — full  
xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py — 1124 — full  
xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py — 1745 — full  
xpcsjax/optimization/nlsq/strategies/hybrid_streaming_layers.py — 568 — full  
xpcsjax/optimization/nlsq/strategies/hybrid_streaming_transforms.py — 416 — full  
xpcsjax/optimization/nlsq/strategies/out_of_core.py — 744 — full  
xpcsjax/optimization/nlsq/strategies/residual.py — 873 — full  
xpcsjax/optimization/nlsq/strategies/residual_jit.py — 661 — full  
xpcsjax/optimization/nlsq/strategies/sequential.py — 1037 — full  
xpcsjax/optimization/nlsq/strategies/stratified_ls.py — 1175 — full

## Bugs

### optstrat-B1: Homodyne stratified fits assign sigma to the wrong angles

- Location: xpcsjax/optimization/nlsq/strategies/stratified_ls.py:155
- Severity: Required
- Failure scenario: Raw phi `[90, 0]` with sigma `2` at 90° and `4` at 0° gives a unit error at 90° a weighted residual of `0.25` instead of `0.5`. Both the stratified solve and the reported chi-squared can change.
- Root cause: The wrapper carries sigma in loader order and supplies `phi_raw`. `create_stratified_chunks` drops `phi_raw` and forwards sigma unchanged. Both residual engines gather sigma using their sorted theory-grid indices.
- Checked: `wrapper.py:2624`, both branches of `create_stratified_chunks`, the gathers in `residual.py:635` and `residual_jit.py:503`, and the strategy residual tests, whose ascending angles and unit sigma hide the swap.
- Red test: Parametrize over both residual engines. Build one chunk from `phi=np.repeat([90., 0.], 4)`, `t1=np.tile([1., 1., 2., 2.], 2)`, `t2=np.tile([1., 2., 1., 2.], 2)`, and `g2=np.full(8, 2.)`. Set `phi_raw=[90.,0.]` and broadcast `[2.,4.]` over the two sigma slabs. With scalar contrast `0` and offset `1`, assert `r[[1,5]] == [.5,.25]`. Today the entries are reversed.
- Confidence: high

### optstrat-B2: Weighted heterodyne streaming swaps sigma slabs for unsorted angles

- Location: xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py:354
- Severity: Required
- Failure scenario: Two descending angles with inverse-variance weights `4` and `25` receive each other’s weights in the streaming objective.
- Root cause: `build_heterodyne_stratified_data` retains input order on sigma’s angle axis. The streaming model gathers that axis with sorted phi indices. The frozen contrast and offset arrays already use sorted order and are correctly aligned.
- Checked: The public heterodyne dispatcher forwards phi and weights to the builder in input order. The frozen-scaling permutation test uses `weights=None`. The streaming sigma test uses ascending phi.
- Red test: In `test_heterodyne_hybrid_streaming.py`, take `_make_synthetic_heterodyne(n_phi=2, n_t=8)`, reverse both `phi` and `c2`, and build stratified data with row weights `[4.,25.]` broadcast across time. Call `build_heterodyne_pointwise_model` and assert `np.testing.assert_allclose(meta["sigma"], np.where(x_data[:, 0] == 1, .5, .2))`. Today the values are reversed.
- Confidence: high

### optstrat-B3: Stratified-LS rejects a legal zero-contrast starting point before solving

- Location: xpcsjax/optimization/nlsq/strategies/stratified_ls.py:597
- Severity: Required
- Failure scenario: An individual-mode fit starts with every contrast at its legal lower bound of zero. The strategy raises `Gradient sanity check FAILED` even though changing contrast or offset changes the residual and the solver can take a step.
- Root cause: The preflight perturbs one physics parameter and treats a zero sum of residual changes as proof that the entire Jacobian vanishes. At zero contrast, the physics derivative vanishes while the scaling derivatives remain active.
- Checked: Individual mode preserves the input scaling vector. The wrapper accepts a zero lower bound and falls back after this exception. The direct laminar parity fixture starts at contrast `0.3`.
- Red test: Use `s, x, bounds, n_phi = _build_laminar_stratified_data()` from `tests.parity.test_laminar_execute_layers`. Set `x[:n_phi] = 0`. Call `fit_with_stratified_least_squares(s, True, _NAMES, x, bounds, _LOG, anti_degeneracy_config={"per_angle_mode": "individual"})` and assert the `LeastSquares.least_squares` spy is reached. The preflight currently raises before the call.
- Confidence: high

### optstrat-B4: Stratified-LS covariance counts observations excluded from the objective

- Location: xpcsjax/optimization/nlsq/strategies/stratified_ls.py:938
- Severity: Required
- Failure scenario: A five-angle, 12-frame unweighted fit supplies 660 off-diagonal observations, but the Gauss–Newton covariance uses 720. The inflated denominator understates uncertainty.
- Root cause: `n_real_points` removes padding from the row count. It still includes the diagonal rows and any rows zeroed by the sigma guard in `residual_jit.py:506–518`.
- Checked: The installed `LeastSquares` result reaches the no-`pcov` branch. The wrapper forwards this covariance. The covariance contract tests check placeholder handling and finite values without checking observation counts.
- Red test: Monkeypatch `stratified_ls.gauss_newton_covariance` with a spy that records `n_valid` and returns `np.eye(J.shape[1])`. Call `tests.parity.test_laminar_execute_layers._fit(False)` and assert the recorded count is `5 * 12 * 11`. Today it is `5 * 12 * 12`.
- Confidence: high

### optstrat-B5: Out-of-core covariance counts masked rows and fixed parameters

- Location: xpcsjax/optimization/nlsq/strategies/out_of_core.py:513
- Severity: Required
- Failure scenario: One angle with a 4×4 time grid and five parameter slots, one fixed, has 12 contributing rows and four fitted parameters. The covariance uses `RSS/11` in place of `RSS/8`.
- Root cause: Both accumulation branches count every input row even though the kernels zero the diagonal. The Hessian is then sliced to free columns, but the covariance denominator still subtracts the full parameter count.
- Checked: `parallel_accumulator.create_ooc_kernels` masks the diagonal. Both wrapper routes forward `resolved_physical`. `test_out_of_core_dof.py` checks mode expansion, and the fixed-parameter integration test checks the fixed value and its zero uncertainty.
- Red test: Call `fit_with_out_of_core_accumulation` with one angle, `t1=t2=np.arange(4.)`, a `(1,4,4)` observation cube, five initial values, and three physical names with one fixed in `ResolvedPhysicalParameters`. Stub `parallel_accumulator.create_ooc_kernels` so accumulation returns `(np.eye(5), np.zeros(5), 8.)` and the cost kernel returns `8.`. Assert `pcov[0,0] == pytest.approx(1.)`. Today it is `8/11`.
- Confidence: high

### optstrat-B6: Sequential uniform and point-count weighting mix fitted scaling with unused guesses

- Location: xpcsjax/optimization/nlsq/strategies/sequential.py:799
- Severity: Required
- Failure scenario: Angle A fits its contrast to `0.4` and angle B fits its contrast to `0.8`. Each leaves the other angle’s initial contrast at `0.2`. Equal weighting returns `[0.3, 0.5]` and can propagate NaN variances into both fitted slots.
- Root cause: The wrapper’s per-angle residual depends only on the current angle’s scaling slots. The two nondefault weighting modes average every slot across all successful fits, including slots absent from that angle’s model.
- Checked: `wrapper.py` selects only the current angle’s scaling and forwards `optimization.sequential.weighting`. The singular-angle covariance regression exercises the default inverse-variance weighting.
- Red test: Pass `combine_angle_results` two successful results with `parameters=[.4,.2]` and `[.2,.8]`, covariance diagonals `[.01,np.nan]` and `[np.nan,.01]`, and equal `n_points`. Parametrize `weighting` over `"uniform"` and `"n_points"`. Assert parameters `[.4,.8]` and covariance diagonal `[.01,.01]`.
- Confidence: high

### optstrat-B7: Sequential fits report finite variance for collinear parameters

- Location: xpcsjax/optimization/nlsq/strategies/sequential.py:647
- Severity: Required
- Failure scenario: With `alpha=0` fixed, the diffusion model depends on `D0 + D_offset`. The sequential fit can nevertheless assign a finite variance to each parameter and use both variances for inverse-variance weighting.
- Root cause: The singular fallback computes a pseudoinverse and invalidates only zero Jacobian columns. Identical nonzero columns have a null direction while both diagonal entries of the normal matrix are positive.
- Checked: `calculate_diffusion_coefficient` gives the two parameters identical sensitivity at `alpha=0`. The wrapper supports fixing alpha. `test_sequential_singular_angle_is_excluded_not_weighted_as_sigma_one` covers a zero column.
- Red test: In `optimize_single_angle`, use 12 observations and `residual_func=lambda p, phi, t1, t2, g2: g2-p.sum()` with `g2=np.arange(12.)`. Stub `_get_nlsq_engine().least_squares` to return `x=np.array([2.,3.5])`, `jac=np.ones((12,2))`, and `success=True`. Assert `np.isnan(np.diag(result["covariance"])).all()`. Today both variances are positive and finite.
- Confidence: high

### optstrat-B8: Hierarchical streaming accepts covariance with a negative variance direction

- Location: xpcsjax/optimization/nlsq/strategies/heterodyne_hybrid_streaming.py:936
- Severity: Required
- Failure scenario: An invertible Hessian with leading block `[[-1,2],[2,-1]]` yields an inverse with positive diagonal and a negative variance for the difference of the first two parameters. The strategy labels the covariance valid. The homodyne streaming path has the same gap at `hybrid_streaming.py:1479`.
- Root cause: Both strategies invert the full nonlinear Hessian. Their acceptance checks inspect finiteness and diagonal entries, which cannot detect all negative eigenvalues. The bounded L2 solver can return points with indefinite full Hessians.
- Checked: Heterodyne’s result builder trusts the placeholder flag. The streaming tests cover a singular Hessian, negative marginal variances, and a positive-definite Hessian.
- Red test: In `test_streaming_l2_real_hessian_covariance_via_inv_on_success`, keep the noisy fixture. Replace the mocked Hessian with `jnp.eye(n).at[:2,:2].set(jnp.array([[-1.,2.],[2.,-1.]]))` and assert `info["covariance_is_placeholder"] is True`. Today it is `False`.
- Confidence: high

### optstrat-B9: Homodyne streaming enables L3 despite `regularization.enable: false`

- Location: xpcsjax/optimization/nlsq/strategies/hybrid_streaming_layers.py:343
- Severity: Required
- Failure scenario: A laminar individual-mode streaming fit with L2 active and L3 explicitly disabled still adds an L3 penalty when its scaling varies across angles. This can change the fitted parameters.
- Root cause: The layer setup gates on `per_angle_scaling` alone, hardcodes `enable=True`, and also enables the backend group-variance setting.
- Checked: The shipped laminar template explicitly defaults `regularization.enable` to `false`. The wrapper forwards that block. The L2 loss applies the penalty whenever the regularizer exists.
- Red test: In `test_hybrid_streaming_singular_hessian.py`, keep the existing `regularization={"enable": False}` case and add `assert info["anti_degeneracy"]["adaptive_regularization_enabled"] is False`. Today it is `True`.
- Confidence: high

### optstrat-B10: Generic hybrid streaming rejects the wrapper’s normal config type

- Location: xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py:155
- Severity: Required
- Failure scenario: A homodyne fit using `performance.strategy_override: streaming` reaches the fallback chain with a `ConfigManager`. Generic hybrid streaming raises `AttributeError` before the optimizer starts, causing the requested strategy to fall back.
- Root cause: The fallback chain passes the wrapper’s original config unchanged. The strategy assumes every non-`None` value is an NLSQ-specific config object with `hybrid_*` attributes.
- Checked: `NLSQWrapper.fit`, its `_fit_with_hybrid_streaming_optimizer` delegate, and `fallback_chain.py:327–338`. The retry tests supply `None`, while the fallback dispatch tests stub the whole strategy.
- Red test: In `test_hybrid_streaming_retry.py`, reuse `_enable` with a successful fake and call `fit_with_hybrid_streaming_optimizer` using the ordinary test inputs but `nlsq_config={}`. Assert the fake’s `fit` is called. Today config construction raises at `.hybrid_normalize`.
- Confidence: high

### optstrat-B11: Generic hybrid streaming supplies an incorrect prediction callback

- Location: xpcsjax/optimization/nlsq/strategies/hybrid_streaming.py:227
- Severity: Required
- Failure scenario: For `x=[1,2,3]`, `y=2*x`, and the wrapper-style model `f(x,p)=p*x`, the starting parameter `p=2` should fit exactly. The callback returns zeros on the full batch and raises a broadcasting error on a two-point chunk when the model honors `x`.
- Root cause: The wrapper already supplies predictions. The strategy treats them as residuals and subtracts them from the full observed vector despite the streaming backend’s chunk-local prediction contract.
- Checked: `wrapper.py:3499–3521` documents the prediction contract. NLSQ’s streaming scan subtracts the callback output from `y_chunk`. The retry fakes leave the callback unevaluated.
- Red test: In `test_hybrid_streaming_retry.py`, use `nlsq_config=None` and a fake optimizer whose `fit` asserts `np.testing.assert_allclose(func(x[:2], 2.), y[:2])` for the inputs above. Today the callback raises.
- Confidence: high

### optstrat-B12: The public streaming executor fails on partial batches

- Location: xpcsjax/optimization/nlsq/strategies/executors.py:403
- Severity: Required
- Failure scenario: `StreamingExecutor` receives 12 observations with chunk size 4 and a residual callback that returns one residual per supplied point. The first four-point call attempts to subtract four residuals from 12 observations.
- Root cause: The executor captures full `ydata` in its residual-to-model adapter. Its `jax_residual` branch also ignores the supplied chunk and returns full-length predictions.
- Checked: `get_executor("streaming")` is publicly exported. NLSQ evaluates the callback on partial batches. `test_streaming_executor_calls_real_fit_api` exercises only full-batch callback evaluation.
- Red test: In that test, use `xdata=np.arange(12.)`, `ydata=2*xdata`, `p0=np.array([1.])`, and `resid=lambda x,p: 2*x-p*x`. Assert `np.testing.assert_allclose(_FakeOpt.last_call["func"](xdata[4:8], *p0), xdata[4:8])`. Today the callback raises.
- Confidence: high

### optstrat-B13: Out-of-core ignores the documented iteration limit

- Location: xpcsjax/optimization/nlsq/strategies/out_of_core.py:290
- Severity: Required
- Failure scenario: `optimization.nlsq.max_iterations: 1` still permits up to 50 out-of-core iterations.
- Root cause: The strategy reads `optimization.max_iterations`. Neighboring tolerance reads use the nested NLSQ block.
- Checked: The laminar template, `ConfigManager`, and `NLSQConfig` use the nested key. Both wrapper routes forward the config. The existing out-of-core tests cover routing and covariance without testing the iteration limit.
- Red test: Call the strategy on a one-angle 3×3 grid with a three-element starting vector of ones and `config={"optimization":{"nlsq":{"max_iterations":1}}}`. Stub `create_ooc_kernels` to return accumulation `(I, p, float(p@p))` and trial cost `float(p@p)`. Assert `info["iterations"] == 1`. Today the strategy accepts more than one step.
- Confidence: high

### optstrat-B14: Out-of-core reports an exact optimum as a line-search stall

- Location: xpcsjax/optimization/nlsq/strategies/out_of_core.py:625
- Severity: Minor
- Failure scenario: The initial parameters exactly reproduce the observations and have a finite Jacobian. Every trial has zero step and zero cost. The resulting status is `line_search_stalled`.
- Root cause: Convergence is evaluated only after a strict reduction in cost. A zero-gradient, zero-cost initial point can never enter that branch.
- Checked: Both accumulation paths feed this loop. The out-of-core result builder forwards the status directly. The existing out-of-core tests do not cover an initially stationary point.
- Red test: Use the three-parameter out-of-core probe with a positive iteration budget, accumulation `(np.eye(3), np.zeros(3), 0.)`, and trial cost `0.`. Assert `info["convergence_status"] == "converged"`. Today it is `"line_search_stalled"`.
- Confidence: high

### optstrat-B15: Sequential fitting drops valid parameter scaling when a parameter is fixed

- Location: xpcsjax/optimization/nlsq/strategies/sequential.py:931
- Severity: Minor
- Failure scenario: With `x_scale=[2.,7.]` and the second parameter fixed, the per-angle solver silently uses its default scale for the first parameter.
- Root cause: The first normalization pass produces a full-length array. The second pass validates that array against the reduced parameter count without applying the free mask, so it drops `x_scale`.
- Checked: The wrapper forwards numeric `x_scale`. The sequential fixed-parameter tests cover value and covariance restoration without asserting scale propagation.
- Red test: Call `optimize_per_angle_sequential` with 12 points at one angle, `initial_params=[1.,1.]`, bounds `([0.,1.],[3.,1.])`, and `x_scale=[2.,7.]`, using NumPy arrays for the parameters and bounds. Spy on `optimize_single_angle` and assert `kwargs["x_scale"]` equals `[2.]`. Today the key is absent.
- Confidence: high

### optstrat-B16: The public standard executor crashes for unbounded fits

- Location: xpcsjax/optimization/nlsq/strategies/executors.py:178
- Severity: Minor
- Failure scenario: Calling `StandardExecutor.execute` with `bounds=None` raises `TypeError` on a valid unbounded linear fit.
- Root cause: The executor forwards `None` into the locked NLSQ implementation, whose bound preparer unpacks a lower and upper bound. `LargeDatasetExecutor` already translates `None` to `(-np.inf, np.inf)`.
- Checked: The public executor signature permits `None`. The installed NLSQ `CurveFit` path calls `prepare_bounds` on the argument. The executor tests mock `curve_fit`.
- Red test: Call `StandardExecutor().execute(lambda x,p: p*x, np.arange(4.), 2*np.arange(4.), np.array([1.]), None, "linear", 1., logging.getLogger())` and assert `result.popt[0] == pytest.approx(2.)`. Today bound preparation raises.
- Confidence: high

## Non-blocking notes

- Static review at `bd0a549`. The red tests are proposed reproducers and remain unrun.

## Verdict

BUGS_FOUND

END_OF_REPORT