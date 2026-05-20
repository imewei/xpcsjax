"""Core NLSQ fitting for heterodyne analysis.

Unified entry point for NLSQ optimization with:
- Global optimization selection (CMA-ES → multi-start → local)
- Adapter/wrapper fallback with automatic recovery
- Memory-aware strategy selection
- Per-angle and multi-angle fitting
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from xpcsjax.core.heterodyne_jax_backend import (
    compute_c2_heterodyne,
    compute_multi_angle_residuals,
    compute_residuals,
)
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.heterodyne_results import NLSQResult
from xpcsjax.optimization.nlsq.results import OptimizationResult
from xpcsjax.optimization.nlsq.validation import classify_fit_quality
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    # The runtime object the fitter receives is the stateful dataclass in
    # ``heterodyne_model_stateful`` (which exposes ``.t``, ``.q``, ``.dt``,
    # ``.scaling``, ``.param_manager``, ``.set_params``). The bare wrapper in
    # ``heterodyne_model`` is a PhysicsModelBase adapter without those fields,
    # so typing against it produced ~10 spurious "no attribute" mypy errors.
    from xpcsjax.core.heterodyne_model_stateful import (
        HeterodyneModel as HeterodyneModel,
    )

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — gated for graceful degradation
# ---------------------------------------------------------------------------

# NOTE: every optional import below binds the imported names to ``None`` in
# the ImportError branch. Without this, Pyright cannot reason through the
# ``if HAS_X: X(...)`` runtime gates and emits ~10 "X is possibly unbound"
# warnings per call site. With explicit ``None`` bindings the type becomes
# ``T | None`` and narrows correctly. Call sites still gate on the ``HAS_X``
# flag; the explicit ``is not None`` check at hot-path sites is belt-and-
# suspenders for readers, not a runtime necessity.
try:
    # The heterodyne-shaped NLSQAdapter / NLSQWrapper expect the upstream
    # contract (parameter_names + residual_fn). xpcsjax's own NLSQAdapter
    # (in adapter.py) is shaped differently. Use the ported heterodyne
    # adapter module so the orchestrator gets the contract it expects.
    from xpcsjax.optimization.nlsq.heterodyne_adapter import (
        NLSQAdapter,
        NLSQWrapper,
    )

    HAS_ADAPTERS = True
    HAS_WRAPPER = True
except ImportError:
    NLSQAdapter = None  # type: ignore[assignment,misc]
    NLSQWrapper = None  # type: ignore[assignment,misc]
    HAS_ADAPTERS = False
    HAS_WRAPPER = False

# Multi-start orchestration is intentionally NOT imported here: the v0.1
# ``_fit_multistart`` function raises NotImplementedError unconditionally (see
# its docstring for why — the upstream homodyne port called a class-style
# ``MultiStartOptimizer.fit(...)`` API that ``xpcsjax.optimization.nlsq.multistart``
# does not expose). Keep ``HAS_MULTISTART`` as a const ``False`` so the
# existing ``if HAS_MULTISTART: _fit_multistart(...)`` dispatch falls through
# to the warning + local-fit path instead of hitting NotImplementedError
# during normal smoke runs.
HAS_MULTISTART = False

try:
    from xpcsjax.optimization.nlsq.cmaes_wrapper import (
        CMAES_AVAILABLE,
        fit_with_cmaes,
    )

    HAS_CMAES = CMAES_AVAILABLE
except ImportError:
    fit_with_cmaes = None  # type: ignore[assignment,misc]
    HAS_CMAES = False

try:
    # Heterodyne uses its own memory module (``STANDARD/LARGE/STREAMING`` enum
    # vocabulary). The homodyne ``memory.py`` uses
    # ``STANDARD/OUT_OF_CORE/HYBRID_STREAMING`` — importing from there left
    # ``NLSQStrategy.LARGE`` undefined at runtime in the heterodyne hot path.
    from xpcsjax.optimization.nlsq.heterodyne_memory import (
        NLSQStrategy,
        select_nlsq_strategy,
    )

    HAS_MEMORY = True
except ImportError:
    NLSQStrategy = None  # type: ignore[assignment,misc]
    select_nlsq_strategy = None  # type: ignore[assignment,misc]
    HAS_MEMORY = False

# Export availability flag for tests
NLSQ_AVAILABLE = HAS_ADAPTERS


# ---------------------------------------------------------------------------
# Shared diagnostics helper (used by every joint multi-phi path that returns
# an OptimizationResult — currently the Fourier path here and the constant
# path in heterodyne_constant_mode.py via re-import)
# ---------------------------------------------------------------------------


def _build_heterodyne_diagnostics(
    per_angle_mode: str,
    chi2_per_angle: np.ndarray,
    scaling_source: str,
    fourier_basis_dim: int | None,
    **extras: Any,
) -> dict[str, Any]:
    """Build the standard heterodyne ``nlsq_diagnostics`` dict.

    Centralises the five canonical keys every heterodyne-side
    :class:`OptimizationResult` carries so the Fourier-mode joint path here
    and the constant-mode joint path in :mod:`heterodyne_constant_mode` stay
    in lockstep. Extra mode-specific keys (e.g. ``contrast_per_angle_fixed``
    in constant mode, ``fourier_coeffs`` in Fourier mode) are passed through
    ``**extras``.

    The ``"not_applicable_heterodyne"`` shear-weighting marker is Task D4's
    L5 N/A semantic: heterodyne does not use the homodyne shear-weighting
    layer, but the OptimizationResult schema may carry the key in other
    modes, so we make the absence explicit rather than omitting the key.
    """
    base: dict[str, Any] = {
        "per_angle_mode": per_angle_mode,
        "chi2_per_angle": chi2_per_angle,
        "scaling_source": scaling_source,
        "fourier_basis_dim": fourier_basis_dim,
        "shear_weighting": "not_applicable_heterodyne",
    }
    base.update(extras)
    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_nlsq_jax(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float = 0.0,
    config: NLSQConfig | None = None,
    weights: np.ndarray | jnp.ndarray | None = None,
    use_nlsq_library: bool = True,
    *,
    _skip_global_selection: bool = False,
) -> NLSQResult:
    """Fit heterodyne model to correlation data using NLSQ.

    This is the unified entry point for all NLSQ optimization.  When called
    it first checks for global optimization methods:

    1. If ``cmaes.enable: true`` → delegates to CMA-ES
    2. If ``multi_start.enable: true`` → delegates to multi-start
    3. Otherwise → runs local trust-region optimization

    The adapter is tried first; on failure the wrapper provides automatic
    retry with progressive recovery (HybridRecoveryConfig).

    Args:
        model: HeterodyneModel instance with parameters configured.
        c2_data: Experimental correlation data, shape (N, N).
        phi_angle: Detector phi angle (degrees).
        config: NLSQ configuration (default if None).
        weights: Optional weights (1/sigma²) for weighted least squares.
        use_nlsq_library: Whether to prefer nlsq library over scipy.
        _skip_global_selection: Internal flag — skip CMA-ES / multi-start check.

    Returns:
        NLSQResult with fitted parameters and diagnostics.
    """
    if config is None:
        config = NLSQConfig()

    logger.info("=" * 60)
    logger.info("NLSQ OPTIMIZATION")
    logger.info("=" * 60)
    logger.info("phi=%s°, method=%s", phi_angle, config.method)

    # ------------------------------------------------------------------
    # Global optimization selection (CMA-ES → multi-start → local)
    # ------------------------------------------------------------------
    if not _skip_global_selection:
        global_result = _try_global_optimization(
            model,
            c2_data,
            phi_angle,
            config,
            weights,
            use_nlsq_library,
        )
        if global_result is not None:
            return global_result

    # ------------------------------------------------------------------
    # Local optimization
    # ------------------------------------------------------------------
    return _fit_local(model, c2_data, phi_angle, config, weights, use_nlsq_library)


def fit_nlsq_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: list[float] | np.ndarray,
    config: NLSQConfig | None = None,
    weights: np.ndarray | None = None,
) -> list[NLSQResult]:
    """Fit model to correlation data at multiple phi angles.

    Two modes of operation controlled by ``config.per_angle_mode``:

    - **Joint fit** (``"fourier"``, ``"independent"``, or ``"auto"``
      with multiple angles) -- All angles are fit simultaneously in a
      single optimization.  In ``"fourier"`` mode, the optimizer vector is
      ``[physics_varying | fourier_contrast_coeffs | fourier_offset_coeffs]``,
      where the Fourier basis constrains smooth angular variation.
      In ``"independent"`` mode, each angle has its own contrast/offset
      (``2*n_phi`` scaling parameters), all optimized jointly.

    - **Sequential mode** (single angle or fallback) -- Angles are fit one
      at a time with warm-starting.

    Args:
        model: HeterodyneModel instance.
        c2_data: Correlation data, shape ``(n_phi, N, N)`` or ``(N, N)``.
        phi_angles: Array of phi angles (degrees).
        config: NLSQ configuration.
        weights: Optional weights, shape ``(n_phi, N, N)`` or ``(N, N)``.

    Returns:
        List of :class:`NLSQResult`, one per angle.
    """
    phi_angles = np.asarray(phi_angles)

    if c2_data.ndim == 2:
        c2_data = c2_data[np.newaxis, ...]

    if len(c2_data) != len(phi_angles):
        raise ValueError(
            f"Number of c2 matrices ({len(c2_data)}) doesn't match "
            f"number of phi angles ({len(phi_angles)})"
        )

    # ------------------------------------------------------------------
    # Determine whether to use homodyne-style joint multi-angle fitting.
    # ------------------------------------------------------------------
    use_joint = False
    # Pre-initialize so the ``if use_joint:`` branch below sees a bound name
    # even when the optional fourier_reparam import fails. ``use_joint`` is
    # only flipped True inside the try block where ``fourier`` is reassigned,
    # so this initial None is never actually consumed at runtime.
    fourier: Any = None
    if config is not None and len(phi_angles) > 1:
        if getattr(config, "enable_cmaes", False) and HAS_CMAES:
            logger.info("CMA-ES enabled, delegating to joint multi-angle CMA-ES")
            return _fit_joint_cmaes_multi_phi(
                _model=model,
                _c2_data=c2_data,
                _phi_angles=phi_angles,
                _config=config,
                _weights=weights,
            )

        # Resolve ``auto`` / explicit modes to a canonical dispatch token.
        # The resolver returns one of: "constant", "averaged", "fourier",
        # "individual". Keeping the table explicit makes the threshold
        # semantics testable in isolation — see
        # tests/optimization/test_heterodyne_modes.py.
        effective_mode = _resolve_effective_mode(config, len(phi_angles))
        logger.info(
            "Per-angle dispatch: requested=%s, n_phi=%d, constant_threshold=%d, "
            "fourier_threshold=%d, effective=%s",
            config.per_angle_mode,
            len(phi_angles),
            config.constant_scaling_threshold,
            config.fourier_auto_threshold,
            effective_mode,
        )

        if effective_mode == "constant":
            # Lazy import: keeps the heterodyne_constant_mode module out of
            # heterodyne_core's namespace so ``hasattr(heterodyne_core,
            # '_fit_joint_constant_multi_phi')`` stays False (the Sub-PR A3
            # contract — the function lives in its own module, not here).
            from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
                _fit_joint_constant_multi_phi,
            )

            return _fit_joint_constant_multi_phi(
                model=model,
                c2_data=c2_data,
                phi_angles=phi_angles,
                config=config,
                weights=weights,
            )

        if effective_mode == "averaged":
            return _fit_joint_averaged_multi_phi(
                model=model,
                c2_data=c2_data,
                phi_angles=phi_angles,
                config=config,
                weights=weights,
            )

        if effective_mode == "fourier":
            try:
                from xpcsjax.optimization.nlsq.fourier_reparam import (
                    FourierReparamConfig,
                    FourierReparameterizer,
                )

                # NOTE: ``FourierReparamConfig.mode`` is typed as
                # ``Literal["independent", "fourier", "auto"]`` — a narrower
                # vocabulary than heterodyne's ``per_angle_mode``
                # (``"individual" | "fourier" | "auto" | "constant" |
                # "independent"``). We reach this branch only when the
                # resolver returned ``"fourier"``, so passing the literal
                # ``"fourier"`` is correct and silences the Pyright
                # incompatibility flagged since A1. The
                # ``FourierReparameterizer`` re-runs the auto/feasibility
                # check via ``_determine_mode`` and falls back to
                # ``independent`` internally if ``n_phi`` is too small for
                # the requested order — so we do not lose the auto-fallback
                # behaviour by pinning the string here.
                fourier_config = FourierReparamConfig(
                    mode="fourier",
                    fourier_order=config.fourier_order,
                    auto_threshold=config.fourier_auto_threshold,
                )
                phi_rad = np.deg2rad(phi_angles.astype(np.float64))
                fourier = FourierReparameterizer(phi_rad, fourier_config)
                use_joint = True
            except ImportError:
                logger.warning(
                    "fourier_reparam not available, falling back to sequential fits"
                )

        # effective_mode == "individual" falls through to sequential per-angle.

    if use_joint:
        # Invariant: ``use_joint`` is only set to True inside the
        # ``if config is not None and len(phi_angles) > 1`` block above,
        # so config is guaranteed non-None here. mypy can't see the implicit
        # invariant — assert it for the type checker and as a belt-and-
        # suspenders runtime check.
        assert config is not None, "use_joint=True only when config is non-None"
        return _fit_joint_multi_phi(
            model,
            c2_data,
            phi_angles,
            config,
            weights,
            fourier,
        )

    # ------------------------------------------------------------------
    # Sequential per-angle fitting (warm-start chain)
    # ------------------------------------------------------------------
    results = []
    for i, phi in enumerate(phi_angles):
        if i > 0:
            logger.info(
                "Fitting phi angle %d/%d: %s° (warm-start from angle %s°)",
                i + 1,
                len(phi_angles),
                phi,
                phi_angles[i - 1],
            )
        else:
            logger.info("Fitting phi angle %d/%d: %s°", i + 1, len(phi_angles), phi)

        c2_i = c2_data[i]
        weights_i = weights[i] if weights is not None and weights.ndim == 3 else weights

        result = fit_nlsq_jax(
            model=model,
            c2_data=c2_i,
            phi_angle=float(phi),
            config=config,
            weights=weights_i,
        )
        result.metadata["phi_angle"] = float(phi)
        results.append(result)

    return results


def _compute_per_angle_chi2(
    residuals: np.ndarray,
    c2_matrix: np.ndarray,
    n_params: int,
) -> tuple[float, float]:
    """Compute per-angle cost and noise-normalised reduced chi-squared.

    Joint fits produce one aggregated cost and chi2 for all angles. This
    helper reconstructs the per-angle statistics so each NLSQResult carries
    its own diagnostics rather than a copy of the joint value.

    Args:
        residuals: Flat off-diagonal residual vector from compute_residuals,
            length n*(n-1).
        c2_matrix: Per-angle experimental C2 matrix, shape (n, n).
        n_params: Number of varying physics parameters.

    Returns:
        ``(per_angle_cost, reduced_chi_squared)`` where ``per_angle_cost``
        is ``0.5*SSR`` and ``reduced_chi_squared`` is noise-normalised
        (target ≈ 1.0 for a good fit; MSE fallback when noise is degenerate).
    """
    ssr = float(np.sum(residuals**2))
    per_angle_cost = 0.5 * ssr

    n_matrix = c2_matrix.shape[0]
    n_valid = c2_matrix.size - n_matrix  # off-diagonal count (matches residuals length)
    n_dof = max(n_valid - n_params, 1)

    # Far-lag photon-noise estimate — same formula as _fit_local
    c2_np = np.asarray(c2_matrix)
    row_idx = np.arange(n_matrix)
    lag_mat = np.abs(row_idx[:, None] - row_idx[None, :])
    far_vals = c2_np[lag_mat >= n_matrix // 2]
    sigma2_noise = float(np.var(far_vals)) if far_vals.size > 1 else 0.0

    if sigma2_noise > 1e-12:
        reduced_chi2 = ssr / (sigma2_noise * n_dof)
    else:
        reduced_chi2 = ssr / n_dof  # MSE fallback

    return per_angle_cost, reduced_chi2


def _fit_joint_averaged_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
) -> OptimizationResult:
    """Joint multi-angle fit with averaged contrast/offset scaling.

    Implements homodyne's `auto`-averaged anti-degeneracy path:
    per-angle quantile estimates are computed first, averaged to one contrast
    and one offset, and those two scaling parameters are optimized jointly
    with the physical model parameters.

    NOTE: despite the legacy filename overlap, this is NOT homodyne's `constant`
    mode. True `constant` mode (quantile estimates pre-fit and frozen) is
    implemented by `fit_joint_constant_multi_phi` (Sub-PR B), defined in
    `heterodyne_constant_mode.py`.

    Returns
    -------
    OptimizationResult
        One result for the entire joint solve. ``parameters`` has the
        ``physics_varying + [avg_contrast, avg_offset]`` layout (2 scaling
        params). Per-angle diagnostics — ``chi2_per_angle``,
        ``per_angle_mode='averaged'``, ``scaling_source='averaged_then_fitted'``,
        ``fourier_basis_dim=None``, ``shear_weighting='not_applicable_heterodyne'``
        — live in ``nlsq_diagnostics``, alongside the ``averaged_contrast`` /
        ``averaged_offset`` scalar extras. Mirrors the contract of
        :func:`_fit_joint_multi_phi` (Sub-PR C2) and
        :func:`xpcsjax.optimization.nlsq.heterodyne_constant_mode._fit_joint_constant_multi_phi`
        (Sub-PR B2).
    """
    from xpcsjax.config.parameter_registry import SCALING_PARAMS
    from xpcsjax.core.heterodyne_scaling_utils import compute_averaged_scaling

    t_start = time.perf_counter()

    param_manager = model.param_manager
    varying_names = list(param_manager.varying_names)
    n_physics_varying = param_manager.n_varying
    n_phi = len(phi_angles)

    physics_initial = np.asarray(param_manager.get_initial_values(), dtype=np.float64)
    physics_lower, physics_upper = param_manager.get_bounds()
    physics_initial = np.clip(physics_initial, physics_lower, physics_upper)

    t = model.t
    q = model.q
    dt = model.dt

    t1_mesh, t2_mesh = np.meshgrid(np.asarray(t), np.asarray(t), indexing="ij")
    n_time_points = t1_mesh.size
    c2_flat = []
    t1_flat = []
    t2_flat = []
    phi_indices = []
    for i in range(n_phi):
        c2_flat.append(np.asarray(c2_data[i], dtype=np.float64).reshape(-1))
        t1_flat.append(t1_mesh.reshape(-1))
        t2_flat.append(t2_mesh.reshape(-1))
        phi_indices.append(np.full(n_time_points, i, dtype=np.int32))

    contrast_bounds = (
        SCALING_PARAMS["contrast"].min_bound,
        SCALING_PARAMS["contrast"].max_bound,
    )
    offset_bounds = (
        SCALING_PARAMS["offset"].min_bound,
        SCALING_PARAMS["offset"].max_bound,
    )

    logger.info("=" * 60)
    logger.info("AUTO AVERAGED SCALING: Computing per-angle scaling from quantiles")
    logger.info("=" * 60)
    avg_contrast, avg_offset, contrast_per_angle, offset_per_angle = (
        compute_averaged_scaling(
            c2_data=np.concatenate(c2_flat),
            t1=np.concatenate(t1_flat),
            t2=np.concatenate(t2_flat),
            phi_indices=np.concatenate(phi_indices),
            n_phi=n_phi,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
            log=logger,
        )
    )

    x0 = np.concatenate([physics_initial, [avg_contrast, avg_offset]])
    lb = np.concatenate([physics_lower, [contrast_bounds[0], offset_bounds[0]]])
    ub = np.concatenate([physics_upper, [contrast_bounds[1], offset_bounds[1]]])
    joint_param_names = [*varying_names, "contrast", "offset"]

    logger.info(
        "Joint auto averaged fit: %d physical + 2 averaged scaling = %d total params, %d angles",
        n_physics_varying,
        len(x0),
        n_phi,
    )

    c2_data_batch = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_batch = (
        jnp.asarray(weights, dtype=jnp.float64)
        if weights is not None
        else jnp.ones_like(c2_data_batch)
    )
    if weights_batch.ndim == 2:
        weights_batch = jnp.broadcast_to(weights_batch, c2_data_batch.shape)
    phi_angles_jax = jnp.asarray(phi_angles, dtype=jnp.float64)
    fixed_values_jax = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(param_manager.varying_indices, dtype=jnp.int32)

    # NOTE: must return a JAX array. NLSQ's masked_residual_func JIT-traces this
    # closure; np.asarray() on a traced result raises TracerArrayConversionError.
    def joint_residual_fn(x: np.ndarray) -> Any:  # type: ignore[return-value]
        physics_varying = x[:n_physics_varying]
        contrast = x[n_physics_varying]
        offset = x[n_physics_varying + 1]

        full_jax = fixed_values_jax.at[varying_indices_jax].set(
            jnp.asarray(physics_varying, dtype=jnp.float64)
        )
        contrasts_jax = jnp.full((n_phi,), contrast, dtype=jnp.float64)
        offsets_jax = jnp.full((n_phi,), offset, dtype=jnp.float64)
        return compute_multi_angle_residuals(
            full_jax,
            t,
            q,
            dt,
            phi_angles_jax,
            c2_data_batch,
            weights_batch,
            contrasts_jax,
            offsets_jax,
        )

    joint_config = NLSQConfig(
        method=config.method if config.method != "lm" else "trf",
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        max_nfev=(config.max_nfev * n_phi if config.max_nfev is not None else None),
        loss=config.loss,
        use_nlsq_library=config.use_nlsq_library,
        n_params=len(x0),
    )

    joint_result: NLSQResult | None = None
    # Narrow via ``is not None`` instead of the HAS_X flag so Pyright sees
    # NLSQAdapter as bound. HAS_ADAPTERS is True iff NLSQAdapter was imported,
    # so the two predicates are equivalent at runtime.
    if NLSQAdapter is not None:
        try:
            joint_adapter = NLSQAdapter(parameter_names=joint_param_names)
            joint_result = joint_adapter.fit(
                residual_fn=joint_residual_fn,
                initial_params=x0,
                bounds=(lb, ub),
                config=joint_config,
            )
            if not joint_result.success:
                raise RuntimeError(
                    f"Joint adapter returned success=False: {joint_result.message}"
                )
        except (ValueError, RuntimeError, TypeError) as adapter_exc:
            logger.warning(
                "Joint auto averaged NLSQAdapter failed, falling back to NLSQWrapper: %s",
                adapter_exc,
            )
            joint_result = None

    if joint_result is None and NLSQWrapper is not None:
        joint_wrapper = NLSQWrapper(parameter_names=joint_param_names)
        joint_result = joint_wrapper.fit(
            residual_fn=joint_residual_fn,
            initial_params=x0,
            bounds=(lb, ub),
            config=joint_config,
        )

    if joint_result is None:
        raise ImportError(
            "No NLSQ backend available for joint auto averaged multi-angle fit."
        )

    fitted_all = np.asarray(joint_result.parameters, dtype=np.float64)
    fitted_physics = fitted_all[:n_physics_varying]
    fitted_contrast = float(fitted_all[n_physics_varying])
    fitted_offset = float(fitted_all[n_physics_varying + 1])

    full_fitted = param_manager.expand_varying_to_full(fitted_physics)
    model.set_params(full_fitted)
    if hasattr(model, "scaling"):
        model.scaling.contrast[:] = fitted_contrast
        model.scaling.offset[:] = fitted_offset

    wall_time = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Decompose per-angle chi^2 from the final residual.
    # ``compute_multi_angle_residuals`` returns an angle-major flat layout
    # (n_phi, n_per_angle) — n_per_angle = n_time * (n_time - 1) because the
    # kernel excludes the diagonal. Re-use the canonical helper from
    # heterodyne_constant_mode (same import the Fourier-mode joint path uses).
    # ------------------------------------------------------------------
    from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
        _decompose_chi2_per_angle,
    )

    final_residual = np.asarray(joint_residual_fn(fitted_all))
    n_time = c2_data.shape[1]
    n_per_angle = n_time * (n_time - 1)  # off-diagonal only — matches kernel
    chi2_per_angle = _decompose_chi2_per_angle(
        final_residual=final_residual,
        n_phi=n_phi,
        n_per_angle=n_per_angle,
    )

    # ------------------------------------------------------------------
    # Build the single joint OptimizationResult.
    # SSR conservation: ``chi_squared`` is the raw residual SSR, not
    # ``2 * nlsq_result.final_cost`` (which is the robust-loss cost when
    # ``config.loss != "linear"``). Using raw residuals keeps
    # ``chi2_per_angle.sum() == chi_squared`` for every loss choice —
    # the same invariant B2 / C2 locked in for the other joint paths.
    # ------------------------------------------------------------------
    ssr = float(np.sum(final_residual**2))
    n_total_params = int(joint_result.parameters.size)
    n_dof = max(final_residual.size - n_total_params, 1)
    reduced_chi2 = (
        float(joint_result.reduced_chi_squared)
        if joint_result.reduced_chi_squared is not None
        else ssr / n_dof
    )

    # NaN-fill uncertainties / covariance when the NLSQ adapter could not
    # produce them (e.g. singular Jacobian after a non-converged solve) —
    # matches B2 / C2's contract so consumers see a uniform array shape.
    uncertainties = (
        np.asarray(joint_result.uncertainties, dtype=np.float64)
        if joint_result.uncertainties is not None
        else np.full(n_total_params, np.nan, dtype=np.float64)
    )
    covariance = (
        np.asarray(joint_result.covariance, dtype=np.float64)
        if joint_result.covariance is not None
        else np.full((n_total_params, n_total_params), np.nan, dtype=np.float64)
    )

    convergence_status = "converged" if joint_result.success else "failed"
    quality_flag = "good" if joint_result.success else "marginal"

    diagnostics = _build_heterodyne_diagnostics(
        per_angle_mode="averaged",
        chi2_per_angle=chi2_per_angle,
        scaling_source="averaged_then_fitted",
        fourier_basis_dim=None,
        averaged_contrast=fitted_contrast,
        averaged_offset=fitted_offset,
        parameter_names=joint_param_names,
        contrast_per_angle_quantile=np.asarray(contrast_per_angle, dtype=np.float64),
        offset_per_angle_quantile=np.asarray(offset_per_angle, dtype=np.float64),
        contrast_initial_average=float(avg_contrast),
        offset_initial_average=float(avg_offset),
        phi_angles=np.asarray(phi_angles, dtype=np.float64),
        n_angles_joint=n_phi,
        convergence_reason=joint_result.convergence_reason,
        n_function_evals=int(joint_result.n_function_evals or 0),
        n_iterations=int(joint_result.n_iterations or 0),
        wall_time_seconds=wall_time,
        message=str(joint_result.message),
    )

    logger.info(
        "Joint auto averaged fit complete: success=%s, cost=%.6f, "
        "n_evals=%d, wall_time=%.2fs, %d angles",
        joint_result.success,
        joint_result.final_cost or 0.0,
        joint_result.n_function_evals or 0,
        wall_time,
        n_phi,
    )

    return OptimizationResult(
        parameters=np.asarray(fitted_all, dtype=np.float64),
        uncertainties=uncertainties,
        covariance=covariance,
        chi_squared=ssr,
        reduced_chi_squared=reduced_chi2,
        convergence_status=convergence_status,
        iterations=int(joint_result.n_iterations or 0),
        execution_time=wall_time,
        device_info={"backend": "cpu", "adapter": "nlsq.CurveFit"},
        recovery_actions=[],
        quality_flag=quality_flag,
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics=diagnostics,
    )


# Phase-6 stub: parameters are renamed with leading underscore to silence
# both ruff (ARG001) and Pyright (reportUnusedParameter). Original public
# names will return when the body is wired against the new cmaes_wrapper
# signature; see docstring below for the migration plan.
def _fit_joint_cmaes_multi_phi(
    _model: HeterodyneModel,
    _c2_data: np.ndarray,
    _phi_angles: np.ndarray,
    _config: NLSQConfig,
    _weights: np.ndarray | None,
) -> list[NLSQResult]:
    """Joint multi-angle CMA-ES with NLSQ warm-start — not wired in v0.1.

    The previous port called ``fit_with_cmaes`` with a homemade keyword API
    (``objective_fn=, residual_fn=, parameter_names=, n_data=, anti_degeneracy=,
    config=CMAESConfig(...)``). Both the keyword set and ``CMAESConfig`` are
    gone from the current ``cmaes_wrapper``: the convenience entry accepts
    ``(model_func, xdata, ydata, p0, bounds, sigma, config: CMAESWrapperConfig)``
    and the result type exposes ``chi_squared`` rather than ``final_cost``,
    ``uncertainties``, ``n_iterations``, etc. that the old joint-fit code
    consumed directly. Rewriting this whole 340-line function against the
    new contract is Phase 6 work (it also needs to thread Fourier-reparam
    anti-degeneracy scaling through the wrapper's ``model_func`` closure).

    Until that lands, the **per-angle** ``_fit_cmaes`` path
    (single ``phi_angle``) is the supported global-search route for
    heterodyne. ``fit_nlsq_multi_phi`` should leave this branch unreached
    in v0.1 — if you hit this NotImplementedError, set
    ``optimization.nlsq.cmaes.joint_multi_phi: false`` (or unset it) in your
    heterodyne config and run per-angle.
    """
    raise NotImplementedError(
        "Joint multi-angle CMA-ES for heterodyne is not wired in v0.1. Use the "
        "per-angle CMA-ES path (default) by leaving "
        "`optimization.nlsq.cmaes.joint_multi_phi` unset, or disable CMA-ES "
        "globally. Tracked for Phase 6: needs rewrite against "
        "cmaes_wrapper.fit_with_cmaes's positional signature + Fourier-reparam "
        "scaling integration."
    )


def _resolve_effective_mode(config: NLSQConfig, n_phi: int) -> str:
    """Map ``config.per_angle_mode`` + ``n_phi`` to a canonical dispatch token.

    Returns one of:

    * ``"constant"`` — frozen per-angle (β, ō) from diagonal-quantile estimator;
      optimizer dimension is ``n_physics_varying`` only.
    * ``"averaged"`` — one (β̄, ō̄) pair optimized jointly with physics. This
      is the homodyne ``auto``-averaged anti-degeneracy path.
    * ``"fourier"`` — Fourier-basis reparameterization of per-angle scaling
      (smooth angular variation).
    * ``"individual"`` — sequential per-angle fits with warm-start chaining.

    ``auto`` threshold semantics match homodyne::

        n_phi <  constant_scaling_threshold (3) -> "constant"
        n_phi <  fourier_auto_threshold     (6) -> "averaged"
        n_phi >= fourier_auto_threshold     (6) -> "fourier"

    Explicit modes (``"constant"``, ``"fourier"``, ``"individual"``) pass
    through unchanged. The legacy alias ``"independent"`` is already rewritten
    to ``"individual"`` by :meth:`NLSQConfig.__post_init__`.
    """
    requested = config.per_angle_mode
    if requested == "constant":
        return "constant"
    if requested == "fourier":
        return "fourier"
    if requested == "individual":
        return "individual"
    # requested == "auto" — route by n_phi
    constant_threshold = max(int(config.constant_scaling_threshold), 1)
    fourier_threshold = max(int(config.fourier_auto_threshold), 1)
    if n_phi < constant_threshold:
        return "constant"
    if n_phi < fourier_threshold:
        return "averaged"
    return "fourier"


def _fit_joint_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
    fourier: Any,
) -> OptimizationResult:
    """Joint multi-angle fit with Fourier-parameterized scaling.

    The optimizer parameter vector is:
        [physics_varying_params | fourier_contrast_coeffs | fourier_offset_coeffs]

    The residual function evaluates all angles, using the Fourier basis to
    convert coefficients → per-angle contrast/offset at each evaluation.

    This is the heterodyne equivalent of homodyne's AntiDegeneracyController
    joint-fit path.

    Returns
    -------
    OptimizationResult
        One result for the entire joint solve.  ``parameters`` has the
        full ``physics_varying + 2*(2K+1)`` layout (K = ``config.fourier_order``).
        Per-angle diagnostics — ``chi2_per_angle``, ``fourier_basis_dim``,
        ``per_angle_mode='fourier'``, ``scaling_source='fitted'``,
        ``shear_weighting='not_applicable_heterodyne'`` — live in
        ``nlsq_diagnostics``.  Mirrors the contract of
        :func:`xpcsjax.optimization.nlsq.heterodyne_constant_mode._fit_joint_constant_multi_phi`
        (Sub-PR B2).
    """
    t_start = time.perf_counter()

    param_manager = model.param_manager
    varying_names = param_manager.varying_names
    n_physics_varying = param_manager.n_varying
    n_phi = len(phi_angles)

    # Physics parameter initial values and bounds
    physics_initial = param_manager.get_initial_values()
    physics_lower, physics_upper = param_manager.get_bounds()
    physics_initial = np.clip(physics_initial, physics_lower, physics_upper)

    # Fourier coefficient initial values and bounds
    scaling = model.scaling
    contrast_init = float(scaling.contrast[0]) if len(scaling.contrast) > 0 else 0.5
    offset_init = float(scaling.offset[0]) if len(scaling.offset) > 0 else 1.0
    fourier_initial = fourier.get_initial_coefficients(contrast_init, offset_init)
    fourier_lower, fourier_upper = fourier.get_bounds()

    # Combined parameter vector
    x0 = np.concatenate([physics_initial, fourier_initial])
    lb = np.concatenate([physics_lower, fourier_lower])
    ub = np.concatenate([physics_upper, fourier_upper])

    logger.info(
        "Joint multi-angle fit: %d physics + %d Fourier = %d total params, %d angles",
        n_physics_varying,
        fourier.n_coeffs,
        len(x0),
        n_phi,
    )

    # Pre-convert data to JAX arrays (outside closure — constants)
    t, q, dt = model.t, model.q, model.dt
    c2_data_list = [jnp.asarray(c2_data[i], dtype=jnp.float64) for i in range(n_phi)]
    weights_list: list[jnp.ndarray | None] = []
    for i in range(n_phi):
        if weights is not None and weights.ndim == 3:
            weights_list.append(jnp.asarray(weights[i], dtype=jnp.float64))
        elif weights is not None:
            weights_list.append(jnp.asarray(weights, dtype=jnp.float64))
        else:
            weights_list.append(None)

    # Pre-stack batched arrays for compute_multi_angle_residuals.
    # weights_list entries may be None (unweighted) — materialise ones_like
    # so the stacked weights_batch is always a concrete (n_phi, N, N) array.
    c2_data_batch = jnp.stack(c2_data_list, axis=0)  # (n_phi, N, N)
    weights_batch = jnp.stack(
        [
            (w if w is not None else jnp.ones_like(c2_data_list[i]))
            for i, w in enumerate(weights_list)
        ],
        axis=0,
    )  # (n_phi, N, N)
    phi_angles_jax = jnp.asarray(phi_angles, dtype=jnp.float64)  # (n_phi,)

    fixed_values_jax = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(param_manager.varying_indices, dtype=jnp.int32)

    # NOTE: must return a JAX array. NLSQ's masked_residual_func JIT-traces
    # this closure; calling ``np.asarray`` on a traced result raises
    # TracerArrayConversionError. Same fix as
    # ``_fit_joint_averaged_multi_phi`` / ``_fit_joint_constant_multi_phi``
    # — the kernel returns ``jnp.ndarray`` and NLSQ casts at its boundary.
    def joint_residual_fn(x: np.ndarray) -> Any:  # type: ignore[return-value]
        """Compute concatenated residuals across all angles via vmap.

        Routes through ``compute_multi_angle_residuals`` (jit + vmap) to
        replace the previous n_phi serial kernel dispatches with a single
        batched XLA call.  Fourier reparameterization is preserved: the
        combined parameter vector is split into physics and Fourier parts,
        and ``fourier.fourier_to_per_angle`` converts coefficients to
        per-angle contrast/offset arrays before the batched residual call.
        """
        # Split combined vector
        physics_varying = x[:n_physics_varying]
        fourier_coeffs = x[n_physics_varying:]

        # Reconstruct full physics parameter array (immutable JAX scatter)
        varying_jax = jnp.asarray(physics_varying, dtype=jnp.float64)
        full_jax = fixed_values_jax.at[varying_indices_jax].set(varying_jax)

        # Convert Fourier coefficients → per-angle contrast/offset
        contrast_arr, offset_arr = fourier.fourier_to_per_angle(fourier_coeffs)
        contrasts_jax = jnp.asarray(contrast_arr, dtype=jnp.float64)  # (n_phi,)
        offsets_jax = jnp.asarray(offset_arr, dtype=jnp.float64)  # (n_phi,)

        # Single batched vmap call — eliminates n_phi serial dispatches
        return compute_multi_angle_residuals(
            full_jax,
            t,
            q,
            dt,
            phi_angles_jax,
            c2_data_batch,
            weights_batch,
            contrasts_jax,
            offsets_jax,
        )

    # Run optimization via NLSQAdapter (primary) with NLSQWrapper fallback
    joint_config = NLSQConfig(
        method=config.method if config.method != "lm" else "trf",
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        max_nfev=(config.max_nfev * n_phi if config.max_nfev is not None else None),
    )

    joint_result: NLSQResult | None = None
    joint_param_names = list(varying_names) + [
        f"fourier_{i}" for i in range(len(fourier_initial))
    ]

    if NLSQAdapter is not None:  # ``HAS_ADAPTERS`` equivalent; narrows for Pyright
        try:
            joint_adapter = NLSQAdapter(parameter_names=joint_param_names)
            joint_result = joint_adapter.fit(
                residual_fn=joint_residual_fn,
                initial_params=x0,
                bounds=(lb, ub),
                config=joint_config,
            )
            if not joint_result.success:
                raise RuntimeError(
                    f"Joint adapter returned success=False: {joint_result.message}"
                )
        except (ValueError, RuntimeError, TypeError) as adapter_exc:
            logger.warning(
                "Joint NLSQAdapter failed, falling back to NLSQWrapper: %s", adapter_exc
            )
            joint_result = None

    if joint_result is None and NLSQWrapper is not None:
        joint_wrapper = NLSQWrapper(parameter_names=joint_param_names)
        joint_result = joint_wrapper.fit(
            residual_fn=joint_residual_fn,
            initial_params=x0,
            bounds=(lb, ub),
            config=joint_config,
        )

    if joint_result is None:
        raise ImportError(
            "No NLSQ backend available for joint multi-angle fit. "
            "Ensure heterodyne.optimization.nlsq.adapter is importable."
        )

    # Extract results
    fitted_params_full = joint_result.parameters
    fitted_physics = fitted_params_full[:n_physics_varying]
    fitted_fourier = fitted_params_full[n_physics_varying:]
    fitted_contrast, fitted_offset = fourier.fourier_to_per_angle(fitted_fourier)

    # Update model with fitted physics parameters
    full_fitted = param_manager.expand_varying_to_full(fitted_physics)
    model.set_params(full_fitted)

    # Update model scaling
    if len(scaling.contrast) == n_phi:
        scaling.contrast[:] = fitted_contrast
        scaling.offset[:] = fitted_offset

    wall_time = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Decompose per-angle chi^2 from the final residual.
    # ``compute_multi_angle_residuals`` returns an angle-major flat layout
    # (n_phi, n_per_angle) — n_per_angle = n_time * (n_time - 1) because the
    # kernel excludes the diagonal. Re-import the helper from the constant-
    # mode module to keep one canonical implementation.
    # TODO(C3): consolidate _decompose_chi2_per_angle when the averaged path
    # also returns OptimizationResult, so all three joint paths share the
    # same helper without crossing module boundaries.
    # ------------------------------------------------------------------
    from xpcsjax.optimization.nlsq.heterodyne_constant_mode import (
        _decompose_chi2_per_angle,
    )

    final_residual = np.asarray(joint_residual_fn(fitted_params_full))
    n_time = c2_data.shape[1]
    n_per_angle = n_time * (n_time - 1)  # off-diagonal only — matches kernel
    chi2_per_angle = _decompose_chi2_per_angle(
        final_residual=final_residual,
        n_phi=n_phi,
        n_per_angle=n_per_angle,
    )

    # ------------------------------------------------------------------
    # Build the single joint OptimizationResult.
    # ------------------------------------------------------------------
    # SSR conservation: ``chi_squared`` is the raw residual SSR, not
    # ``2 * nlsq_result.final_cost`` (which is the robust-loss cost when
    # ``config.loss != "linear"``). Using raw residuals keeps
    # ``chi2_per_angle.sum() == chi_squared`` for every loss choice —
    # the same invariant B2 locked in for constant mode.
    ssr = float(np.sum(final_residual**2))
    n_total_params = int(joint_result.parameters.size)
    n_dof = max(final_residual.size - n_total_params, 1)
    reduced_chi2 = (
        float(joint_result.reduced_chi_squared)
        if joint_result.reduced_chi_squared is not None
        else ssr / n_dof
    )

    # NaN-fill uncertainties/covariance when the NLSQ adapter could not
    # produce them (e.g. singular Jacobian after a non-converged solve) —
    # matches B2's contract so consumers see a uniform array shape.
    uncertainties = (
        np.asarray(joint_result.uncertainties, dtype=np.float64)
        if joint_result.uncertainties is not None
        else np.full(n_total_params, np.nan, dtype=np.float64)
    )
    covariance = (
        np.asarray(joint_result.covariance, dtype=np.float64)
        if joint_result.covariance is not None
        else np.full((n_total_params, n_total_params), np.nan, dtype=np.float64)
    )

    convergence_status = "converged" if joint_result.success else "failed"
    quality_flag = "good" if joint_result.success else "marginal"

    diagnostics = _build_heterodyne_diagnostics(
        per_angle_mode="fourier",
        chi2_per_angle=chi2_per_angle,
        scaling_source="fitted",
        fourier_basis_dim=fourier.n_coeffs_per_param,
        parameter_names=joint_param_names,
        fourier_mode=fourier.config.mode,
        fourier_order=fourier.order,
        fourier_coeffs=fitted_fourier.tolist(),
        fourier_n_coeffs=fourier.n_coeffs,
        fourier_reduction=fourier.get_diagnostics()["reduction_ratio"],
        contrast_per_angle_fitted=np.asarray(fitted_contrast, dtype=np.float64),
        offset_per_angle_fitted=np.asarray(fitted_offset, dtype=np.float64),
        phi_angles=np.asarray(phi_angles, dtype=np.float64),
        n_angles_joint=n_phi,
        convergence_reason=joint_result.convergence_reason,
        n_function_evals=int(joint_result.n_function_evals or 0),
        n_iterations=int(joint_result.n_iterations or 0),
        wall_time_seconds=wall_time,
        message=str(joint_result.message),
    )

    logger.info(
        "Joint multi-angle fit complete: success=%s, cost=%.6f, "
        "n_evals=%d, wall_time=%.2fs, %d angles",
        joint_result.success,
        joint_result.final_cost or 0.0,
        joint_result.n_function_evals or 0,
        wall_time,
        n_phi,
    )

    return OptimizationResult(
        parameters=np.asarray(fitted_params_full, dtype=np.float64),
        uncertainties=uncertainties,
        covariance=covariance,
        chi_squared=ssr,
        reduced_chi_squared=reduced_chi2,
        convergence_status=convergence_status,
        iterations=int(joint_result.n_iterations or 0),
        execution_time=wall_time,
        device_info={"backend": "cpu", "adapter": "nlsq.CurveFit"},
        recovery_actions=[],
        quality_flag=quality_flag,
        streaming_diagnostics=None,
        stratification_diagnostics=None,
        nlsq_diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_global_optimization(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    weights: np.ndarray | jnp.ndarray | None,
    use_nlsq_library: bool,
) -> NLSQResult | None:
    """Attempt CMA-ES or multi-start if configured.

    Returns the result if a global method was selected, or ``None`` to
    fall through to local optimization.
    """
    # CMA-ES has highest priority
    if getattr(config, "enable_cmaes", False):
        if HAS_CMAES:
            logger.info("CMA-ES enabled, delegating to fit_with_cmaes")
            return _fit_cmaes(model, c2_data, phi_angle, config, weights)
        logger.warning(
            "CMA-ES enabled in config but not available (cma not installed). "
            "Install with: uv add cma. Falling back."
        )

    # Multi-start is second priority
    if getattr(config, "multistart", False):
        if HAS_MULTISTART:
            logger.info("Multi-start enabled, delegating to multi-start optimizer")
            return _fit_multistart(
                model,
                c2_data,
                phi_angle,
                config,
                weights,
                use_nlsq_library,
            )
        logger.warning(
            "Multi-start enabled in config but multistart module not available. "
            "Falling back to local optimization."
        )

    return None


def _fit_cmaes(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    weights: np.ndarray | jnp.ndarray | None,
) -> NLSQResult:
    """Run CMA-ES global optimization with NLSQ warm-start and two-phase comparison.

    Phase structure (mirrors the homodyne CMA-ES path):

    - **Phase 1**: Local NLSQ refinement to get a warm-start point.
    - **Phase 2**: CMA-ES global search using the NLSQ result as initial guess.
      Calls :func:`xpcsjax.optimization.nlsq.cmaes_wrapper.fit_with_cmaes`
      with its real positional signature
      ``(model_func, xdata, ydata, p0, bounds, sigma, config)``. The previous
      port called it with a homemade keyword API
      (``objective_fn=, residual_fn=, n_data=, anti_degeneracy=``) that no
      longer exists; mypy flagged it and the smoke tests never reached the
      branch. Fixed here so v0.1 actually delivers on the "CMA-ES global
      search for multi-scale problems" claim for heterodyne.
    - **Phase 3**: Compare NLSQ vs CMA-ES by least-squares cost, keep the
      better result. ``CMAESResult`` exposes ``chi_squared`` (sum of squared
      residuals); we halve it to compare against NLSQ's
      ``final_cost = 0.5 * SSR`` convention.
    """
    from xpcsjax.optimization.nlsq.cmaes_wrapper import CMAESWrapperConfig

    param_manager = model.param_manager

    initial_varying = param_manager.get_initial_values()
    lower_bounds, upper_bounds = param_manager.get_bounds()
    initial_varying = np.clip(initial_varying, lower_bounds, upper_bounds)

    c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_jax = (
        jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
    )
    t, q, dt = model.t, model.q, model.dt
    contrast_val, offset_val = model.scaling.get_for_angle(0)

    # ------------------------------------------------------------------
    # Phase 1: NLSQ warm-start
    # ------------------------------------------------------------------
    nlsq_result: NLSQResult | None = None
    cmaes_x0 = initial_varying

    try:
        logger.info("CMA-ES Phase 1: NLSQ warm-start refinement")
        nlsq_result = _fit_local(
            model,
            c2_data,
            phi_angle,
            config,
            weights,
            use_nlsq_library=config.use_nlsq_library,
        )
        if nlsq_result.success:
            cmaes_x0 = nlsq_result.parameters.copy()
            logger.info(
                "NLSQ warm-start succeeded: cost=%.6e, chi2_red=%.4f",
                nlsq_result.final_cost or float("inf"),
                nlsq_result.reduced_chi_squared or float("inf"),
            )
        else:
            logger.warning(
                "NLSQ warm-start failed (%s), using raw initial params for CMA-ES",
                nlsq_result.message,
            )
    except (ValueError, RuntimeError, ImportError) as e:
        logger.warning(
            "NLSQ warm-start raised %s: %s — proceeding with raw p0",
            type(e).__name__,
            e,
        )

    # Ensure model parameters are reset for CMA-ES (NLSQ may have modified them)
    model.set_params(param_manager.expand_varying_to_full(initial_varying))

    # ------------------------------------------------------------------
    # Phase 2: CMA-ES global optimization
    # ------------------------------------------------------------------
    # Build the ``model_func(xdata, *params) -> ydata_flat`` closure that
    # fit_with_cmaes requires. xdata is a dummy index array — the heterodyne
    # kernel pulls t/q/dt/phi/contrast/offset from closure, not from xdata.
    #
    # IMPORTANT (tracer-safety): CMA-ES wraps this closure in
    # ``normalized_model_func`` (cmaes_wrapper.py:967) which passes JAX
    # *tracers* for ``varying_params`` when JIT-tracing the parameter
    # normalization. Mixing numpy assignment (``full[idx] = tracer``) with
    # tracer values raises ``ValueError: setting an array element with a
    # sequence``. Use pure-JAX scatter (``.at[].set()``) instead so the
    # closure JIT-traces cleanly.
    full_template_jax = jnp.asarray(
        param_manager.get_full_values(), dtype=jnp.float64
    )
    varying_indices_jax = jnp.asarray(
        list(param_manager.varying_indices), dtype=jnp.int32
    )

    def model_func(_: np.ndarray, *varying_params: Any) -> Any:
        varying_jax = jnp.stack(varying_params).astype(jnp.float64)
        full_jax = full_template_jax.at[varying_indices_jax].set(varying_jax)
        c2_pred = compute_c2_heterodyne(
            full_jax, t, q, dt, phi_angle, contrast_val, offset_val
        )
        return c2_pred.flatten()

    ydata = np.asarray(c2_jax).flatten().astype(np.float64)
    xdata = np.arange(ydata.size, dtype=np.float64)
    if weights_jax is not None:
        weights_np = np.asarray(weights_jax).flatten().astype(np.float64)
        # weights = 1/σ² ⇒ σ = 1/√weights. Guard zeros (unweighted samples)
        # by passing σ = 1 there so they fall back to uniform weighting.
        safe_w = np.where(weights_np > 0, weights_np, 1.0)
        sigma = 1.0 / np.sqrt(safe_w)
    else:
        sigma = None

    # Build the wrapper config directly. Don't use
    # ``CMAESWrapperConfig.from_nlsq_config(config)`` here: that helper expects
    # the *homodyne* :class:`NLSQConfig` (different module, different field
    # names — heterodyne uses ``cmaes_tolx`` / ``cmaes_tolfun`` /
    # ``cmaes_max_iterations`` where homodyne has ``cmaes_tol_x`` /
    # ``cmaes_tol_fun`` / ``cmaes_max_generations``). Pyright correctly flags
    # the cross-class pass; mapping the heterodyne fields by hand is the right
    # answer until the two NLSQConfigs converge in Phase 6.
    cmaes_wrapper_config = CMAESWrapperConfig(
        max_generations=getattr(config, "cmaes_max_iterations", None),
        popsize=getattr(config, "cmaes_population_size", None),
        tol_x=float(getattr(config, "cmaes_tolx", 1e-8)),
        tol_fun=float(getattr(config, "cmaes_tolfun", 1e-8)),
        restart_strategy=str(getattr(config, "cmaes_restart_strategy", "bipop")),
        max_restarts=int(getattr(config, "cmaes_max_restarts", 9)),
    )
    logger.info("CMA-ES Phase 2: global search (warm-started)")
    # Invariant: this function is only entered from ``_try_global_optimization``
    # when ``HAS_CMAES`` is True, which is True iff ``fit_with_cmaes`` was
    # imported. Narrow for Pyright.
    assert fit_with_cmaes is not None, "HAS_CMAES guards entry to _fit_cmaes"
    cmaes_result = fit_with_cmaes(
        model_func=model_func,
        xdata=xdata,
        ydata=ydata,
        p0=np.asarray(cmaes_x0, dtype=np.float64),
        bounds=(lower_bounds, upper_bounds),
        sigma=sigma,
        config=cmaes_wrapper_config,
    )

    # ------------------------------------------------------------------
    # Phase 3: Compare NLSQ vs CMA-ES, keep the better result
    # ------------------------------------------------------------------
    nlsq_cost = (
        float(nlsq_result.final_cost)
        if (nlsq_result and nlsq_result.success and nlsq_result.final_cost is not None)
        else float("inf")
    )
    cmaes_cost = (
        0.5 * float(cmaes_result.chi_squared)
        if (cmaes_result.success and cmaes_result.chi_squared is not None)
        else float("inf")
    )

    if nlsq_cost <= cmaes_cost and nlsq_result is not None and nlsq_result.success:
        result = nlsq_result
        winner = "nlsq"
        logger.info(
            "Phase 3: NLSQ wins (cost=%.6e vs CMA-ES=%.6e)",
            nlsq_cost,
            cmaes_cost,
        )
    else:
        result = _cmaes_to_nlsq_result(
            cmaes_result, cmaes_cost, parameter_names=param_manager.varying_names
        )
        winner = "cmaes"
        logger.info(
            "Phase 3: CMA-ES wins (cost=%.6e vs NLSQ=%.6e)",
            cmaes_cost,
            nlsq_cost,
        )

    # ------------------------------------------------------------------
    # Post-fit: update model, classify quality
    # ------------------------------------------------------------------
    if result.success:
        full_fitted = param_manager.expand_varying_to_full(result.parameters)
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted), t, q, dt, phi_angle, contrast_val, offset_val
        )
        result.fitted_correlation = np.asarray(fitted_c2)
        model.set_params(full_fitted)

    # Apply same chi2 correction as _fit_local (DOF + σ² normalization)
    if result.final_cost is not None:
        n_matrix = c2_jax.shape[0]
        n_valid = c2_jax.size - n_matrix
        n_dof_valid = max(n_valid - len(param_manager.varying_names), 1)
        c2_np = np.asarray(c2_jax)
        row_idx = np.arange(n_matrix)
        lag_mat = np.abs(row_idx[:, None] - row_idx[None, :])
        far_vals = c2_np[lag_mat >= n_matrix // 2]
        sigma2_noise = float(np.var(far_vals)) if far_vals.size > 1 else 0.0
        if sigma2_noise > 1e-12:
            ssr = 2.0 * result.final_cost
            result.reduced_chi_squared = ssr / (sigma2_noise * n_dof_valid)

    quality_flag = classify_fit_quality(result.reduced_chi_squared)
    result.metadata["optimizer"] = "cmaes"
    result.metadata["cmaes_winner"] = winner
    result.metadata["cmaes_cost"] = cmaes_cost
    result.metadata["nlsq_warmstart_cost"] = nlsq_cost
    result.metadata["quality_flag"] = quality_flag

    _log_result(result)
    return result


def _cmaes_to_nlsq_result(
    cmaes_result: Any,
    final_cost: float,
    *,
    parameter_names: list[str],
) -> NLSQResult:
    """Pack a :class:`CMAESResult` into the :class:`NLSQResult` shape so
    downstream consumers (DOF correction, post-fit logging, multi-phi joining)
    see a uniform structure regardless of which optimizer won Phase 3.

    Naming convention: ``final_cost = 0.5 * SSR`` matches NLSQ's least-squares
    convention; CMA-ES reports ``chi_squared = SSR`` so the caller already
    halved it before passing it in.
    """
    diag = dict(cmaes_result.diagnostics) if cmaes_result.diagnostics else {}
    return NLSQResult(
        parameters=np.asarray(cmaes_result.parameters),
        parameter_names=list(parameter_names),
        success=bool(cmaes_result.success),
        message=str(cmaes_result.message),
        covariance=np.asarray(cmaes_result.covariance)
        if cmaes_result.covariance is not None
        else None,
        final_cost=final_cost,
        n_iterations=int(diag.get("generations", 0)),
        n_function_evals=int(diag.get("evaluations", 0)),
        convergence_reason=str(diag.get("convergence_reason", "")),
        metadata={"cmaes_diagnostics": diag},
    )


# Phase-6 stub: parameters are renamed with leading underscore to silence
# both ruff (ARG001) and Pyright (reportUnusedParameter). The signature
# shape (and parameter order) is kept identical to _fit_cmaes / _fit_local
# for the dispatcher at line ~958, which calls this positionally; the
# original public names will return when the body is wired against
# run_multistart_nlsq.
def _fit_multistart(
    _model: HeterodyneModel,
    _c2_data: np.ndarray | jnp.ndarray,
    _phi_angle: float,
    _config: NLSQConfig,
    _weights: np.ndarray | jnp.ndarray | None,
    _use_nlsq_library: bool,
) -> NLSQResult:
    """Heterodyne multi-start optimization — not wired in v0.1.

    The previous port called the upstream homodyne ``MultiStartOptimizer``
    class API (``MultiStartOptimizer(adapter=…, config=…).fit(…)``), but
    xpcsjax's :mod:`xpcsjax.optimization.nlsq.multistart` only exposes the
    function :func:`run_multistart_nlsq` with signature
    ``(data, bounds, config, single_fit_func, …)``. The class-style alias at
    the top of this module rebinds the function under the upstream name, but
    that rebind cannot satisfy ``optimizer.fit(...)`` — the call would crash
    with ``AttributeError: 'function' object has no attribute 'fit'`` the
    first time a user enabled ``optimization.nlsq.multi_start.enable: true``
    on a heterodyne config. Smoke tests never reached this branch, so the
    bug was silently latent.

    Replacing the silent crash with an explicit
    :class:`NotImplementedError` is the honest v0.1 contract: heterodyne
    callers should use ``optimization.nlsq.cmaes.enable: true`` (the
    fully-wired global-search path) or the default local trust-region fit
    until Phase 6 lands a heterodyne-shaped ``single_fit_func`` adapter for
    ``run_multistart_nlsq``.

    Raises
    ------
    NotImplementedError
        Always, with a pointer to the supported paths.
    """
    raise NotImplementedError(
        "Heterodyne multi-start is not wired in v0.1. Supported global-search "
        "path is CMA-ES (set `optimization.nlsq.cmaes.enable: true` in your "
        "config). Local trust-region fitting also remains available with "
        "all global flags disabled. Tracked for Phase 6: needs a heterodyne "
        "single_fit_func adapter for xpcsjax.optimization.nlsq.multistart."
        "run_multistart_nlsq."
    )


def _fit_local(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    config: NLSQConfig,
    weights: np.ndarray | jnp.ndarray | None,
    use_nlsq_library: bool,
) -> NLSQResult:
    """Run local (single-start) optimization with adapter/wrapper fallback.

    Tries adapter first; on failure falls back to wrapper with progressive
    recovery.
    """
    t_start = time.perf_counter()

    param_manager = model.param_manager
    varying_names = param_manager.varying_names
    n_varying = param_manager.n_varying

    logger.info("Fitting %d parameters: %s", n_varying, varying_names)

    # Memory-aware strategy selection. ``HAS_MEMORY`` is True iff both
    # ``select_nlsq_strategy`` and ``NLSQStrategy`` imported successfully —
    # narrow on the names themselves so Pyright sees them as bound.
    if select_nlsq_strategy is not None and NLSQStrategy is not None:
        n_data_est = np.asarray(c2_data).size
        decision = select_nlsq_strategy(n_data_est, n_varying)
        if decision.strategy in (NLSQStrategy.LARGE, NLSQStrategy.STREAMING):
            logger.warning(
                "Estimated peak memory (%.2f GB) exceeds threshold (%.2f GB). "
                "Fit may fail with OOM.",
                decision.peak_memory_gb,
                decision.threshold_gb,
            )

    # Get initial values and bounds
    initial_varying = param_manager.get_initial_values()
    lower_bounds, upper_bounds = param_manager.get_bounds()
    initial_varying = np.clip(initial_varying, lower_bounds, upper_bounds)

    # Convert data to JAX arrays
    c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_jax = (
        jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
    )

    if weights_jax is not None and weights_jax.shape != c2_jax.shape:
        raise ValueError(
            f"Weights shape {weights_jax.shape} does not match data shape {c2_jax.shape}"
        )

    # Capture constants
    fixed_values = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices = jnp.array(param_manager.varying_indices)
    n_data = c2_jax.size
    t, q, dt = model.t, model.q, model.dt

    # Per-angle scaling — fixed during local optimization (constant mode parity)
    contrast_val, offset_val = model.scaling.get_for_angle(0)

    # Build residual functions
    def jax_residual_fn(_x: jnp.ndarray, *varying_params: float) -> jnp.ndarray:
        """Pure JAX residual function for nlsq tracing."""
        varying_array = jnp.array(varying_params, dtype=jnp.float64)
        full_params = fixed_values.at[varying_indices].set(varying_array)
        return compute_residuals(
            full_params,
            t,
            q,
            dt,
            phi_angle,
            c2_jax,
            weights_jax,
            contrast_val,
            offset_val,
        )

    numpy_residual_fn = _make_numpy_residual_fn(
        model, c2_data, phi_angle, weights, contrast_val, offset_val
    )

    # ------------------------------------------------------------------
    # Adapter → wrapper fallback chain
    # ------------------------------------------------------------------
    adapter_error: Exception | None = None
    fallback_occurred = False
    result: NLSQResult | None = None

    if use_nlsq_library and NLSQAdapter is not None:  # HAS_ADAPTERS equivalent
        try:
            adapter = NLSQAdapter(parameter_names=varying_names)
            logger.debug("Attempting optimization with NLSQAdapter (JAX)")

            result = adapter.fit_jax(
                jax_residual_fn=jax_residual_fn,
                initial_params=initial_varying,
                bounds=(lower_bounds, upper_bounds),
                config=config,
                n_data=n_data,
            )

            if result.success:
                logger.info("NLSQAdapter optimization succeeded")
            else:
                raise RuntimeError(f"Adapter returned success=False: {result.message}")

        except (ValueError, RuntimeError, TypeError, ImportError, OSError) as e:
            adapter_error = e
            logger.warning("NLSQAdapter failed, falling back to wrapper: %s", e)
            fallback_occurred = True
            result = None

    # Wrapper fallback (or primary if use_nlsq_library=False)
    if result is None and NLSQWrapper is not None:  # HAS_WRAPPER equivalent
        try:
            wrapper = NLSQWrapper(parameter_names=varying_names)
            logger.debug("Attempting optimization with NLSQWrapper")

            result = wrapper.fit(
                residual_fn=numpy_residual_fn,
                initial_params=initial_varying,
                bounds=(lower_bounds, upper_bounds),
                config=config,
            )

            if fallback_occurred:
                logger.info("NLSQWrapper fallback succeeded")
            else:
                logger.info("NLSQWrapper optimization succeeded")

        except (ValueError, RuntimeError, TypeError, MemoryError) as wrapper_error:
            logger.error(
                "Both adapter and wrapper failed: adapter=%s, wrapper=%s",
                adapter_error,
                wrapper_error,
            )
            result = NLSQResult(
                parameters=initial_varying,
                parameter_names=varying_names,
                success=False,
                message=f"All optimizers failed. Adapter: {adapter_error}; "
                f"Wrapper: {wrapper_error}",
            )

    if result is None:
        raise ImportError(
            "No NLSQ optimization backend available. "
            "Ensure heterodyne.optimization.nlsq.adapter is importable."
        )

    # ------------------------------------------------------------------
    # Post-fit: compute fitted correlation, update model
    # ------------------------------------------------------------------
    if result.success:
        full_fitted = param_manager.expand_varying_to_full(result.parameters)
        fitted_c2 = compute_c2_heterodyne(
            jnp.asarray(full_fitted),
            t,
            q,
            dt,
            phi_angle,
            contrast_val,
            offset_val,
        )
        result.fitted_correlation = np.asarray(fitted_c2)
        model.set_params(full_fitted)

    # ------------------------------------------------------------------
    # Post-fit: correct reduced chi-squared
    #
    # The raw chi2 from adapter.fit_jax is SSR / (N² − n_params), where
    # SSR = Σ r² over the full N×N residual vector.  Two corrections:
    #
    #   1. DOF: the N diagonal residuals are forced to 0 by the
    #      non_diagonal mask in compute_residuals — they should be
    #      excluded from the degrees-of-freedom count.
    #      n_valid = N*(N−1) instead of N².
    #
    #   2. σ² normalization: without dividing by measurement noise,
    #      chi2 = MSE ≪ 1 for normalized C2 data (C2 ~ 1, residuals ~ 5%).
    #      We estimate σ²_noise from the far-lag plateau of the C2 matrix
    #      (|t2−t1| ≥ N/2), where correlations have fully decayed and
    #      the remaining variance is photon-counting noise.
    #
    # chi2_corrected = SSR / (σ²_noise × n_dof_valid)  →  ~1 for good fits
    # ------------------------------------------------------------------
    if result.final_cost is not None:
        n_matrix = c2_jax.shape[0]
        n_valid = c2_jax.size - n_matrix  # exclude N diagonal zeros
        n_dof_valid = max(n_valid - n_varying, 1)

        c2_np = np.asarray(c2_jax)
        row_idx = np.arange(n_matrix)
        lag_mat = np.abs(row_idx[:, None] - row_idx[None, :])
        far_mask = lag_mat >= n_matrix // 2  # diagonal (lag=0) not included
        far_vals = c2_np[far_mask]
        sigma2_noise = float(np.var(far_vals)) if far_vals.size > 1 else 0.0

        if sigma2_noise > 1e-12:
            ssr = 2.0 * result.final_cost
            chi2_corrected = ssr / (sigma2_noise * n_dof_valid)
            logger.debug(
                "chi2 correction: σ²_noise=%.4e  n_valid=%d  SSR=%.4e  "
                "raw_chi2=%.4g → chi2_corrected=%.4f",
                sigma2_noise,
                n_valid,
                ssr,
                result.reduced_chi_squared or float("nan"),
                chi2_corrected,
            )
            result.reduced_chi_squared = chi2_corrected
        else:
            logger.warning(
                "chi2 noise estimate near-zero (σ²=%.2e); "
                "reporting uncorrected MSE chi2",
                sigma2_noise,
            )

    result.metadata["fallback_occurred"] = fallback_occurred
    if adapter_error is not None:
        result.metadata["adapter_error"] = str(adapter_error)
    result.metadata["optimizer"] = "local"
    result.metadata["wall_time_total"] = time.perf_counter() - t_start

    _log_result(result)
    return result


def _make_numpy_residual_fn(
    model: HeterodyneModel,
    c2_data: np.ndarray | jnp.ndarray,
    phi_angle: float,
    weights: np.ndarray | jnp.ndarray | None,
    contrast: float = 1.0,
    offset: float = 1.0,
) -> Any:
    """Create a numpy residual function closed over model/data.

    Returns a callable ``(varying_params: np.ndarray) -> np.ndarray``.

    Hot-path optimisation: ``fixed_values`` and ``varying_indices`` are
    pre-captured as JAX device arrays at construction time so each call
    only performs a single ``jnp.asarray`` (for the incoming numpy vector)
    and one ``jnp.ndarray.at[].set()`` scatter instead of a Python loop
    plus a full host copy.
    """
    param_manager = model.param_manager
    c2_jax = jnp.asarray(c2_data, dtype=jnp.float64)
    weights_jax = (
        jnp.asarray(weights, dtype=jnp.float64) if weights is not None else None
    )
    t, q, dt = model.t, model.q, model.dt

    # Pre-capture as JAX device arrays — allocated once, reused every call.
    # NOTE: fixed_values snapshot is taken at construction time. Do not mutate
    # param_manager between construction and optimizer completion.
    fixed_values = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices = jnp.array(param_manager.varying_indices, dtype=jnp.int32)

    def residual_fn(varying_params: np.ndarray) -> np.ndarray:
        varying_jax = jnp.asarray(varying_params, dtype=jnp.float64)
        full_params = fixed_values.at[varying_indices].set(varying_jax)
        residuals = compute_residuals(
            full_params,
            t,
            q,
            dt,
            phi_angle,
            c2_jax,
            weights_jax,
            contrast,
            offset,
        )
        return np.asarray(residuals)

    return residual_fn


def _log_result(result: NLSQResult) -> None:
    """Log optimization results summary."""
    logger.info("=" * 60)
    logger.info("NLSQ OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    status = "SUCCESS" if result.success else "FAILED"
    logger.info("Status: %s", status)
    logger.info("Message: %s", result.message)

    if result.final_cost is not None:
        logger.info("Final cost: %.6e", result.final_cost)
    if result.reduced_chi_squared is not None:
        logger.info("Reduced χ²: %.4f", result.reduced_chi_squared)
    if result.wall_time_seconds is not None:
        logger.info("Wall time: %.2f s", result.wall_time_seconds)

    if result.success:
        for name, val in zip(result.parameter_names, result.parameters, strict=True):
            unc_val = result.get_uncertainty(name)
            if unc_val is not None:
                logger.info("  %s: %.6g ± %.3g", name, val, unc_val)
            else:
                logger.info("  %s: %.6g", name, val)

    logger.info("=" * 60)
