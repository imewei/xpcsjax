"""Heterodyne ``constant`` mode: quantile-fixed per-angle scaling.

Implements homodyne's ``constant`` semantics for heterodyne:

1. Pre-estimate β(φ_k), ō(φ_k) via the diagonal-quantile estimator (dual-region
   helper from ``heterodyne_scaling_utils``).
2. Hold both fixed during NLSQ — they enter the residual via closure capture,
   NOT through the optimizer parameter vector.
3. Optimize only the ``n_physics_varying`` physical model parameters.

Distinct from the ``auto``-averaged path
(:func:`xpcsjax.optimization.nlsq.heterodyne_core._fit_joint_averaged_multi_phi`),
which optimizes a single averaged ``(contrast, offset)`` pair JOINTLY with the
physics parameters. In ``constant`` mode the scaling is frozen pre-fit and the
optimizer dimensionality is exactly ``n_physics_varying``.

The function returned by this module returns an
:class:`xpcsjax.optimization.nlsq.results.OptimizationResult` (the homodyne-side
result type — *not* the heterodyne ``NLSQResult`` produced elsewhere), per the
Phase 6 Sub-PR B contract.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from xpcsjax.config.parameter_registry import SCALING_PARAMS
from xpcsjax.core.heterodyne_jax_backend import compute_multi_angle_residuals
from xpcsjax.core.heterodyne_scaling_utils import (
    estimate_per_angle_scaling_from_quantile,
)
from xpcsjax.optimization.nlsq.anti_degeneracy_diagnostics import (
    assemble_anti_degeneracy_diagnostics,
)
from xpcsjax.optimization.nlsq.heterodyne_config import NLSQConfig
from xpcsjax.optimization.nlsq.results import (
    ConvergenceStatus,
    OptimizationResult,
    QualityFlag,
)
from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    # The runtime object the fitter receives is the stateful dataclass in
    # ``heterodyne_model_stateful`` (which exposes ``.t``, ``.q``, ``.dt``,
    # ``.scaling``, ``.param_manager``). The bare wrapper in
    # ``heterodyne_model`` is a PhysicsModelBase adapter without those fields.
    from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Optional backend imports — gated for graceful degradation
# ---------------------------------------------------------------------------
# Same pattern as ``heterodyne_core.py``: bind to ``None`` in the ImportError
# branch so Pyright/mypy can narrow on ``is not None`` at call sites.
try:
    from xpcsjax.optimization.nlsq.heterodyne_adapter import (
        NLSQAdapter,
        NLSQWrapper,
    )
except ImportError:  # pragma: no cover — backend always present in v0.1
    NLSQAdapter = None  # type: ignore[assignment,misc]
    NLSQWrapper = None  # type: ignore[assignment,misc]


def _fit_joint_constant_multi_phi(
    model: HeterodyneModel,
    c2_data: np.ndarray,
    phi_angles: np.ndarray,
    config: NLSQConfig,
    weights: np.ndarray | None,
) -> OptimizationResult:
    """Joint multi-angle fit with quantile-fixed per-angle scaling.

    The optimizer parameter vector contains only the ``n_physics_varying``
    physics parameters. Per-angle contrast and offset are estimated pre-fit
    via :func:`estimate_per_angle_scaling_from_quantile` and held fixed by
    capturing them inside the residual closure.

    Parameters
    ----------
    model
        :class:`HeterodyneModel` providing ``t``, ``q``, ``dt`` time/scattering
        constants and a ``param_manager`` with ``n_varying``, ``varying_names``,
        ``varying_indices``, ``get_initial_values``, ``get_bounds``,
        ``get_full_values``, and ``expand_varying_to_full``.
    c2_data
        Correlation data, shape ``(n_phi, N, N)``.
    phi_angles
        Detector angles in degrees, shape ``(n_phi,)``.
    config
        :class:`NLSQConfig`; only solver-related fields (``method``, tolerances,
        ``max_nfev``, ``loss``, ``use_nlsq_library``) are consumed here.
        ``per_angle_mode`` must be ``"constant"`` for this routine to be called.
    weights
        Optional weight stack matching ``c2_data`` shape, or ``None`` for
        unit weights.

    Returns
    -------
    OptimizationResult
        ``parameters`` has shape ``(n_physics_varying,)``.
        ``nlsq_diagnostics`` contains:

        * ``scaling_source``: ``"quantile_fixed"``
        * ``contrast_per_angle_fixed``: ``np.ndarray`` of shape ``(n_phi,)``
        * ``offset_per_angle_fixed``: ``np.ndarray`` of shape ``(n_phi,)``
        * ``chi2_per_angle``: ``np.ndarray`` of shape ``(n_phi,)``
        * ``per_angle_mode``: ``"constant"``
        * ``fourier_basis_dim``: ``None``
        * ``shear_weighting``: ``"not_applicable_heterodyne"``
        * ``parameter_names``: ``list[str]`` of varying physics names
    """
    if NLSQAdapter is None and NLSQWrapper is None:  # pragma: no cover
        raise ImportError(
            "No NLSQ backend available for _fit_joint_constant_multi_phi. "
            "Ensure xpcsjax.optimization.nlsq.heterodyne_adapter is importable."
        )

    t_start = time.perf_counter()
    phi_angles_np = np.asarray(phi_angles, dtype=np.float64)
    n_phi = int(phi_angles_np.size)

    param_manager = model.param_manager
    varying_names = list(param_manager.varying_names)
    n_physics = int(param_manager.n_varying)

    physics_initial = np.asarray(param_manager.get_initial_values(), dtype=np.float64)
    physics_lower, physics_upper = param_manager.get_bounds()
    physics_lower = np.asarray(physics_lower, dtype=np.float64)
    physics_upper = np.asarray(physics_upper, dtype=np.float64)
    physics_initial = np.clip(physics_initial, physics_lower, physics_upper)

    logger.info("=" * 60)
    logger.info(
        "HETERODYNE CONSTANT MODE: quantile-fixed scaling, %d physics params, %d angles",
        n_physics,
        n_phi,
    )
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Flatten (c2, t1, t2, phi_indices) for the quantile estimator.
    # ------------------------------------------------------------------
    c2_flat, t1_flat, t2_flat, phi_idx_flat = _flatten_inputs(model, c2_data, n_phi)

    # ------------------------------------------------------------------
    # 2. Pre-estimate per-angle (contrast, offset) and clamp to registry bounds.
    # ------------------------------------------------------------------
    contrast_fixed, offset_fixed = estimate_per_angle_scaling_from_quantile(
        c2_data=c2_flat,
        t1=t1_flat,
        t2=t2_flat,
        phi_indices=phi_idx_flat,
        n_phi=n_phi,
        quantile=0.95,
    )
    contrast_info = SCALING_PARAMS["contrast"]
    offset_info = SCALING_PARAMS["offset"]
    contrast_fixed = np.clip(
        contrast_fixed, contrast_info.min_bound, contrast_info.max_bound
    ).astype(np.float64)
    offset_fixed = np.clip(offset_fixed, offset_info.min_bound, offset_info.max_bound).astype(
        np.float64
    )
    logger.info(
        "Frozen per-angle scaling: contrast=%s, offset=%s",
        np.array2string(contrast_fixed, precision=4),
        np.array2string(offset_fixed, precision=4),
    )

    # ------------------------------------------------------------------
    # 3. Build the JAX-side closure constants.
    # ------------------------------------------------------------------
    t = model.t
    q = model.q
    dt = model.dt

    c2_data_batch = jnp.asarray(c2_data, dtype=jnp.float64)
    if weights is None:
        weights_batch = jnp.ones_like(c2_data_batch)
    else:
        weights_arr = jnp.asarray(weights, dtype=jnp.float64)
        if weights_arr.ndim == 2:
            weights_arr = jnp.broadcast_to(weights_arr, c2_data_batch.shape)
        weights_batch = weights_arr
    phi_angles_jax = jnp.asarray(phi_angles_np, dtype=jnp.float64)
    contrast_jax = jnp.asarray(contrast_fixed, dtype=jnp.float64)
    offset_jax = jnp.asarray(offset_fixed, dtype=jnp.float64)
    fixed_values_jax = jnp.asarray(param_manager.get_full_values(), dtype=jnp.float64)
    varying_indices_jax = jnp.array(list(param_manager.varying_indices), dtype=jnp.int32)

    # Physics-only residual; scaling enters by closure (frozen).
    #
    # IMPORTANT (tracer-safety, mirrors _fit_joint_averaged_multi_phi):
    # NLSQ's masked_residual_func JIT-traces this closure. Returning a JAX
    # array is fine — calling ``np.asarray`` on a traced result raises
    # TracerArrayConversionError. The kernel returns ``jnp.ndarray`` and
    # NLSQ handles the cast at its boundary.
    def joint_residual_fn(x: np.ndarray) -> Any:  # type: ignore[return-value]
        physics_varying = jnp.asarray(x, dtype=jnp.float64)
        full_jax = fixed_values_jax.at[varying_indices_jax].set(physics_varying)
        return compute_multi_angle_residuals(
            full_jax,
            t,
            q,
            dt,
            phi_angles_jax,
            c2_data_batch,
            weights_batch,
            contrast_jax,
            offset_jax,
        )

    # ------------------------------------------------------------------
    # 4. Build a solver config compatible with the NLSQ adapter.
    # max_nfev is multiplied by n_phi here because the joint solve packs
    # all angles into a single residual vector; the per-angle budget
    # documented on NLSQConfig.max_nfev is preserved by scaling the
    # combined cap. See NLSQConfig.max_nfev docstring for the contract.
    # ------------------------------------------------------------------
    joint_config = NLSQConfig(
        method=config.method if config.method != "lm" else "trf",
        ftol=config.ftol,
        xtol=config.xtol,
        gtol=config.gtol,
        max_nfev=(config.max_nfev * n_phi if config.max_nfev is not None else None),
        loss=config.loss,
        use_nlsq_library=config.use_nlsq_library,
        n_params=n_physics,
    )

    # ------------------------------------------------------------------
    # 5. Dispatch to NLSQ (adapter primary, wrapper fallback).
    # ------------------------------------------------------------------
    nlsq_result = None
    if NLSQAdapter is not None:
        try:
            adapter = NLSQAdapter(parameter_names=varying_names)
            nlsq_result = adapter.fit(
                residual_fn=joint_residual_fn,
                initial_params=physics_initial,
                bounds=(physics_lower, physics_upper),
                config=joint_config,
            )
            if not nlsq_result.success:
                raise RuntimeError(
                    f"Constant-mode adapter returned success=False: {nlsq_result.message}"
                )
        except (ValueError, RuntimeError, TypeError) as adapter_exc:
            logger.warning(
                "Constant-mode NLSQAdapter failed, falling back to NLSQWrapper: %s",
                adapter_exc,
            )
            nlsq_result = None

    if nlsq_result is None and NLSQWrapper is not None:
        wrapper = NLSQWrapper(parameter_names=varying_names)
        nlsq_result = wrapper.fit(
            residual_fn=joint_residual_fn,
            initial_params=physics_initial,
            bounds=(physics_lower, physics_upper),
            config=joint_config,
        )

    if nlsq_result is None:  # pragma: no cover — guarded at function entry
        raise ImportError("No NLSQ backend produced a result for _fit_joint_constant_multi_phi.")

    # ------------------------------------------------------------------
    # 6. Update model state with fitted physics + frozen scaling.
    # ------------------------------------------------------------------
    fitted_physics = np.asarray(nlsq_result.parameters, dtype=np.float64)
    full_fitted = param_manager.expand_varying_to_full(fitted_physics)
    model.set_params(full_fitted)
    if hasattr(model, "scaling") and len(model.scaling.contrast) == n_phi:
        model.scaling.contrast[:] = contrast_fixed
        model.scaling.offset[:] = offset_fixed

    # ------------------------------------------------------------------
    # 7. Decompose per-angle chi2 from the final residual.
    # ------------------------------------------------------------------
    final_residual = np.asarray(joint_residual_fn(fitted_physics))
    n_time = c2_data.shape[1]
    n_per_angle = (n_time - 1) * (n_time - 2)  # off-diag, t=0 boundary excluded — matches kernel
    chi2_per_angle = _decompose_chi2_per_angle(
        final_residual=final_residual,
        n_phi=n_phi,
        n_per_angle=n_per_angle,
    )

    # ------------------------------------------------------------------
    # 8. Translate heterodyne NLSQResult → homodyne-side OptimizationResult.
    # ------------------------------------------------------------------
    uncertainties = (
        np.asarray(nlsq_result.uncertainties, dtype=np.float64)
        if nlsq_result.uncertainties is not None
        else np.full(n_physics, np.nan, dtype=np.float64)
    )
    covariance = (
        np.asarray(nlsq_result.covariance, dtype=np.float64)
        if nlsq_result.covariance is not None
        else np.full((n_physics, n_physics), np.nan, dtype=np.float64)
    )

    # The OptimizationResult ``chi_squared`` field carries SSR (sum of
    # squared *raw* residuals, homodyne convention). We compute it directly
    # from ``final_residual`` rather than from ``nlsq_result.final_cost``
    # because the adapter's final_cost is the *robust-loss* cost (e.g.
    # ``soft_l1``), not 0.5*SSR — so ``2 * final_cost`` would diverge from
    # ``chi2_per_angle.sum()`` whenever ``config.loss != "linear"``. By
    # using the raw residual sum here, SSR conservation
    # (``chi2_per_angle.sum() == chi_squared``) holds for every loss choice.
    ssr = float(np.sum(final_residual**2))
    # Noise-normalised reduced chi^2 (targets ~1.0). ``nlsq_result.reduced_chi_squared``
    # is SSR/N² (MSE ≪ 1 on normalised C2 data); apply the same far-lag
    # photon-noise correction the other heterodyne paths use. ``chi_squared``
    # (= ssr) and ``chi2_per_angle`` are untouched, so SSR conservation
    # (``chi2_per_angle.sum() == chi_squared``) still holds.
    from xpcsjax.optimization.nlsq.heterodyne_data_prep import (
        noise_normalized_reduced_chi2,
    )

    reduced_chi2 = noise_normalized_reduced_chi2(
        ssr=ssr,
        c2_data=c2_data,
        n_data_valid=int(final_residual.size),
        n_params=n_physics,
    )

    wall_time = time.perf_counter() - t_start
    convergence_status: ConvergenceStatus = "converged" if nlsq_result.success else "failed"
    quality_flag: QualityFlag = "good" if nlsq_result.success else "marginal"

    # L2 hierarchical: no-op for constant mode. The whole solve IS the
    # "stage 1" of the two-stage pattern (physics-only with quantile-fixed
    # scaling); there is no stage 2 because scaling is permanently frozen.
    # Record the flag-handling so downstream consumers can confirm the
    # config-side request was observed. Only the DETAIL keys are conditional;
    # the activation flags below are emitted on every path by the assembler.
    hierarchical_extras: dict[str, Any] = {}
    if config.enable_hierarchical:
        hierarchical_extras = {
            "hierarchical_stages": 1,
            "hierarchical_scope": "constant_mode_no_stage2",
        }

    diagnostics: dict[str, Any] = {
        "scaling_source": "quantile_fixed",
        "contrast_per_angle_fixed": contrast_fixed,
        "offset_per_angle_fixed": offset_fixed,
        "chi2_per_angle": chi2_per_angle,
        "per_angle_mode": "constant",
        "fourier_basis_dim": None,
        "parameter_names": varying_names,
        "convergence_reason": nlsq_result.convergence_reason,
        "n_function_evals": int(nlsq_result.n_function_evals or 0),
        "n_iterations": int(nlsq_result.n_iterations or 0),
        "wall_time_seconds": wall_time,
        "message": str(nlsq_result.message),
    }
    # L2/L3/L4/L5 activation block via the shared assembler. Constant mode never
    # runs L2 stage-2 (hierarchical_active=False), L3 (regularization_active=
    # False), or the L4 monitor (gradient_monitor omitted); the
    # ``"not_applicable_heterodyne"`` marker makes the homodyne L5 N/A explicit.
    diagnostics.update(
        assemble_anti_degeneracy_diagnostics(
            hierarchical_active=False,
            regularization_active=False,
            shear_weighting="not_applicable_heterodyne",
            gradient_monitor=None,
            **hierarchical_extras,
        )
    )

    return OptimizationResult(
        parameters=fitted_physics,
        uncertainties=uncertainties,
        covariance=covariance,
        chi_squared=ssr,
        reduced_chi_squared=reduced_chi2,
        convergence_status=convergence_status,
        iterations=int(nlsq_result.n_iterations or 0),
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


def _flatten_inputs(
    model: HeterodyneModel, c2_data: np.ndarray, n_phi: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a ``(n_phi, N, N)`` c2 stack to flattened ``(c2, t1, t2, phi_idx)``
    arrays expected by :func:`estimate_per_angle_scaling_from_quantile`.

    Mirrors the flattening block at the top of
    :func:`xpcsjax.optimization.nlsq.heterodyne_core._fit_joint_averaged_multi_phi`.
    The full ``(N, N)`` grid is flattened (no diagonal exclusion) — the
    quantile estimator's dual-region split (small-lag ceiling vs large-lag
    floor) needs both diagonal and off-diagonal samples to recover offset
    correctly.
    """
    t = np.asarray(model.t, dtype=np.float64)
    t1_mesh, t2_mesh = np.meshgrid(t, t, indexing="ij")
    n_time_points = int(t1_mesh.size)
    c2_arr = np.asarray(c2_data, dtype=np.float64)

    c2_chunks: list[np.ndarray] = []
    t1_chunks: list[np.ndarray] = []
    t2_chunks: list[np.ndarray] = []
    phi_idx_chunks: list[np.ndarray] = []
    for i in range(n_phi):
        c2_chunks.append(c2_arr[i].reshape(-1))
        t1_chunks.append(t1_mesh.reshape(-1))
        t2_chunks.append(t2_mesh.reshape(-1))
        phi_idx_chunks.append(np.full(n_time_points, i, dtype=np.int32))

    return (
        np.concatenate(c2_chunks),
        np.concatenate(t1_chunks),
        np.concatenate(t2_chunks),
        np.concatenate(phi_idx_chunks),
    )


def _decompose_chi2_per_angle(
    final_residual: np.ndarray, n_phi: int, n_per_angle: int
) -> np.ndarray:
    """Sum squared residual per angle from the flattened multi-angle vector.

    :func:`compute_multi_angle_residuals` returns
    ``residuals_batch.ravel()`` where ``residuals_batch`` has shape
    ``(n_phi, n_per_angle)`` — i.e., the flat layout is angle-major, so a
    straight reshape recovers the per-angle slices.
    """
    expected = n_phi * n_per_angle
    if final_residual.size != expected:  # pragma: no cover — guarded by callers
        raise ValueError(
            f"final_residual size {final_residual.size} != n_phi*n_per_angle "
            f"= {expected} (n_phi={n_phi}, n_per_angle={n_per_angle})"
        )
    r2 = final_residual.reshape(n_phi, n_per_angle) ** 2
    return np.asarray(r2.sum(axis=1), dtype=np.float64)
