"""Shared out-of-core (>=75% RAM) fit route for :class:`NLSQWrapper.fit`.

``NLSQWrapper.fit`` can decide to route to the out-of-core accumulator from
two places: the initial memory-based strategy decision (before any
stratification), and a strategy *recheck* after angle-stratified chunking
resolves the effective per-angle parameter count. Both call sites used to
carry their own ~90-line copy of "run the accumulator, un-placeholder pcov,
re-zero fixed-parameter uncertainties, compute DOF, derive quality_flag,
build ``OptimizationResult``" -- and had drifted: the initial copy resolved
the requested ``per_angle_mode`` with the plain (non-pinned) resolver, so a
``static_anisotropic`` fit with an explicit ``per_angle_mode: "constant"``
token reported DOF=n_physical there but DOF=n_physical+2*n_phi from the
recheck copy for the same dense parameter vector (static modes have no flow
direction, so every other path pins them to the dense ``individual`` layout
regardless of the requested token -- see
:func:`~xpcsjax.optimization.nlsq.per_angle_mode.resolve_per_angle_mode_static_pinned`).

This module is the single owner of that route now; both call sites in
``wrapper.py`` funnel through :func:`run_out_of_core_route`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.config.parameter_registry import AnalysisMode
from xpcsjax.optimization.nlsq.per_angle_mode import (
    effective_constrained_dof,
    resolve_per_angle_mode_static_pinned,
)
from xpcsjax.optimization.nlsq.results import (
    OptimizationResult,
    quality_flag_from_reduced_chi2,
)
from xpcsjax.optimization.nlsq.strategies.out_of_core import (
    fit_with_out_of_core_accumulation,
)

if TYPE_CHECKING:
    from xpcsjax.optimization.nlsq.parameter_utils import ResolvedPhysicalParameters

__all__ = ["run_out_of_core_route"]


def run_out_of_core_route(
    *,
    stratified_data: Any,
    data: Any,
    per_angle_scaling: bool,
    physical_param_names: list[str],
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    logger: Any,
    config: Any,
    analysis_mode: AnalysisMode,
    n_points: int,
    n_phi: int,
    resolved_physical: ResolvedPhysicalParameters | None,
    start_time: float,
    strategy_reason: str,
    recovery_tag: str,
    fast_mode_override: bool = False,
    fit_ooc: Callable[..., tuple[np.ndarray, np.ndarray, dict]] = fit_with_out_of_core_accumulation,
) -> OptimizationResult:
    """Run the out-of-core accumulator and assemble its ``OptimizationResult``.

    Parameters mirror what each of ``NLSQWrapper.fit``'s two OOC call sites
    already had in hand -- see the module docstring for why this now lives in
    one place. ``n_phi`` must be the unique-angle count computed by the
    caller (``data.phi`` for the initial trigger, ``stratified_data.phi_flat``
    for the recheck trigger); DOF is always resolved through the SAME static
    pin regardless of which trigger fired.
    """
    import time

    # Deferred import: wrapper.py imports this module at load time, and these
    # three helpers still live in wrapper.py (shared by every other fit tier
    # too) -- importing at module level here would be circular.
    from xpcsjax.optimization.nlsq.wrapper import (
        _info_cov_placeholder,
        _laminar_anti_degeneracy_block,
        _uncertainties_from_pcov,
    )

    ad_cfg: dict[str, Any] = {}
    if config is not None and hasattr(config, "config"):
        nlsq_cfg = (config.config.get("optimization") or {}).get("nlsq") or {}
        ad_cfg = nlsq_cfg.get("anti_degeneracy") or {}

    use_fast_mode = fast_mode_override or (
        bool((config.config.get("optimization") or {}).get("fast_chi2_mode", False))
        if config is not None and hasattr(config, "config")
        else False
    )

    popt, pcov, info = fit_ooc(
        stratified_data=stratified_data,
        data=data,
        per_angle_scaling=per_angle_scaling,
        physical_param_names=physical_param_names,
        initial_params=initial_params,
        bounds=bounds,
        log=logger,
        config=config,
        fast_chi2_mode=use_fast_mode,
        anti_degeneracy_config=ad_cfg,
        resolved_physical=resolved_physical,
    )

    execution_time = time.time() - start_time
    is_placeholder = _info_cov_placeholder(info)
    pcov, uncertainties = _uncertainties_from_pcov(pcov, len(popt), is_placeholder=is_placeholder)

    # A fixed physical parameter's true covariance diagonal is exactly 0
    # (out_of_core.py's masked pcov build never writes a nonzero value
    # there); keep the reported uncertainty exactly 0.0 there even on a
    # placeholder (NaN) covariance.
    if resolved_physical is not None and not resolved_physical.free_mask.all():
        n_physical = len(resolved_physical.physical_names)
        uncertainties = np.array(uncertainties, dtype=float)
        for i, free in enumerate(resolved_physical.free_mask):
            if not free:
                uncertainties[-n_physical + i] = 0.0

    # Effective DOF for reduced chi-squared: in averaged/constant mode the
    # optimizer works on a compressed param vector but the true model DOF is
    # the EXPANDED per-angle count. Resolved through the static pin so a
    # static fit's DOF always reflects the dense individual vector it
    # actually fits, regardless of the requested token -- the single owner
    # both OOC triggers now share (this is the fix for A4).
    n_params_effective: int | None = None
    if per_angle_scaling and ad_cfg:
        n_physical = len(physical_param_names)
        resolved_mode = resolve_per_angle_mode_static_pinned(
            ad_cfg.get("per_angle_mode", "auto"),
            n_phi,
            ad_cfg.get("constant_scaling_threshold", 3),
            is_laminar_flow=(analysis_mode == AnalysisMode.LAMINAR_FLOW),
        )
        n_params_effective = effective_constrained_dof(
            resolved_mode, n_phi=n_phi, n_physical=n_physical
        )
    dof = n_params_effective if n_params_effective is not None else len(popt)
    reduced_chi2 = info.get("chi_squared", 0.0) / max(1, n_points - dof)
    quality_flag = quality_flag_from_reduced_chi2(reduced_chi2)

    return OptimizationResult(
        parameters=popt,
        uncertainties=uncertainties,
        covariance=pcov,
        chi_squared=info.get("chi_squared", 0.0),
        reduced_chi_squared=reduced_chi2,
        convergence_status=info.get("convergence_status", "unknown"),
        iterations=info.get("iterations", 0),
        execution_time=execution_time,
        device_info={
            "device": "cpu_accumulated",
            "strategy": "out_of_core",
            "fast_mode": use_fast_mode,
            "decision": strategy_reason,
        },
        recovery_actions=[recovery_tag],
        quality_flag=quality_flag,
        # Anti-degeneracy is unsupported on the out-of-core path; emit the
        # symmetric inactive markers so this result mirrors the contract.
        nlsq_diagnostics={
            **_laminar_anti_degeneracy_block(None),
            "covariance_is_placeholder": bool(is_placeholder),
        },
    )
