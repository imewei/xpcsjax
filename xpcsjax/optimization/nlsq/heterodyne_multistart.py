"""Heterodyne joint multistart (Phase 1).

Wires the ``optimization.nlsq.multi_start`` config section into the heterodyne
(``two_component``) path. Mirrors the xpcsjax homodyne pattern
(``core.fit_nlsq_multistart``): each Latin-Hypercube start re-runs the whole
joint multi-phi fit, then a final authoritative fit from the winning start
produces the heterodyne ``OptimizationResult`` with full ``nlsq_diagnostics``.

Heterodyne multistart runs **sequentially**: the single-fit worker closes over a
JAX ``HeterodyneModel`` that is not process-picklable, so parallel workers
(``n_workers > 1``) are deferred to a follow-up.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from xpcsjax.optimization.nlsq.heterodyne_core import fit_nlsq_multi_phi
from xpcsjax.optimization.nlsq.multistart import (
    MultiStartConfig,
    SingleStartResult,
    run_multistart_nlsq,
)
from xpcsjax.utils.logging import get_logger

logger = get_logger(__name__)


def build_multistart_config(ms_dict: dict[str, Any]) -> MultiStartConfig:
    """Build a ``MultiStartConfig`` from the nested ``multi_start`` config dict.

    ``n_workers`` is clamped to 1 (sequential): the heterodyne single-fit worker
    closes over a JAX model that cannot be pickled for ``ProcessPoolExecutor``.
    """
    requested_workers = int(ms_dict.get("n_workers", 0) or 0)
    if requested_workers not in (0, 1):
        logger.warning(
            "Heterodyne multistart runs sequentially; ignoring n_workers=%d "
            "(parallel heterodyne multistart is a follow-up).",
            requested_workers,
        )
    return MultiStartConfig(
        enable=bool(ms_dict.get("enable", False)),
        n_starts=int(ms_dict.get("n_starts", 10)),
        seed=int(ms_dict.get("seed", 42)),
        sampling_strategy=str(ms_dict.get("sampling_strategy", "latin_hypercube")),
        n_workers=1,
        use_screening=bool(ms_dict.get("use_screening", True)),
        screen_keep_fraction=float(ms_dict.get("screen_keep_fraction", 0.5)),
        refine_top_k=int(ms_dict.get("refine_top_k", 3)),
        refinement_ftol=float(ms_dict.get("refinement_ftol", 1e-12)),
        degeneracy_threshold=float(ms_dict.get("degeneracy_threshold", 0.1)),
    )


def fit_nlsq_multistart_heterodyne(model, c2, phi, nlsq_cfg, weights, ms_cfg):
    """Run joint multi-phi multistart, then re-fit once from the best start.

    Each Latin-Hypercube start sets the model's varying physics initial values
    and re-runs ``fit_nlsq_multi_phi``. The winning start is re-fit once to
    produce the authoritative heterodyne ``OptimizationResult``.
    """
    pm = model.param_manager
    varying_names = list(pm.varying_names)
    lower, upper = pm.get_bounds()
    bounds = np.column_stack([np.asarray(lower), np.asarray(upper)])

    c2 = np.asarray(c2)
    phi = np.asarray(phi)
    data = {"c2_exp": c2, "phi_angles_list": phi}

    def _single_fit(_data, start_params):
        import time

        t0 = time.perf_counter()
        start_params = np.asarray(start_params, dtype=np.float64)
        try:
            pm.update_values(
                {name: float(v) for name, v in zip(varying_names, start_params, strict=True)}
            )
            res = fit_nlsq_multi_phi(model, c2, phi, nlsq_cfg, weights)
            return SingleStartResult(
                start_idx=0,
                initial_params=start_params,
                final_params=np.asarray(res.parameters, dtype=np.float64),
                chi_squared=float(res.chi_squared),
                reduced_chi_squared=float(getattr(res, "reduced_chi_squared", res.chi_squared)),
                success=bool(getattr(res, "success", True)),
                message=str(getattr(res, "message", "")),
                wall_time=time.perf_counter() - t0,
            )
        except (ValueError, RuntimeError, TypeError, FloatingPointError) as exc:
            return SingleStartResult(
                start_idx=0,
                initial_params=start_params,
                final_params=start_params,
                chi_squared=float("inf"),
                success=False,
                message=str(exc),
                wall_time=time.perf_counter() - t0,
            )

    def _cost_func(params):
        params = np.asarray(params, dtype=np.float64)
        if np.any(params <= lower) or np.any(params >= upper):
            return 1e20
        center = (lower + upper) / 2.0
        scale = upper - lower
        return float(np.sum(((params - center) / scale) ** 2))

    custom_starts = [pm.get_initial_values().astype(float).tolist()]

    logger.info("Heterodyne joint multistart: %d starts (sequential)", ms_cfg.n_starts)
    ms_result = run_multistart_nlsq(
        data=data,
        bounds=bounds,
        config=ms_cfg,
        single_fit_func=_single_fit,
        cost_func=_cost_func if ms_cfg.use_screening else None,
        custom_starts=custom_starts,
    )

    best_start = np.asarray(ms_result.best.initial_params, dtype=np.float64)
    pm.update_values(
        {name: float(v) for name, v in zip(varying_names, best_start, strict=True)}
    )
    final = fit_nlsq_multi_phi(model, c2, phi, nlsq_cfg, weights)

    # Attach multistart provenance. nlsq_diagnostics defaults to None on
    # OptimizationResult, so initialise it rather than silently dropping the
    # metadata when the joint result carried no diagnostics dict.
    diagnostics = getattr(final, "nlsq_diagnostics", None)
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        final.nlsq_diagnostics = diagnostics
    diagnostics["multistart"] = {
        "n_starts": ms_cfg.n_starts,
        "best_start_idx": ms_result.best.start_idx,
        "n_unique_basins": ms_result.n_unique_basins,
        "degeneracy_detected": ms_result.degeneracy_detected,
    }
    return final
