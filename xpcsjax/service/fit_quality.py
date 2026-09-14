"""Mode-agnostic, scale-free fit-quality metric (NRMSE).

``reduced_chi_squared`` is not comparable across analysis modes: the homodyne
paths (``static_*`` / ``laminar_flow``) report ``sum((r / sigma)**2) / dof`` over
the full matrix, where ``sigma`` is the data uncertainty when provided and a
constant ``0.01`` placeholder otherwise (absolute scale, not relative); the
heterodyne path (``two_component``) reports the unweighted SSR over the
off-diagonal/t>0 mask divided by a far-lag noise-variance estimate. The same
``good`` / ``marginal`` / ``poor`` bands are applied to both, so the labels do
not mean the same thing.

This module computes ONE metric identically for every mode, from the final
parameters and the data only::

    nrmse = sqrt(SSR / n_valid) / std(c2_data over the same mask)

* the fitted surface is evaluated on the FIT-TIME model: homodyne modes reuse
  the plots' per-angle evaluator (``xpcsjax.viz.nlsq_plots._evaluate_c2_per_angle``,
  same kernel and grid as the fit); ``two_component`` uses the stateful
  ``HeterodyneModel`` the heterodyne fit ran (see the note in
  :func:`compute_fit_quality` — the viz evaluator's time origin differs by one
  ``dt`` there). No parameter layout is re-derived here (the viz unpackers are
  reused);
* the mask excludes the ``t=0`` row/column and the main diagonal (the
  zero-lag self-correlation spike) for every mode, i.e. the heterodyne
  ``(n_t - 1) * (n_t - 2)`` support;
* the denominator is a DATA statistic (not a fitted contrast), so the optimizer
  cannot improve the metric by collapsing a parameter.

``nrmse`` reads as "RMS misfit as a fraction of the data's own spread": 0.05 =
residuals are 5 % of the c2 dynamic range. It is advisory and does not drive
``quality_flag`` (that stays on ``reduced_chi_squared``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from xpcsjax.config.manager import ConfigManager
    from xpcsjax.optimization.nlsq.results import OptimizationResult

logger = get_logger(__name__)

MASK_DESCRIPTION = "exclude_t0_row_col_and_diagonal"


def sigma_source(result: OptimizationResult, mode: str) -> str:
    """Name the ``sigma`` that ``reduced_chi_squared`` was normalized with.

    * ``"far_lag_estimate"`` — ``two_component``: SSR / (var(c2 at lag >= n_t/2) * dof),
      falling back to plain MSE when that estimate is degenerate.
    * ``"default_constant_0.01"`` — homodyne fit with no data ``sigma``
      (``result.sigma_is_default``): chi2 is on an arbitrary absolute scale.
    * ``"data"`` — homodyne fit weighted by the data's own ``sigma``.
    """
    if mode == "two_component":
        return "far_lag_estimate"
    return "default_constant_0.01" if bool(getattr(result, "sigma_is_default", False)) else "data"


def _off_diagonal_mask(n_t: int) -> np.ndarray:
    m = np.ones((n_t, n_t), dtype=bool)
    m[0, :] = False
    m[:, 0] = False
    np.fill_diagonal(m, False)
    return m


def compute_fit_quality(
    result: OptimizationResult,
    data: dict[str, Any],
    config_manager: ConfigManager,
) -> dict[str, Any]:
    """Compute the NRMSE block for ``result`` against ``data``.

    Returns a JSON-safe dict with ``nrmse``, ``rms_residual``, ``c2_std``,
    ``n_valid``, ``n_angles_evaluated``, ``mask`` and ``sigma_source``. Angles
    whose model evaluation raises are skipped (counted in
    ``n_angles_failed``); if none evaluate, ``nrmse`` is ``NaN``.
    """
    # ponytail: the per-angle model evaluator + scaling-layout unpackers live in
    # viz (pulls in matplotlib). Moving them under optimization/ is the upgrade
    # path if a matplotlib-free fit worker ever matters.
    from xpcsjax.viz.nlsq_plots import _evaluate_c2_per_angle, _unpack_heterodyne_scaling

    cfg = config_manager.get_config()
    mode = str(cfg.get("analysis_mode", ""))
    model = config_manager.get_model()

    c2_exp = np.asarray(data["c2_exp"], dtype=np.float64)
    phi_angles = np.asarray(data["phi_angles_list"], dtype=np.float64).ravel()
    n_phi = int(min(c2_exp.shape[0], phi_angles.size))

    if mode == "two_component":
        # Evaluate on the FIT-TIME model: the heterodyne fit runs the stateful
        # HeterodyneModel whose time axis starts at t_start (= dt by default,
        # see heterodyne_model_stateful.from_config), synced to the data length
        # exactly as optimization/nlsq/__init__.py does. The viz evaluator
        # instead feeds the loader's t1 (starting at 0) to the adapter kernel,
        # a one-dt origin shift that is invisible on 1000-frame data but not on
        # short synthetic grids — the metric must not inherit it.
        from xpcsjax.core.heterodyne_model_stateful import HeterodyneModel

        hm = HeterodyneModel.from_config(config_manager.config)
        hm.sync_time_axis(np.arange(c2_exp.shape[1], dtype=np.float64))
        contrasts, offsets, physical, _ = _unpack_heterodyne_scaling(
            model, result, n_phi_expected=int(phi_angles.size)
        )
        full = hm.param_manager.expand_varying_to_full(np.asarray(physical, dtype=np.float64))

        def _c2_fit(i: int) -> np.ndarray:
            return np.asarray(
                hm.compute_correlation(
                    phi_angle=float(phi_angles[i]),
                    params=full,
                    contrast=float(contrasts[i]),
                    offset=float(offsets[i]),
                    angle_idx=i,
                ),
                dtype=np.float64,
            )

    else:

        def _c2_fit(i: int) -> np.ndarray:
            return np.asarray(
                _evaluate_c2_per_angle(model, result, data, cfg, float(phi_angles[i]), phi_index=i),
                dtype=np.float64,
            )

    ssr = 0.0
    s1 = 0.0  # sum of data over mask
    s2 = 0.0  # sum of squared data over mask
    n_valid = 0
    n_failed = 0
    for i in range(n_phi):
        try:
            c2_fit = _c2_fit(i)
        except Exception as exc:  # best-effort: one bad angle must not kill the metric
            n_failed += 1
            logger.debug("fit_quality: angle %d (phi=%.2f) failed: %s", i, phi_angles[i], exc)
            continue
        obs = c2_exp[i]
        if c2_fit.shape != obs.shape or obs.ndim != 2 or obs.shape[0] != obs.shape[1]:
            n_failed += 1
            continue
        mask = _off_diagonal_mask(obs.shape[0])
        r = obs - c2_fit
        mask &= np.isfinite(r)
        if not mask.any():
            continue
        o = obs[mask]
        ssr += float(np.sum(r[mask] ** 2))
        s1 += float(o.sum())
        s2 += float(np.sum(o * o))
        n_valid += int(mask.sum())

    if n_valid > 1:
        mean = s1 / n_valid
        var = max(s2 / n_valid - mean * mean, 0.0)
        c2_std = math.sqrt(var)
        rms = math.sqrt(ssr / n_valid)
        nrmse = rms / c2_std if c2_std > 0 else float("nan")
    else:
        c2_std = rms = nrmse = float("nan")

    block = {
        "nrmse": nrmse,
        "rms_residual": rms,
        "c2_std": c2_std,
        "n_valid": n_valid,
        "n_angles_evaluated": n_phi - n_failed,
        "n_angles_failed": n_failed,
        "mask": MASK_DESCRIPTION,
        "sigma_source": sigma_source(result, mode),
    }
    logger.info(
        "Fit quality (mode-agnostic): nrmse=%.4g (rms=%.4g / c2_std=%.4g, n_valid=%d); "
        "reduced_chi_squared=%.4g normalized by sigma_source=%s",
        nrmse,
        rms,
        c2_std,
        n_valid,
        float(result.reduced_chi_squared),
        block["sigma_source"],
    )
    return block


def attach_fit_quality(
    result: OptimizationResult,
    data: dict[str, Any],
    config_manager: ConfigManager,
) -> None:
    """Best-effort: store the NRMSE block under ``result.nlsq_diagnostics["fit_quality"]``.

    Never raises — a metric failure must not break a completed fit.
    """
    try:
        block = compute_fit_quality(result, data, config_manager)
    except Exception as exc:
        logger.warning("fit_quality: metric unavailable (%s: %s)", type(exc).__name__, exc)
        return
    if result.nlsq_diagnostics is None:
        result.nlsq_diagnostics = {}
    result.nlsq_diagnostics["fit_quality"] = block
