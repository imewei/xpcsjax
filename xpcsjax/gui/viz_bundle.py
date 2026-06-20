"""Load the JAX-free interactive-plot data bundle from the fit's own artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile


@dataclass
class VizBundle:
    """Arrays needed for in-app interactive plots (numpy; no JAX)."""

    exp_c2: np.ndarray
    model_c2: np.ndarray | None = None
    residuals: np.ndarray | None = None
    t1: np.ndarray | None = None
    t2: np.ndarray | None = None
    phi_angles: np.ndarray | None = None


# The fit already writes every field we need (generate_nlsq_plots,
# nlsq_plots.py:1155-1162) — no separate viz_data.npz is produced.
_FITTED = ("plots", "simulated_data", "c2_fitted_data.npz")


def _get(npz: NpzFile, key: str) -> np.ndarray | None:
    return np.asarray(npz[key]) if key in npz.files else None


def load_viz_bundle(result_dir: str | Path) -> VizBundle | None:
    """Load the interactive-plot bundle from ``<result_dir>/plots/simulated_data/c2_fitted_data.npz``.

    Returns ``None`` if that artifact is absent/unreadable or has no ``c2_exp``.
    ``residuals`` is read directly (the fit already stored it, ``nlsq_plots.py:1158``);
    ``model_c2``/``residuals`` are left ``None`` when the fitted surface is absent
    (exp-only views). The GUI never recomputes — it stays JAX-free.
    """
    fitted_path = Path(result_dir).joinpath(*_FITTED)
    if not fitted_path.is_file():
        return None
    try:
        with np.load(fitted_path) as npz:
            exp = _get(npz, "c2_exp")
            if exp is None:
                return None
            model = _get(npz, "c2_fitted")
            residuals = _get(npz, "residuals")
            t1, t2, phi = _get(npz, "t1"), _get(npz, "t2"), _get(npz, "phi_angles")
    except (OSError, ValueError):
        return None

    if model is not None and model.shape != exp.shape:
        model, residuals = None, None
    if model is None:
        residuals = None  # no model surface -> exp-only views
    elif residuals is None:
        residuals = exp - model  # fall back to the viz convention if not stored

    return VizBundle(exp_c2=exp, model_c2=model, residuals=residuals, t1=t1, t2=t2, phi_angles=phi)
