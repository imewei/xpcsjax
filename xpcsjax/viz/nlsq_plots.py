"""NLSQ fit visualization and artifact serialization.

Symbols defined here are wired into ``xpcsjax.viz``'s lazy export map by later
tasks (Task 2 onward).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from xpcsjax.utils.logging import get_logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = get_logger(__name__)


def _resolve_color_limits(
    matrix: np.ndarray,
    percentile_min: float = 1.0,
    percentile_max: float = 99.0,
) -> tuple[float, float]:
    """Percentile-based color limits with NaN/empty/flat fallbacks.

    Returns ``(1.0, 1.5)`` when the input is empty or all-NaN. Widens flat data
    to ``(vmin, vmin + 1.0)`` so matplotlib's imshow doesn't render a blank
    image with an invalid colorbar.
    """
    if matrix.size == 0 or not np.any(np.isfinite(matrix)):
        return 1.0, 1.5
    vmin = float(np.nanpercentile(matrix, percentile_min))
    vmax = float(np.nanpercentile(matrix, percentile_max))
    if not np.isfinite(vmin):
        vmin = 1.0
    if not np.isfinite(vmax):
        vmax = 1.5
    if vmin >= vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _save_fig(fig: Figure, save_path: Path | str | None, dpi: int = 150) -> None:
    """Save figure to disk and close. No-op when ``save_path`` is None.

    Creates parent directories as needed. Logs the saved path at INFO level.
    """
    if save_path is None:
        return
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", p)


def _unpack_result_params(
    model: Any,
    result: Any,
    config: dict[str, Any],
) -> tuple[float, float, np.ndarray, list[str]]:
    """Extract ``(contrast, offset, physical_params, names)`` per model type.

    HomodyneModel
        ``result.parameters[0]`` is contrast, ``[1]`` is offset, ``[2:]`` are the
        physical params. ``parameter_names`` excludes contrast/offset.

    HeterodyneModel
        ``contrast`` and ``offset`` are named slots inside the 14-element registry
        vector. ``physical_params`` is the full 14-element vector (the
        ``compute_g1`` API consumes the whole vector). ``parameter_names`` is the
        full 14-element registry-ordered name list.
    """
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.core.homodyne_model import HomodyneModel

    if isinstance(model, HomodyneModel):
        params = np.asarray(result.parameters, dtype=float)
        if params.size < 3:
            raise ValueError(
                f"HomodyneModel needs >=3 params (contrast, offset, physical...); got {params.size}"
            )
        # Resolve names from either the wrapper or its inner CombinedModel; no
        # hardcoded fallback — if upstream refactors so neither attribute exists,
        # the AttributeError surfaces the real bug instead of silently lying.
        names_obj = getattr(model, "parameter_names", None)
        if names_obj is None:
            inner = getattr(model, "model", None)
            names_obj = getattr(inner, "parameter_names", None)
        if names_obj is None:
            raise AttributeError(
                "HomodyneModel exposes no parameter_names (neither directly "
                "nor via .model). xpcsjax viz cannot label physical parameters."
            )
        full_names = list(names_obj)
        physical_params = params[2:].copy()
        # Slice names to match the actual physical-param count. In static mode
        # the inner CombinedModel has 3 names and physical_params is length 3;
        # in laminar_flow it has 7 names and physical_params is length 7 — so
        # the slice is a no-op in valid cases. The slice guards against a
        # length mismatch silently corrupting downstream labels.
        names = full_names[: physical_params.size]
        return float(params[0]), float(params[1]), physical_params, names

    if isinstance(model, HeterodyneModel):
        params = np.asarray(result.parameters, dtype=float)
        names = list(model.parameter_names)
        if params.size != len(names):
            raise ValueError(f"HeterodyneModel expects {len(names)} params; got {params.size}")
        if "contrast" in names and "offset" in names:
            c = float(params[names.index("contrast")])
            o = float(params[names.index("offset")])
        else:
            raise ValueError(
                "HeterodyneModel parameter_names registry is missing required "
                "'contrast' and/or 'offset' slots."
            )
        return c, o, params.copy(), names

    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        f"Expected HomodyneModel or HeterodyneModel."
    )


def _evaluate_c2_per_angle(
    model: Any,
    result: Any,
    data: dict[str, Any],
    config: dict[str, Any],
    phi_deg: float,
) -> np.ndarray:
    """Compute fitted c2 surface at one phi angle.

    Dispatches on model type:

    HomodyneModel
        Uses ``_unpack_result_params`` to extract contrast/offset/physical_params,
        then calls ``model.compute_c2_single_angle(physical_params, phi, contrast,
        offset)`` which uses the model's stored t-grid/q/L/dt state.

    HeterodyneModel
        Not yet wired up. ``HeterodyneModel.compute_g1`` returns g1² (range
        [0, 1]), not a fittable c2 surface. The real c2 reconstruction needs
        per-angle contrast/offset from
        ``xpcsjax.optimization.nlsq.heterodyne_scaling_utils`` whose formulas
        vary by analysis mode (constant/auto/fourier/individual). Out of scope
        for Task 5; raises ``NotImplementedError`` until a follow-up task
        wires it up. See plan spec amendment 3.
    """
    from xpcsjax.core.heterodyne_model import HeterodyneModel
    from xpcsjax.core.homodyne_model import HomodyneModel

    if isinstance(model, HomodyneModel):
        contrast, offset, physical_params, _ = _unpack_result_params(model, result, config)
        c2 = model.compute_c2_single_angle(physical_params, phi_deg, contrast, offset)
        return np.asarray(c2)

    if isinstance(model, HeterodyneModel):
        raise NotImplementedError(
            "Heterodyne c2 reconstruction in viz is not yet wired up. "
            "HeterodyneModel.compute_g1 returns g1² (range [0, 1]), not a "
            "fittable c2 surface — the real c2 needs per-angle contrast/offset "
            "from xpcsjax.optimization.nlsq.heterodyne_scaling_utils, with "
            "formulas that vary by analysis mode (constant/auto/fourier/individual). "
            "See plan spec amendment 3."
        )

    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        f"Expected HomodyneModel or HeterodyneModel."
    )
